import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import auc

from safe.cramer import gini_via_lorenz, cvm1_concordance_weighted
from safe.utils import apply_patch_occlusion, ensure_prob_matrix, get_predictions_from_features, apply_importance_masking


def rge_cramer(pred, pred_reduced):
    """
    RGE which compares original predictions with perturbed predictions.

    Parameters
    ----------
    pred : array-like
        Predictions from full model
    pred_reduced : array-like
        Predictions from reduced model

    Returns
    -------
    float
        RGE score
    """
    g = gini_via_lorenz(pred)
    if not np.isfinite(g) or g == 0:
        return np.nan
    cvm = cvm1_concordance_weighted(pred, pred_reduced)
    if not np.isfinite(cvm):
        return np.nan
    return cvm / g


def rge_cramer_multiclass(prob_full, prob_reduced, class_order=None, class_weights=None, verbose=False):
    """
    Calculate RGE for multiclass classification.
    Measures impact of feature removal/occlusion on predictions.
    Use align_proba_to_class_order() before calling this function.

    Parameters
    ----------
    prob_full : array-like, shape (n_samples, n_classes)
        Predictions from original model
    prob_reduced : array-like, shape (n_samples, n_classes)
        Predictions from occluded model
    class_order :
        Class order
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (rge_weighted, rge_per_class, weights_used)
        - rge_weighted: Overall weighted RGE score
        - rge_per_class: RGE score for each class
        - weights_used: Weights used for each class
    """
    prob_full = np.asarray(prob_full)
    prob_reduced = np.asarray(prob_reduced)

    if class_order is not None:
        class_order = np.asarray(class_order)
        prob_full = ensure_prob_matrix(prob_full, class_order)
        prob_reduced = ensure_prob_matrix(prob_reduced, class_order)
    else:
        if prob_full.ndim != 2 or prob_reduced.ndim != 2:
            raise ValueError('For 1D binary probabilities, pass class_order with 2 classes')

    n_samples, n_classes = prob_full.shape

    if prob_reduced.shape != prob_full.shape:
        raise ValueError(
            f'Shape mismatch: prob_full {prob_full.shape} and prob_reduced {prob_reduced.shape}'
        )

    # Set up class weights
    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes
    else:
        class_weights = np.asarray(class_weights)
        if len(class_weights) != n_classes:
            raise ValueError(
                f'class_weights length {len(class_weights)} does not match n_classes {n_classes}'
            )

    rges = []

    for k in range(n_classes):
        pred_full = prob_full[:, k]
        pred_reduced = prob_reduced[:, k]

        # RGE uses same computation as RGR
        rge_k = 1 - rge_cramer(pred_full, pred_reduced)
        rges.append(rge_k)

        if verbose:
            print(f'Class {k}: RGE = {rge_k:.4f}')

    rges = np.array(rges)

    # Weighted average
    rge_weighted = np.nansum(rges * class_weights) / np.nansum(class_weights)

    return rge_weighted, rges, class_weights


def evaluate_rge_multiclass_occlusion(
        model, preprocess_fn, images_dataset, removal_fractions,
        model_class_order, class_order,
        model_type='sklearn', device=None,
        patch_size=32, batch_size=64,
        class_weights=None, model_name='Model', rga_full=None,
        occlusion_method='random', patch_rankings=None, patch_meta=None,
        plot=True, fig_size=(10, 6), verbose=True,
        random_seed=None, mask_value=0.0, save_path=None
):
    """
    Evaluate RGE across increasing occlusion levels and compute AURGE.

    Parameters
    ----------
    model :
        The classifier to evaluate (sklearn or torch)
    preprocess_fn :
        Callable mapping images tensor (N,C,H,W) -> features ndarray (N,D) ready for `model`.
        This is where you typically call: feature extractor -> PCA -> scaler.
    images_dataset :
        Torch dataset yielding images and possibly labels
    removal_fractions :
        Fractions of image area to occlude in [0,1]
    model_class_order :
        Model's class order (e.g. sklearn model.classes_)
    class_order :
        Canonical class order
    model_type :
        'sklearn' or 'pytorch'
    device :
        Torch device
    patch_size :
        Patch size for random occlusion or importance patching
    batch_size :
        Batch size for loading dataset and for prediction
    class_weights :
        Optional weights for RGE aggregation
    model_name :
        Name used for logging and plots
    rga_full :
        If provided, RGE curve is rescaled by this value (required by SAFE)
    occlusion_method :
        'random' or 'gradcam_most'
    patch_rankings, patch_meta :
        Required when occlusion_method is gradcam_
    plot :
        Whether to plot the RGE curve
    fig_size :
        Figure size for plotting
    verbose :
        Verbose logging
    random_seed :
        Seed used for random occlusion
    mask_value :
        Fill value for masked pixels when using constant baseline
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Contains raw/rescaled RGE values, AURGE, per-class RGE, and metadata.
    """
    removal_fractions = np.asarray(removal_fractions, dtype=float)

    if occlusion_method in ('gradcam_most', 'gradcam_least'):
        if patch_rankings is None or patch_meta is None:
            raise ValueError('For Grad-CAM masking you must pass patch_rankings and patch_meta')

    if verbose:
        print(f'RGE Evaluation: {model_name}')
        print(f'Occlusion: {occlusion_method}')
        print(f'Testing {len(removal_fractions)} removal fractions')

    # Load all images once
    loader = DataLoader(images_dataset, batch_size=batch_size, shuffle=False)
    images_all = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        images_all.append(x)
    images_all = torch.cat(images_all, dim=0)

    _, _, h, w = images_all.shape
    total_pixels = h * w
    patch_pixels = patch_size * patch_size

    # Baseline predictions
    if verbose:
        print('Extracting features from original images...')
    feat_full = preprocess_fn(images_all)
    prob_full = get_predictions_from_features(
        feat_full, model, model_class_order, class_order,
        model_type=model_type, device=device, batch_size=batch_size
    )

    rge_scores = []
    per_class_rge_list = []

    for frac in removal_fractions:
        if verbose:
            print(f'\nOcclusion level: {frac * 100:.0f}%')

        if occlusion_method == 'random':
            pixels_to_remove = int(frac * total_pixels)
            num_patches = pixels_to_remove // patch_pixels
            images_occ = apply_patch_occlusion(
                images_all, num_patches, patch_size,
                random_seed=random_seed, mask_value=mask_value
            )

        elif occlusion_method == 'gradcam_most':
            images_occ = apply_importance_masking(
                images_all, patch_rankings, patch_meta, frac,
                mask_strategy='most_important', mask_value=mask_value
            )

        else:
            raise ValueError(f'Unknown occlusion_method: {occlusion_method}')

        feat_occ = preprocess_fn(images_occ)
        prob_occ = get_predictions_from_features(
            feat_occ, model, model_class_order, class_order,
            model_type=model_type, device=device, batch_size=batch_size
        )

        rge_val, rge_per_class, _ = rge_cramer_multiclass(prob_full, prob_occ, class_order=class_order, class_weights=class_weights)
        rge_val = 0.0 if np.isnan(rge_val) else float(rge_val)

        rge_scores.append(rge_val)
        per_class_rge_list.append(rge_per_class)

        if verbose:
            print(f'RGE = {rge_val:.4f}')

    rge_scores = np.asarray(rge_scores, dtype=float)
    per_class_rge_list = np.asarray(per_class_rge_list)

    # Rescale by RGA
    rge_rescaled = rge_scores * float(rga_full) if (
                rga_full is not None and np.isfinite(rga_full)) else rge_scores

    # AUC on normalized x-axis
    max_frac = float(np.max(removal_fractions)) if len(removal_fractions) else 1.0
    x = removal_fractions / max_frac if max_frac > 0 else removal_fractions
    aurge = auc(x, rge_rescaled)

    if verbose:
        print(f'AURGE: {aurge:.4f}')

    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(removal_fractions * 100, rge_rescaled, '-o', linewidth=2.5, markersize=6)
        plt.fill_between(removal_fractions * 100, 0, rge_rescaled, alpha=0.2)
        plt.xlabel('Occluded Image Area (%)', fontsize=11, fontweight='bold')
        plt.ylabel('RGE Score', fontsize=11, fontweight='bold')
        plt.title(f'RGE Curve: {model_name} ({occlusion_method})', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    return {
        'rge_scores': rge_scores,
        'rge_rescaled': rge_rescaled,
        'aurge': aurge,
        'removal_fractions': removal_fractions,
        'per_class_rge': per_class_rge_list,
        'class_order': class_order,
        'occlusion_method': occlusion_method,
    }


def compare_models_rge(
        models_dict, images_dataset, removal_fractions, class_order,
        occlusion_method='random',
        patch_size=32, batch_size=64, class_weights=None,
        rga_dict=None, device=None, fig_size=(12, 6), verbose=True,
        random_seed=None, patch_rankings=None, patch_meta=None, save_path=None
):
    """
    Evaluate and plot RGE curves for multiple models.

    Parameters
    ----------
    models_dict :
        Mapping model_name -> (model, preprocess_fn, model_class_order, model_type)
    images_dataset :
        Dataset images
    removal_fractions :
        Occlusion fractions in [0,1]
    class_order :
        Canonical class order.
    occlusion_method :
        Single method for all models OR per-model dict
    patch_size :
        Patch size for random occlusion or importance patching
    batch_size :
        Batch size for loading dataset and for prediction
    class_weights :
        Optional weights for RGE aggregation
    rga_dict :
        Needed for rescaling
    device :
        Torch device
    fig_size :
        Figure size for plotting
    verbose :
        Verbose logging
    random_seed :
        Seed used for random occlusion
    patch_rankings, patch_meta :
        Shared Grad-CAM patch ranking info (compute once) for gradcam_ methods
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Results per model name
    """
    if isinstance(occlusion_method, str):
        methods = {name: occlusion_method for name in models_dict}
    elif isinstance(occlusion_method, dict):
        methods = occlusion_method
    else:
        raise TypeError(
            'occlusion_method must be a string (single method) or a dict {model_name: method}.'
        )

    results = {}

    for name, (model, preprocess_fn, model_class_order, model_type) in models_dict.items():
        if verbose:
            print(f'\nEvaluating {name}')

        res = evaluate_rge_multiclass_occlusion(
            model=model,
            preprocess_fn=preprocess_fn,
            images_dataset=images_dataset,
            removal_fractions=removal_fractions,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            patch_size=patch_size,
            batch_size=batch_size,
            class_weights=class_weights,
            model_name=name,
            rga_full=(rga_dict.get(name) if rga_dict else None),
            occlusion_method=methods.get(name, 'random'),
            patch_rankings=patch_rankings,
            patch_meta=patch_meta,
            plot=False,
            verbose=verbose,
            random_seed=random_seed,
            save_path=save_path
        )
        results[name] = res

    # Plot comparison.
    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (name, res), col in zip(results.items(), colors):
        plt.plot(
            res['removal_fractions'] * 100,
            res['rge_rescaled'],
            '-o',
            linewidth=2.2,
            markersize=5,
            color=col,
            label=f"{name} [{res['occlusion_method']}] (AURGE={res['aurge']:.3f})",
        )

    plt.xlabel('Occluded Image Area (%)', fontsize=11, fontweight='bold')
    plt.ylabel('RGE Score', fontsize=11, fontweight='bold')
    plt.title('RGE Curves Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    if verbose:
        print('\nExplainability Comparison Summary (AURGE)')
        for name in results:
            print(f"{name:15s}: AURGE={results[name]['aurge']:.4f}")

    return results


def evaluate_rge_multiclass_text(
        model, x, removal_fractions, model_class_order, class_order,
        model_type='sklearn', device=None, batch_size=256,
        class_weights=None, model_name='Model', rga_full=None,
        masking_method='random',           # 'random' | 'most_important'
        feature_ranking=None,              # required for 'most_important' (array of feature indices, sorted desc)
        baseline='zero',                   # 'zero' | 'mean'
        plot=True, fig_size=(10, 6), verbose=True, random_seed=None, save_path=None, prob_full_cached=None
):
    """
    Text analogue of image occlusion RGE

    Returns
    -------
    dict with keys:
      - rge_scores, rge_rescaled, aurge
      - removal_fractions, per_class_rge
      - masking_method

    """
    removal_fractions = np.asarray(removal_fractions, dtype=float)
    x = np.asarray(x, dtype=float)
    n_samples, n_features = x.shape

    if masking_method == 'most_important' and feature_ranking is None:
        raise ValueError("feature_ranking is required when masking_method='most_important'")

    if baseline not in ('zero', 'mean'):
        raise ValueError(f"Unknown baseline: {baseline}. Use 'zero' or 'mean'.")

    if verbose:
        print(f'RGE (Text Feature Removal): {model_name}')
        print(f'Masking: {masking_method} | Baseline: {baseline}')
        print(f'x: {x.shape} | Testing {len(removal_fractions)} removal fractions')

    rng = np.random.RandomState(random_seed if random_seed is not None else 42)

    # Baseline feature values for "mean" replacement
    feat_mean = None
    if baseline == 'mean':
        feat_mean = np.nanmean(x, axis=0)

    # Baseline predictions (full, unmasked)
    if prob_full_cached is None:
        prob_full = get_predictions_from_features(
            x, model, model_class_order, class_order,
            model_type=model_type, device=device, batch_size=batch_size
        )
    else:
        prob_full = np.asarray(prob_full_cached)

    rge_scores = []
    per_class_rge_list = []

    for frac in removal_fractions:
        frac = float(frac)
        if frac < 0 or frac > 1:
            raise ValueError(f"removal fraction must be in [0,1], got {frac}")

        k = int(np.floor(frac * n_features))

        if verbose:
            print(f'\nRemoval level: {frac * 100:.0f}% | masking {k}/{n_features} features')

        # Prepare masked X
        x_masked = x.copy()

        if k > 0:
            if masking_method == 'random':
                cols = rng.choice(n_features, size=k, replace=False)
            elif masking_method == 'most_important':
                cols = np.asarray(feature_ranking, dtype=int)[:k]
            else:
                raise ValueError(f"Unknown masking_method: {masking_method}")

            if baseline == 'zero':
                x_masked[:, cols] = 0.0
            else:  # baseline == 'mean'
                x_masked[:, cols] = feat_mean[cols]

        # Reduced predictions
        prob_reduced = get_predictions_from_features(
            x_masked, model, model_class_order, class_order,
            model_type=model_type, device=device, batch_size=batch_size
        )

        rge_val, rge_per_class, _ = rge_cramer_multiclass(
            prob_full, prob_reduced, class_order=class_order, class_weights=class_weights, verbose=False
        )
        rge_val = 0.0 if np.isnan(rge_val) else float(rge_val)

        rge_scores.append(rge_val)
        per_class_rge_list.append(rge_per_class)

        if verbose:
            print(f'RGE = {rge_val:.4f}')

    rge_scores = np.asarray(rge_scores, dtype=float)
    per_class_rge_list = np.asarray(per_class_rge_list)

    # Rescale by RGA
    rge_rescaled = rge_scores * float(rga_full) if (rga_full is not None and np.isfinite(rga_full)) else rge_scores

    # AUC on normalized x-axis
    max_frac = float(np.max(removal_fractions)) if len(removal_fractions) else 1.0
    x = removal_fractions / max_frac if max_frac > 0 else removal_fractions
    aurge = auc(x, rge_rescaled)

    if verbose:
        print(f'AURGE: {aurge:.4f}')

    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(removal_fractions * 100, rge_rescaled, '-o', linewidth=2.5, markersize=6)
        plt.fill_between(removal_fractions * 100, 0, rge_rescaled, alpha=0.2)
        plt.xlabel('Removed Features (%)', fontsize=11, fontweight='bold')
        plt.ylabel('RGE Score', fontsize=11, fontweight='bold')
        plt.title(f'RGE Curve (Text): {model_name} ({masking_method})', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'rge_scores': rge_scores,
        'rge_rescaled': rge_rescaled,
        'aurge': aurge,
        'removal_fractions': removal_fractions,
        'per_class_rge': per_class_rge_list,
        'class_order': class_order,
        'masking_method': masking_method,
        'baseline': baseline,
    }


def compare_models_rge_text(
        models_dict,
        removal_fractions,
        class_order,
        masking_method='random',
        baseline='zero',
        class_weights=None,
        rga_dict=None,
        batch_size=256,
        fig_size=(12, 6),
        verbose=True,
        random_seed=None,
        save_path=None,
        feature_rankings=None,
):
    """
    Evaluate and plot RGE curves for multiple models on TEXT feature matrices.

    Expected models_dict format (same as you used in main text code):
        model_name -> (model, X, prob_full, class_order, model_type, device)

    Notes
    -----
    - prob_full can be None; if provided, it should already be aligned to class_order.
    - feature_rankings (optional) can be:
        - None (random masking for all)
        - dict model_name -> 1D ranking array (for most_important masking per model)
        - 1D ranking array (shared for all models)

    Returns
    -------
    dict: results per model name (same keys as evaluate_rge_multiclass_feature_removal()).
    """
    results = {}

    # Normalize feature rankings input
    if isinstance(feature_rankings, dict):
        rankings_map = feature_rankings
    elif feature_rankings is None:
        rankings_map = {}
    else:
        rankings_map = {name: feature_rankings for name in models_dict}

    # Compute per-model curves
    for name, tpl in models_dict.items():
        if len(tpl) != 6:
            raise ValueError(
                f"models_dict['{name}'] must be (model, X, prob_full, class_order, model_type, device). Got: {tpl}"
            )

        model, x, prob_full, model_class_order, model_type, device = tpl

        if verbose:
            print(f'\nEvaluating RGE (text) for {name}')

        res = evaluate_rge_multiclass_text(
            model=model,
            x=x,
            removal_fractions=removal_fractions,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            class_weights=class_weights,
            model_name=name,
            rga_full=(rga_dict.get(name) if rga_dict else None),
            masking_method=masking_method,
            feature_ranking=rankings_map.get(name),
            baseline=baseline,
            plot=False,
            fig_size=fig_size,
            verbose=verbose,
            random_seed=random_seed,
            save_path=None,
            prob_full_cached=prob_full,
        )
        results[name] = res

    # Plot comparison
    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (name, res), col in zip(results.items(), colors):
        plt.plot(
            res['removal_fractions'] * 100,
            res['rge_rescaled'],
            '-o',
            linewidth=2.2,
            markersize=5,
            color=col,
            label=f"{name} ({res['masking_method']}, AURGE={res['aurge']:.3f})",
        )

    plt.xlabel('Removed Features (%)', fontsize=11, fontweight='bold')
    plt.ylabel('RGE Score', fontsize=11, fontweight='bold')
    plt.title('RGE Curves Comparison (Text Feature Removal)', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print('\nExplainability Comparison Summary (AURGE)')
        for name in results:
            print(f"{name:15s}: AURGE={results[name]['aurge']:.4f}")

    return results


def evaluate_rge_multiclass_tabular(
    model, x, feature_names,
    model_class_order, class_order,
    model_type='sklearn', device=None,
    class_weights=None, model_name='Model', rga_full=None,
    masking_method='greedy',
    feature_ranking=None,
    baseline='zero',
    n_steps=None,
    random_seed=None,
    verbose=True,
    prob_full_cached=None,
):
    x = np.asarray(x, dtype=float)
    n_samples, n_features = x.shape
    if n_steps is None:
        n_steps = n_features

    rng = np.random.RandomState(random_seed if random_seed is not None else 42)

    # Baseline values for masking
    feat_mean = np.nanmean(x, axis=0) if baseline == 'mean' else None

    # Full predictions once
    if prob_full_cached is None:
        prob_full = get_predictions_from_features(
            x, model, model_class_order, class_order,
            model_type=model_type, device=device, batch_size=256
        )
    else:
        prob_full = np.asarray(prob_full_cached)

    removed = []
    remaining = list(range(n_features))

    rge_scores = [1.0]
    per_class_rge_list = []

    def mask_cols(x_in, cols):
        x_masked = x_in.copy()
        if len(cols) == 0:
            return x_masked
        if baseline == 'zero':
            x_masked[:, cols] = 0.0
        else:
            x_masked[:, cols] = feat_mean[cols]
        return x_masked

    for step in range(1, n_steps + 1):
        if verbose:
            print(f"[RGE-tabular] step {step}/{n_steps} | removed={len(removed)}")

        if masking_method == 'random':
            k = min(step, n_features)
            cols = rng.choice(n_features, size=k, replace=False)
            x_masked = mask_cols(x, cols)

            prob_reduced = get_predictions_from_features(
                x_masked, model, model_class_order, class_order,
                model_type=model_type, device=device, batch_size=256
            )
            rge_val, rge_per_class, _ = rge_cramer_multiclass(prob_full, prob_reduced, class_order=class_order, class_weights=class_weights)
            rge_scores.append(0.0 if np.isnan(rge_val) else float(rge_val))
            per_class_rge_list.append(rge_per_class)

        elif masking_method == 'most_important':
            if feature_ranking is None:
                raise ValueError("feature_ranking required for masking_method='most_important'")
            cols = np.asarray(feature_ranking, dtype=int)[:step]
            x_masked = mask_cols(x, cols)

            prob_reduced = get_predictions_from_features(
                x_masked, model, model_class_order, class_order,
                model_type=model_type, device=device, batch_size=256
            )
            rge_val, rge_per_class, _ = rge_cramer_multiclass(prob_full, prob_reduced, class_order=class_order, class_weights=class_weights)
            rge_scores.append(0.0 if np.isnan(rge_val) else float(rge_val))
            per_class_rge_list.append(rge_per_class)

        elif masking_method == 'greedy':
            best_j = None
            best_rge = -np.inf
            best_per_class = None

            for j in remaining:
                cols = removed + [j]
                x_masked = mask_cols(x, cols)

                prob_reduced = get_predictions_from_features(
                    x_masked, model, model_class_order, class_order,
                    model_type=model_type, device=device, batch_size=256
                )

                rge_val, rge_per_class, _ = rge_cramer_multiclass(
                    prob_full, prob_reduced, class_order=class_order, class_weights=class_weights
                )

                rge_val = -np.inf if np.isnan(rge_val) else float(rge_val)
                if rge_val > best_rge:
                    best_rge = rge_val
                    best_j = j
                    best_per_class = rge_per_class

            removed.append(best_j)
            remaining.remove(best_j)

            rge_scores.append(0.0 if not np.isfinite(best_rge) else float(best_rge))
            per_class_rge_list.append(best_per_class)

            if verbose:
                print(f"  picked: {feature_names[best_j]} | rge={rge_scores[-1]:.4f}")

        else:
            raise ValueError(f"Unknown masking_method: {masking_method}")

    rge_scores = np.asarray(rge_scores, dtype=float)

    x_axis = np.linspace(0, 1, len(rge_scores))

    rge_rescaled = rge_scores * float(rga_full) if (rga_full is not None and np.isfinite(rga_full)) else rge_scores
    aurge = auc(x_axis, rge_rescaled)

    return {
        "x_axis": x_axis,
        "rge_scores": rge_scores,
        "rge_rescaled": rge_rescaled,
        "aurge": aurge,
        "removed_features": [feature_names[i] for i in removed] if masking_method == "greedy" else None,
        "per_class_rge": np.asarray(per_class_rge_list) if len(per_class_rge_list) else None,
        "class_order": class_order,
        "masking_method": masking_method,
        "baseline": baseline,
    }


def compare_models_rge_tabular(
    models_dict,
    class_order,
    class_weights=None,
    rga_dict=None,
    masking_method='greedy',      # 'greedy'|'random'|'most_important'
    baseline='zero',
    n_steps=None,
    verbose=True,
    random_seed=None,
    fig_size=(12, 6),
    save_path=None,
    feature_rankings=None,        # None | dict(name->ranking) | ranking shared
):
    """
    models_dict format:
      model_name -> (model, X, feature_names, prob_full, model_class_order, model_type, device)

    - prob_full can be None; if provided it should already be aligned to class_order.
    - feature_rankings: optional ranking(s) for 'most_important'
    """
    results = {}

    # normalize feature_rankings input
    if isinstance(feature_rankings, dict):
        rankings_map = feature_rankings
    elif feature_rankings is None:
        rankings_map = {}
    else:
        rankings_map = {name: feature_rankings for name in models_dict}

    for name, tpl in models_dict.items():
        if len(tpl) != 7:
            raise ValueError(
                f"models_dict['{name}'] must be (model, X, feature_names, prob_full, model_class_order, model_type, device)."
            )
        model, x, feature_names, prob_full, model_class_order, model_type, device = tpl

        if verbose:
            print(f"\nEvaluating RGE (tabular) for {name}")

        res = evaluate_rge_multiclass_tabular(
            model=model,
            x=x,
            feature_names=feature_names,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            class_weights=class_weights,
            model_name=name,
            rga_full=(rga_dict.get(name) if rga_dict else None),
            masking_method=masking_method,
            feature_ranking=rankings_map.get(name),
            baseline=baseline,
            n_steps=n_steps,
            random_seed=random_seed,
            verbose=verbose,
            prob_full_cached=prob_full,
        )
        results[name] = res

    # plot comparison
    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (name, res), col in zip(results.items(), colors):
        plt.plot(
            res["x_axis"],
            res["rge_rescaled"],
            "-o",
            linewidth=2.2,
            markersize=5,
            color=col,
            label=f"{name} ({res['masking_method']}, AURGE={res['aurge']:.3f})",
        )

    plt.xlabel("Fraction of Features Removed", fontsize=11, fontweight="bold")
    plt.ylabel("RGE Score", fontsize=11, fontweight="bold")
    plt.title("RGE Curves Comparison (Tabular Feature Removal)", fontsize=12, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print("\nExplainability Comparison Summary (AURGE)")
        for name in results:
            print(f"{name:15s}: AURGE={results[name]['aurge']:.4f}")

    return results
