import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import auc
from safe.cramer import gini_via_lorenz, cvm1_concordance_weighted
from safe.utils import align_proba_to_class_order, ensure_prob_matrix


def rgr_cramer(pred, pred_pert):
    """
    RGR which compares original predictions with perturbed predictions.

    Parameters
    ----------
    pred : array-like
        Original predictions
    pred_pert : array-like
        Perturbed predictions

    Returns
    -------
    float
        RGR score
    """
    g = gini_via_lorenz(pred)
    if not np.isfinite(g) or g == 0:
        return np.nan
    cvm = cvm1_concordance_weighted(pred, pred_pert)
    if not np.isfinite(cvm):
        return np.nan
    return 1 - cvm / g


def rgr_cramer_multiclass(prob_original, prob_perturbed, class_order=None, class_weights=None, verbose=False):
    """
    Calculate RGR for multiclass classification.

    Parameters
    ----------
    prob_original : array-like, shape (n_samples, n_classes)
        Original predicted probabilities
    prob_perturbed : array-like, shape (n_samples, n_classes)
        Perturbed predicted probabilities
    class_order :
        Class order
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (rgr_weighted, rgr_per_class, weights_used)
        - rgr_weighted: Overall weighted RGR score
        - rgr_per_class: RGR score for each class
        - weights_used: Weights used for each class
    """
    prob_original = np.asarray(prob_original)
    prob_perturbed = np.asarray(prob_perturbed)

    if class_order is not None:
        class_order = np.asarray(class_order)
        prob_original = ensure_prob_matrix(prob_original, class_order)
        prob_perturbed = ensure_prob_matrix(prob_perturbed, class_order)
    else:
        if prob_original.ndim != 2 or prob_perturbed.ndim != 2:
            raise ValueError('For 1D binary probabilities, pass class_order with 2 classes')

    n_samples, n_classes = prob_original.shape

    if prob_perturbed.shape != prob_original.shape:
        raise ValueError(
            f'Shape mismatch: prob_original {prob_original.shape} and prob_perturbed {prob_perturbed.shape}'
        )

    # Set up class weights
    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes
    else:
        class_weights = np.asarray(class_weights)
        if len(class_weights) != n_classes:
            raise ValueError(
                f'Class_weights length {len(class_weights)} does not match n_classes {n_classes}'
            )

    rgrs = []

    for k in range(n_classes):
        pred_orig = prob_original[:, k]
        pred_pert = prob_perturbed[:, k]

        rgr_k = rgr_cramer(pred_orig, pred_pert)
        rgrs.append(rgr_k)

        if verbose:
            print(f'Class {k}: RGR = {rgr_k:.4f}')

    rgrs = np.array(rgrs)

    # Weighted average
    rgr_weighted = np.nansum(rgrs * class_weights) / np.nansum(class_weights)

    return rgr_weighted, rgrs, class_weights


def evaluate_rgr_multiclass_noise(model, x_data, prob_original, noise_levels,
                                   model_class_order, class_order,
                                   class_weights=None, model_type='sklearn',
                                   device=None, rga_full=None, model_name="Model",
                                   plot=True, fig_size=(10, 6), verbose=True,
                                   random_seed=None, save_path=None):
    """
    Evaluate RGR robustness for multiclass classification with noise perturbation.

    Parameters
    ----------
    model : sklearn model or PyTorch model
        Trained model to evaluate
    x_data : array-like or torch.Tensor
        Input features
    prob_original : array-like, shape (n_samples, n_classes)
        Original predicted probabilities (columns in model_class_order)
    noise_levels : array-like
        Standard deviations of Gaussian noise to test
    model_class_order : array-like
        Order of classes in model's output (e.g., model.classes_ for sklearn)
    class_order : array-like
        Target class order for alignment (shared across all models in comparison)
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    model_type : {'sklearn', 'pytorch'}, optional
        Type of model
    device : torch.device, optional
        Device for PyTorch models
    rga_full : float, optional
        Full RGA score for rescaling. If None, no rescaling is applied.
    model_name : str, optional
        Name of model for display
    plot : bool, optional
        Whether to generate visualization
    fig_size : tuple, optional
        Figure size for plot
    verbose : bool, optional
        Print detailed results
    random_seed : int, optional
        Random seed for reproducibility
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Dictionary containing:
        - 'rgr_scores': RGR scores at each noise level
        - 'rgr_rescaled': Rescaled RGR scores (if rga_full provided)
        - 'aurgr': Area under RGR curve
        - 'noise_levels': Noise levels tested
        - 'per_class_rgr': Per-class RGR at each noise level
        - 'class_order': Class order used
    """
    prob_original = np.asarray(prob_original)
    noise_levels = np.asarray(noise_levels)
    model_class_order = np.asarray(model_class_order)
    class_order = np.asarray(class_order)

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_seed)

    # Align prob_original to target class_order at start
    prob_original_aligned = align_proba_to_class_order(
        prob_original, model_class_order, class_order
    )

    n_samples, n_classes = prob_original_aligned.shape

    # Set up class weights
    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes

    # Convert PyTorch tensor once if needed
    if model_type == 'pytorch':
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data, dtype=torch.float32)

    rgr_scores = []
    per_class_rgr_list = []

    if verbose:
        print(f'RGR Evaluation: {model_name}')
        print(f'Testing {len(noise_levels)} noise levels')

    for i, sigma_val in enumerate(noise_levels):
        sigma = float(sigma_val.item() if hasattr(sigma_val, 'item') else sigma_val)

        if model_type == 'sklearn':
            # Add Gaussian noise to features
            noise = rng.normal(0, sigma, size=x_data.shape)
            x_noisy = x_data + noise

            # Get predictions and align to target class order
            prob_perturbed_raw = model.predict_proba(x_noisy)
            prob_perturbed = align_proba_to_class_order(
                prob_perturbed_raw, model_class_order, class_order
            )

        elif model_type == 'pytorch':
            # Add Gaussian noise to tensor
            noise = torch.randn_like(x_data) * sigma
            x_noisy = x_data + noise

            # Get predictions and align to target class order
            with torch.no_grad():
                logits = model(x_noisy.to(device))
                prob_perturbed_raw = torch.softmax(logits, dim=1).cpu().numpy()

            prob_perturbed = align_proba_to_class_order(
                prob_perturbed_raw, model_class_order, class_order
            )
        else:
            raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got '{model_type}'")

        # Calculate RGR
        rgr_val, rgr_per_class, _ = rgr_cramer_multiclass(
            prob_original_aligned,
            prob_perturbed,
            class_order=class_order,
            class_weights=class_weights,
            verbose=False
        )

        rgr_scores.append(0.0 if np.isnan(rgr_val) else rgr_val)
        per_class_rgr_list.append(rgr_per_class)

        if verbose and i % max(1, len(noise_levels) // 5) == 0:
            print(f"σ = {sigma:.3f}: RGR = {rgr_scores[-1]:.4f}")

    rgr_scores = np.array(rgr_scores)
    per_class_rgr_list = np.array(per_class_rgr_list)

    # Rescale by full RGA if provided
    if rga_full is not None:
        rgr_rescaled = rgr_scores * rga_full
    else:
        rgr_rescaled = rgr_scores

    # Normalize noise for AUC calculation
    max_noise = np.max(noise_levels)
    noise_norm = noise_levels / max_noise if max_noise > 0 else noise_levels

    # Calculate AURGR
    aurgr = auc(noise_norm, rgr_rescaled)

    if verbose:
        print(f'AURGR: {aurgr:.4f}')

    # Visualization
    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(noise_levels * 100, rgr_rescaled, '-o', linewidth=2.5,
                 markersize=6, color='steelblue',
                 label=f'{model_name} (AURGR={aurgr:.3f})')
        plt.fill_between(noise_levels * 100, 0, rgr_rescaled,
                         alpha=0.2, color='steelblue')
        plt.xlabel('Noise Standard Deviation', fontsize=11, fontweight='bold')
        plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
        plt.title(f'RGR Curve: {model_name}', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlim([0, noise_levels[-1] * 100])
        plt.ylim([0, max(rgr_rescaled) * 1.1 if max(rgr_rescaled) > 0 else 1])
        plt.legend(fontsize=10)
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    return {
        'rgr_scores': rgr_scores,
        'rgr_rescaled': rgr_rescaled,
        'aurgr': aurgr,
        'noise_levels': noise_levels,
        'per_class_rgr': per_class_rgr_list,
        'class_order': class_order
    }


def compare_models_rgr(models_dict, noise_levels, class_order,
                        rga_dict=None, class_weights=None,
                        fig_size=(12, 6), verbose=True, random_seed=None, save_path=None):
    """
    Compare robustness of multiple models using RGR metrics.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to tuples of:
        (model, X_data, prob_original, model_class_order, model_type, device)

        Example: {
            'RF': (rf_model, X_test, prob_rf, rf.classes_, 'sklearn', None),
            'VQC': (vqc_model, X_tensor, prob_vqc, np.array([0,1,2]), 'pytorch', device)
        }
    noise_levels : array-like
        Standard deviations of noise to test
    class_order : array-like
        Target class order for alignment (shared across all models)
    rga_dict : dict, optional
        Dictionary mapping model names to RGA scores for rescaling
    class_weights : array-like, optional
        Class weights for all models
    fig_size : tuple, optional
        Figure size for comparison plot
    verbose : bool, optional
        Print detailed results
    random_seed : int, optional
        Random seed for reproducibility
    save_path:
        Path for saving the plot

    Returns
    -------
    dict
        RGR evaluation results for all models
    """
    results = {}

    for model_name, model_config in models_dict.items():
        model, x_data, prob_original, model_class_order, model_type, device = model_config
        rga_full = rga_dict.get(model_name) if rga_dict else None

        if verbose:
            print(f'\nEvaluating {model_name}...')

        result = evaluate_rgr_multiclass_noise(
            model=model,
            x_data=x_data,
            prob_original=prob_original,
            noise_levels=noise_levels,
            model_class_order=model_class_order,
            class_order=class_order,
            class_weights=class_weights,
            model_type=model_type,
            device=device,
            rga_full=rga_full,
            model_name=model_name,
            plot=False,
            verbose=verbose,
            random_seed=random_seed,
            save_path=save_path
        )
        results[model_name] = result

    model_names = list(results.keys())
    aurgr_scores = np.array([results[name]['aurgr'] for name in model_names], dtype=float)

    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (model_name, result), color in zip(results.items(), colors):
        plt.plot(
            result['noise_levels'] * 100,
            result['rgr_rescaled'],
            '-o',
            linewidth=2.5,
            markersize=5,
            color=color,
            label=f"{model_name} (AURGR={result['aurgr']:.3f})"
        )

    plt.xlabel('Noise Standard Deviation', fontsize=11, fontweight='bold')
    plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
    plt.title('RGR Curves Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlim([0, float(np.max(noise_levels)) * 100])
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    if verbose:
        print('Robustness Comparison Summary')
        for name, score in zip(model_names, aurgr_scores):
            print(f'{name}: AURGR = {score:.4f}')

        if len(model_names) >= 2:
            best_idx = int(np.nanargmax(aurgr_scores))
            worst_idx = int(np.nanargmin(aurgr_scores))

            best = aurgr_scores[best_idx]
            worst = aurgr_scores[worst_idx]

            print(f'Best: {model_names[best_idx]} (AURGR={best:.4f})')
            print(f'Worst: {model_names[worst_idx]} (AURGR={worst:.4f})')

    return results