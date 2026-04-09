import numpy as np
import matplotlib.pyplot as plt

from safe.cramer import gini_via_lorenz, cvm1_concordance_weighted
from safe.utils import ensure_prob_matrix, fill_nan_tail, aurga_from_curve, ideal_prob_matrix


def rga_cramer(y, yhat):
    """
    RGA using Cramér–von Mises (CvM) distance
    RGA = 1 - CvM(y, yhat) / G(y)

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        RGA score
    """
    g = gini_via_lorenz(y)
    if not np.isfinite(g) or g == 0:
        return np.nan

    cvm = cvm1_concordance_weighted(y, yhat)
    if not np.isfinite(cvm):
        return np.nan

    return 1 - cvm / g


def partial_rga_cramer(y, yhat, n_segments):
    """
    Decompose RGA into partial contributions across segments.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values
    n_segments : int
        Number of segments to decompose into

    Returns
    -------
    dict
        Dictionary containing:
        - 'full_rga': RGA score
        - 'partial_rga': Partial RGA contributions for each segment
        - 'cumulative_vector': Cumulative vector [RGA, RGA-RGA_1, ..., 0]
        - 'segment_indices': List of index ranges for each segment
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    n = len(y)
    if n == 0:
        return {
            'full_rga': np.nan,
            'partial_rga': np.array([]),
            'cumulative_vector': np.array([]),
            'segment_indices': []
        }

    # Calculate full RGA
    full_rga = rga_cramer(y, yhat)
    full_gini = gini_via_lorenz(y)

    if not np.isfinite(full_rga) or not np.isfinite(full_gini) or full_gini == 0:
        return {
            'full_rga': full_rga,
            'partial_rga': np.array([np.nan] * n_segments),
            'cumulative_vector': np.array([np.nan] * (n_segments + 1)),
            'segment_indices': []
        }

    # Sort by predictions (descending)
    ord_yhat_desc = np.argsort(yhat)[::-1]
    y_sorted = y[ord_yhat_desc]
    yhat_sorted = yhat[ord_yhat_desc]

    # Divide into segments
    segment_size = n // n_segments
    remainder = n % n_segments

    partial_rga = []
    segment_indices = []

    start_idx = 0
    for k in range(n_segments):
        # Remainder across first segments
        current_size = segment_size + (1 if k < remainder else 0)
        end_idx = start_idx + current_size

        segment_indices.append((start_idx, end_idx))

        # Extract segment
        y_segment = y_sorted[start_idx:end_idx]
        yhat_segment = yhat_sorted[start_idx:end_idx]

        # Calculate RGA for this segment
        segment_rga = rga_cramer(y_segment, yhat_segment)

        # Weight by segment's contribution to total Gini
        segment_gini = gini_via_lorenz(y_segment)

        if np.isfinite(segment_gini) and segment_gini > 0:
            # Normalize by segment size relative to total
            weight = len(y_segment) / n
            weighted_contribution = segment_rga * segment_gini * weight / full_gini
        else:
            weighted_contribution = 0.0

        partial_rga.append(weighted_contribution)
        start_idx = end_idx

    partial_rga = np.array(partial_rga)

    # Normalize
    sum_partial = np.sum(partial_rga)
    if sum_partial > 0:
        partial_rga = partial_rga * (full_rga / sum_partial)

    # Build cumulative vector
    cumulative_vector = np.zeros(n_segments + 1)
    cumulative_vector[0] = full_rga

    cumsum = 0.0
    for k in range(n_segments):
        cumsum += partial_rga[k]
        cumulative_vector[k + 1] = full_rga - cumsum

    return {
        'full_rga': full_rga,
        'partial_rga': partial_rga,
        'cumulative_vector': cumulative_vector,
        'segment_indices': segment_indices
    }


def rga_cramer_multiclass(y_labels, prob_matrix, class_order=None, verbose=False):
    """
    Calculate RGA for multiclass classification using one-vs-rest approach.

    Parameters
    ----------
    y_labels : array-like
        True class labels
    prob_matrix : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class.
        Columns must correspond to `class_order` if provided,
        or to sorted unique classes in y_labels if not.
    class_order : array-like, optional
        Order of classes corresponding to prob_matrix columns (.classes_).
        If None, assumes prob_matrix columns match sorted unique(y_labels).
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (rga_weighted, rga_per_class, class_weights, classes_used)
        - rga_weighted: Overall weighted RGA score
        - rga_per_class: RGA score for each class
        - class_weights: Weight of each class
        - classes_used: The class order used for computation
    """
    y_labels = np.asarray(y_labels)

    # Determine class order
    if class_order is None:
        if verbose:
            print('WARNING: class_order is not provided. Assuming prob_matrix columns match sorted unique classes.')
        class_order = np.unique(y_labels)
    else:
        class_order = np.asarray(class_order)

    prob_matrix = ensure_prob_matrix(prob_matrix, class_order)

    n_classes = len(class_order)

    # Validate dimensions
    if prob_matrix.shape[1] != n_classes:
        raise ValueError(
            f'prob_matrix has {prob_matrix.shape[1]} columns but class_order has {n_classes} classes.'
        )

    rgas = []
    weights = []

    for k, c in enumerate(class_order):
        # One-vs-rest encoding
        y_bin = np.equal(y_labels, c).astype(np.float32)
        yhat_c = prob_matrix[:, k]

        if np.sum(y_bin) == 0:
            if verbose:
                print(f'Warning: Class {c} has zero samples. Skipping.')
            rgas.append(0.0)
            weights.append(0.0)
            continue

        rga_k = rga_cramer(y_bin, yhat_c)
        rgas.append(rga_k)
        weights.append(np.mean(y_bin))

    rgas = np.array(rgas)
    weights = np.array(weights)

    # Weighted average
    denom = np.nansum(weights)
    rga_weighted = np.nansum(rgas * weights) / denom if denom > 0 else np.nan

    return rga_weighted, rgas, weights, class_order


def rga_curve_multiclass(y_labels, prob_matrix, class_order, n_segments=10):
    """
    Removes top-x most confident samples within each class and
    recomputes multiclass OvR RGA on the union of remaining samples
    """
    y_labels = np.asarray(y_labels)
    classes = np.asarray(class_order)
    p = ensure_prob_matrix(prob_matrix, classes)

    x_axis = np.linspace(0, 1, n_segments + 1)
    curve = np.zeros_like(x_axis, dtype=float)

    col_of_class = {int(c): int(k) for k, c in enumerate(classes)}
    idx_by_class = {int(c): np.where(y_labels == c)[0] for c in classes}

    for i, frac in enumerate(x_axis):
        keep_all = []

        for c in classes:
            c_int = int(c)
            idx_c = idx_by_class[c_int]
            n_c = len(idx_c)
            if n_c == 0:
                continue

            k = col_of_class[c_int]
            conf_c = p[idx_c, k]

            order_c = idx_c[np.lexsort((idx_c, -conf_c))]

            m_c = int(np.floor(frac * n_c))
            keep_c = order_c[m_c:]
            keep_all.append(keep_c)

        keep = np.concatenate(keep_all) if keep_all else np.array([], dtype=int)

        if len(keep) < 2:
            curve[i] = 0.0
            continue

        y_rem = y_labels[keep]
        p_rem = p[keep, :]

        rga_full, _, _, _ = rga_cramer_multiclass(
            y_rem, p_rem, class_order=classes, verbose=False
        )
        curve[i] = 0.0 if not np.isfinite(rga_full) else float(rga_full)

    curve = fill_nan_tail(curve)
    aurga_val = aurga_from_curve(curve)
    return x_axis, curve, aurga_val


def partial_rga_cramer_multiclass(y_labels, prob_matrix, n_segments, class_order=None, verbose=False):
    """
    Calculate partial RGA curves for multiclass classification.

    Parameters
    ----------
    y_labels : array-like
        True class labels
    prob_matrix : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class.
    n_segments : int
        Number of segments for partial decomposition
    class_order : array-like, optional
        Order of classes corresponding to prob_matrix columns.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    dict
        Dictionary containing:
        - 'cumulative_vector': Weighted average cumulative vector
        - 'per_class_vectors': Cumulative vectors for each class
        - 'class_weights': Weight of each class
        - 'classes': Class order used
    """
    y_labels = np.asarray(y_labels)

    # Determine class order
    if class_order is None:
        if verbose:
            print('WARNING: class_order is not provided. Assuming prob_matrix columns match sorted unique classes.')
        class_order = np.unique(y_labels)
    else:
        class_order = np.asarray(class_order)

    prob_matrix = ensure_prob_matrix(prob_matrix, class_order)

    cum_vectors = []
    class_weights = []

    for k, c in enumerate(class_order):
        y_bin = np.equal(y_labels, c).astype(np.float32)
        yhat_c = prob_matrix[:, k]

        res = partial_rga_cramer(y_bin, yhat_c, n_segments)
        cum_vectors.append(res['cumulative_vector'])
        class_weights.append(np.mean(y_bin))

    cum_vectors = np.vstack(cum_vectors)
    class_weights = np.array(class_weights)

    # Weighted average across classes
    weighted_curve = np.average(cum_vectors, weights=class_weights, axis=0)

    return {
        'cumulative_vector': weighted_curve,
        'per_class_vectors': cum_vectors,
        'class_weights': class_weights,
        'classes': class_order
    }


# Evaluation Function
def evaluate_rga_multiclass(y_labels, prob_matrix, class_order=None, n_segments=10,
                            model_name='Model', plot=True, fig_size=(12, 5),
                            verbose=True, save_path=None):
    """
    RGA evaluation for multiclass classification.

    Parameters
    ----------
    y_labels : array-like
        True class labels
    prob_matrix : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class
    class_order : array-like, optional
        Order of classes corresponding to prob_matrix columns.
        For sklearn models, pass `model.classes_`.
        For PyTorch models, pass the class order used in output layer, like np.array([0, 1, 2, ...]).
    n_segments : int, optional
        Number of segments for partial RGA decomposition
    model_name : str, optional
        Name of the model for display
    plot : bool, optional
        Whether to generate visualization
    fig_size : tuple, optional
        Figure size for plots
    verbose : bool, optional
        Print detailed results
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Comprehensive results dictionary containing:
        - 'rga_full': Overall RGA score
        - 'rga_per_class': RGA for each class
        - 'class_weights': Weight of each class
        - 'aurga': Area under RGA curve
        - 'cumulative_vector': Cumulative RGA vector
        - 'per_class_vectors': Per-class cumulative vectors
        - 'classes': Class order used
    """
    # Scalar RGA at full data (same as old)
    rga_full, rga_per_class, class_weights, classes_used = rga_cramer_multiclass(
        y_labels, prob_matrix, class_order=class_order, verbose=verbose
    )

    # Model curve (new)
    x_axis, curve_model, aurga_model = rga_curve_multiclass(
        y_labels, prob_matrix, classes_used, n_segments=n_segments
    )

    # Perfect baseline
    p_ideal = ideal_prob_matrix(y_labels, classes_used)
    _, curve_perfect, aurga_perfect = rga_curve_multiclass(
        y_labels, p_ideal, classes_used, n_segments=n_segments
    )

    aurga_norm = aurga_model / aurga_perfect if (np.isfinite(aurga_perfect) and aurga_perfect > 0) else np.nan

    if verbose:
        print(f"RGA Evaluation: {model_name}")
        print(f"Full RGA: {rga_full:.4f}")
        print(f"AURGA (new): {aurga_model:.4f}")
        print(f"AURGA_perfect: {aurga_perfect:.4f}")
        print(f"AURGA_normalized_to_perfect: {aurga_norm:.4f}")
        print(f"\nClass order: {classes_used}")
        print("\nPer-Class RGA:")
        for cls, rga_val, w in zip(classes_used, rga_per_class, class_weights):
            print(f"Class {cls}: RGA={rga_val:.4f}, Weight={w:.4f}")

    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(x_axis, curve_model, "-o", linewidth=2.5, markersize=5,
                 label=f"{model_name} (nAURGA={aurga_norm:.3f})")
        plt.plot(x_axis, curve_perfect, "--", linewidth=2.0, label='Perfect')

        plt.xlabel('Fraction of Data Removed', fontsize=11, fontweight='bold')
        plt.ylabel('RGA Score', fontsize=11, fontweight='bold')
        plt.title('RGA Curve', fontsize=12, fontweight="bold")
        plt.grid(alpha=0.3, linestyle="--")
        plt.xlim([0, 1])
        ymax = np.nanmax([np.nanmax(curve_model), np.nanmax(curve_perfect)])
        plt.ylim([0, ymax * 1.1 if np.isfinite(ymax) else 1])
        plt.legend(fontsize=9)
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "rga_full": rga_full,
        "rga_per_class": rga_per_class,
        "class_weights": class_weights,
        "classes": classes_used,
        "x_axis": x_axis,
        "curve_model": curve_model,
        "curve_perfect": curve_perfect,
        "aurga": aurga_model,
        "aurga_perfect": aurga_perfect,
        "aurga_normalized_to_perfect": aurga_norm
    }


def compare_models_rga(models_dict, y_labels, n_segments=10,
                        fig_size=(14, 6), verbose=True, save_path=None):
    """
    Compare multiple models using RGA metrics.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to tuples of (prob_matrix, class_order).
        Example: {
            'Random Forest': (rf.predict_proba(x_test), rf.classes_),
            'Neural Network': (nn_probs, np.array([0, 1, 2]))
        }
    y_labels : array-like
        True class labels
    n_segments : int, optional
        Number of segments for partial RGA
    fig_size : tuple, optional
        Figure size for comparison plot
    verbose : bool, optional
        Print detailed comparison
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Comparison results for all models

    """
    results = {}

    for model_name, (p, classes) in models_dict.items():
        if verbose:
            print(f"\nEvaluating {model_name}...")

        res = evaluate_rga_multiclass(
            y_labels=y_labels,
            prob_matrix=p,
            class_order=classes,
            n_segments=n_segments,
            model_name=model_name,
            plot=False,
            verbose=verbose,
            save_path=None
        )
        results[model_name] = res

    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (model_name, res), color in zip(results.items(), colors):
        plt.plot(
            res['x_axis'], res['curve_model'], '-o',
            linewidth=2.3, markersize=4.5, color=color,
            label=f"{model_name} (nAURGA={res['aurga_normalized_to_perfect']:.3f})"
        )

    plt.xlabel('Fraction of Data Removed', fontsize=11, fontweight='bold')
    plt.ylabel('RGA Score', fontsize=11, fontweight='bold')
    plt.title('RGA Curves Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle="--")
    plt.xlim([0, 1])
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    if verbose:
        print("\nRGA Comparison Summary")
        for name, res in results.items():
            print(
                f"{name}: RGA={res['rga_full']:.4f}, "
                f"AURGA={res['aurga']:.4f}, "
                f"nAURGA={res['aurga_normalized_to_perfect']:.4f}"
            )

    return results