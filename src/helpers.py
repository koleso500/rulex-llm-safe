import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_std(vals):
    arr = np.asarray(vals, float)
    return (
        float(arr.mean()) if len(arr) else float('nan'),
        float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    )


def print_standard_summary(results, model_names, n_splits):
    print(f'\n================ OVERALL {n_splits}-FOLD (STANDARD) ================')

    for model_name in model_names:
        acc_mu, acc_sd = mean_std(results[model_name]['acc'])
        f1_mu, f1_sd = mean_std(results[model_name]['f1'])
        mse_mu, mse_sd = mean_std(results[model_name]['mse'])

        print(
            f'{model_name:>6} | '
            f'ACC={acc_mu:.4f}±{acc_sd:.4f} '
            f'F1={f1_mu:.4f}±{f1_sd:.4f} '
            f'MSE={mse_mu:.6f}±{mse_sd:.6f}'
        )


def plot_mean_curve(
    x,
    safe_store,
    curve_key,
    title,
    x_label,
    y_label,
    save_path,
    model_names,
):
    plt.figure(figsize=(8, 6))

    for model_name in model_names:
        curves = safe_store[model_name][curve_key]
        arr = np.stack(curves, axis=0)
        mu = arr.mean(axis=0)
        plt.plot(x, mu, label=model_name)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_safe_summary(results, safe_store, model_names, csv_path):
    print('\nSAFE summary (mean ± std across folds):')

    rows = []

    for model_name in model_names:
        rga_mu, rga_sd = mean_std(safe_store[model_name]['rga_full'])
        aurga_mu, aurga_sd = mean_std(safe_store[model_name]['aurga'])
        aurgr_mu, aurgr_sd = mean_std(safe_store[model_name]['aurgr'])
        aurge_mu, aurge_sd = mean_std(safe_store[model_name]['aurge'])

        acc_mu, acc_sd = mean_std(results[model_name]['acc'])
        f1_mu, f1_sd = mean_std(results[model_name]['f1'])
        mse_mu, mse_sd = mean_std(results[model_name]['mse'])

        print(
            f'{model_name:>6} | '
            f'RGA={rga_mu:.4f}±{rga_sd:.4f} | '
            f'AURGA={aurga_mu:.4f}±{aurga_sd:.4f} | '
            f'AURGR={aurgr_mu:.4f}±{aurgr_sd:.4f} | '
            f'AURGE={aurge_mu:.4f}±{aurge_sd:.4f}'
        )

        rows.append({
            'model': model_name,
            'ACC_mean': acc_mu,
            'ACC_std': acc_sd,
            'F1_mean': f1_mu,
            'F1_std': f1_sd,
            'MSE_mean': mse_mu,
            'MSE_std': mse_sd,
            'RGA_mean': rga_mu,
            'RGA_std': rga_sd,
            'AURGA_mean': aurga_mu,
            'AURGA_std': aurga_sd,
            'AURGR_mean': aurgr_mu,
            'AURGR_std': aurgr_sd,
            'AURGE_mean': aurge_mu,
            'AURGE_std': aurge_sd,
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'Summary saved to: {csv_path}')