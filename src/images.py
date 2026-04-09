import copy
import json
import random
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

from helpers import print_standard_summary, plot_mean_curve, save_safe_summary
from models import (
    MODEL_PARAM_FUNCTIONS,
    build_mlp,
    create_sklearn_model,
    train_torch_model,
)

from safe.rga import compare_models_rga
from safe.rge import compare_models_rge
from safe.rgr import compare_models_rgr
from safe.utils import (
    CroppedImage,
    align_proba_to_class_order,
    compute_gradcam_maps,
    crop_img,
    extract_features_from_images,
    precompute_patch_rankings,
    show_heatmap_per_class,
    show_occlusions_same_idx,
    train_cam_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
config_path = PROJECT_ROOT / 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

GLOBAL_CFG = config.get('global', {})
PATHS_CFG = config['paths']
IMAGE_CFG = config['image']

SEED = int(IMAGE_CFG.get('seed', GLOBAL_CFG.get('seed', 42)))
N_SPLITS = int(IMAGE_CFG.get('n_splits', GLOBAL_CFG.get('n_splits', 5)))

DATASET_CFG = IMAGE_CFG['dataset']
OPTUNA_CFG = IMAGE_CFG['optuna']
TRAIN_CFG = IMAGE_CFG['training']
SAFE_CFG = IMAGE_CFG['safe']
GRADCAM_CFG = IMAGE_CFG['gradcam']
MODELS_CFG = IMAGE_CFG['models']
ENSEMBLES_CFG = IMAGE_CFG['ensembles']

DATA_DIR = Path(DATASET_CFG['data_dir'])
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR

CSV_DIR = PROJECT_ROOT / PATHS_CFG['csv_dir']
FIG_DIR = PROJECT_ROOT / PATHS_CFG['fig_image_dir']

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f'Dataset not found at {DATA_DIR}. Place the dataset to the configured data_dir.'
    )

IMG_SIZE = int(DATASET_CFG['img_size'])
BATCH_SIZE_IMAGES = int(DATASET_CFG['batch_size_images'])
FOLD_EXPORT_PREFIX = DATASET_CFG.get('fold_export_prefix', 'brain')

OPTUNA_TRIALS = int(OPTUNA_CFG['trials'])
OPTUNA_CV_FOLDS = int(OPTUNA_CFG['cv_folds'])
TUNE_MODELS = list(OPTUNA_CFG['tuned_models'])

optuna_json_path = CSV_DIR / OPTUNA_CFG['files']['best_params_json']
optuna_csv_path = CSV_DIR / OPTUNA_CFG['files']['best_params_csv']

EPOCHS = int(TRAIN_CFG['epochs'])
BATCH_SIZE_TRAIN = int(TRAIN_CFG['batch_size'])
LR_TORCH = float(TRAIN_CFG['learning_rate'])

N_SEGMENTS = int(SAFE_CFG['n_segments'])
FIG_SIZE = tuple(SAFE_CFG.get('fig_size', [8, 6]))
SAFE_VERBOSE = bool(SAFE_CFG.get('verbose', True))
BATCH_SIZE_SAFE = int(SAFE_CFG['batch_size_safe'])

PATCH_SIZE = int(SAFE_CFG['rge']['patch_size'])
OCCLUSION_METHOD = SAFE_CFG['rge'].get('occlusion_method', 'gradcam_most')

CAM_EPOCHS = int(GRADCAM_CFG['epochs'])
CAM_LR = float(GRADCAM_CFG['learning_rate'])
CAM_BATCH_SIZE = int(GRADCAM_CFG['batch_size'])

MODEL_NAMES = list(MODELS_CFG['model_names'])

device_name = TRAIN_CFG.get('device', 'auto')
device = torch.device('cuda' if device_name == 'auto' and torch.cuda.is_available() else device_name)
print('Using device:', device)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def make_grid(start: float, end: float, step: float) -> np.ndarray:
    n = int(round((end - start) / step)) + 1
    return start + step * np.arange(n, dtype=float)


noise_levels = make_grid(
    float(SAFE_CFG['rgr']['noise_start']),
    float(SAFE_CFG['rgr']['noise_end']),
    float(SAFE_CFG['rgr']['noise_step']),
)

removal_fractions = make_grid(
    float(SAFE_CFG['rge']['removal_start']),
    float(SAFE_CFG['rge']['removal_end']),
    float(SAFE_CFG['rge']['removal_step']),
)


def save_and_close(fig_path: Path, dpi: int = 300):
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def compute_metrics(y_true, probs):
    preds = np.argmax(probs, axis=1)
    acc_value = accuracy_score(y_true, preds)
    f1_value = f1_score(y_true, preds, average='macro')
    mse_value = mean_squared_error(np.eye(probs.shape[1], dtype=float)[y_true], probs)
    return acc_value, f1_value, mse_value


def optimize_model_optuna(model_name, x, y, *, n_classes, n_trials, cv_folds, seed):
    metric_name = OPTUNA_CFG.get('metric', 'f1_macro')
    print(f'\n[Optuna] Optimizing {model_name.upper()} | trials={n_trials} cv={cv_folds} metric={metric_name}')

    param_fn = MODEL_PARAM_FUNCTIONS[model_name]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial):
        params = param_fn(trial)
        scores = []

        for tr_idx, va_idx in cv.split(x, y):
            x_tr, x_va = x[tr_idx], x[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = create_sklearn_model(
                model_name,
                params,
                n_classes=n_classes,
                seed=seed,
            )
            model.fit(x_tr, y_tr)
            preds = model.predict(x_va)
            scores.append(f1_score(y_va, preds, average='macro'))

        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize')
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(
        f'[Optuna] Done {model_name.upper()} | '
        f'best_f1_macro={study.best_value:.4f} | elapsed={time.time() - t0:.1f}s'
    )

    return {
        'best_params': study.best_trial.params,
        'best_value': float(study.best_value),
        'n_trials': n_trials,
        'cv_folds': cv_folds,
    }


def init_safe_store(model_names):
    return {
        name: {
            'rga_curve': [],
            'rgr_curve': [],
            'rge_curve': [],
            'rga_full': [],
            'aurga': [],
            'aurgr': [],
            'aurge': [],
        }
        for name in model_names
    }


def mean_std(vals):
    arr = np.asarray(vals, float)
    return (
        float(arr.mean()) if len(arr) else float('nan'),
        float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    )


def plot_mean(inputs, safe_dict, curve_key, title, x_label, y_label, save_path, model_names):
    plt.figure(figsize=(8, 6))
    for model_name in model_names:
        curves = safe_dict[model_name][curve_key]
        arr = np.stack(curves, axis=0)
        mu = arr.mean(axis=0)
        plt.plot(inputs, mu, label=model_name)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    save_and_close(save_path)


def main():
    print('Dataset exists:', DATA_DIR.exists())

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = CroppedImage(DATA_DIR, transform=transform, apply_crop=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_IMAGES, shuffle=False)

    class_names = dataset.classes
    n_classes = len(class_names)

    fig, axes = plt.subplots(n_classes, 2, figsize=(10, 5 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx, class_name in enumerate(class_names):
        class_images = [sample for sample in dataset.dataset.samples if sample[1] == class_idx]
        img_path, _ = class_images[np.random.randint(len(class_images))]

        img_bgr = cv2.imread(img_path)
        cropped_bgr = crop_img(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

        axes[class_idx, 0].imshow(img_rgb)
        axes[class_idx, 0].set_title(f'{class_name} - Original')
        axes[class_idx, 0].axis('off')

        axes[class_idx, 1].imshow(cropped_rgb)
        axes[class_idx, 1].set_title(f'{class_name} - Cropped')
        axes[class_idx, 1].axis('off')

    plt.tight_layout()
    save_and_close(FIG_DIR / 'cropping_examples.png')

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()

    features, labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            feat = resnet(x_batch)
            features.append(feat.cpu())
            labels.append(y_batch)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    print('Feature shape:', features.shape)

    scaler = StandardScaler()
    x = scaler.fit_transform(features).astype(np.float32)
    y = labels.astype(int)

    feature_names = [f'feat_{j:04d}' for j in range(x.shape[1])]

    df_full = pd.DataFrame(x, columns=feature_names)
    df_full.insert(0, 'label', y.astype(int))

    full_path = CSV_DIR / f'{FOLD_EXPORT_PREFIX}_full_dataset.csv'
    df_full.to_csv(full_path, index=False)
    print(f'Saved full brain dataset to {full_path}')

    k_fold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    best_params = {}

    if optuna_json_path.exists():
        print('[Optuna] Loading existing parameters...')
        with open(optuna_json_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)
    else:
        print('[Optuna] Running hyperparameter tuning...')
        optuna_records = []

        for mn in TUNE_MODELS:
            optuna_result = optimize_model_optuna(
                mn,
                x,
                y,
                n_classes=n_classes,
                n_trials=OPTUNA_TRIALS,
                cv_folds=OPTUNA_CV_FOLDS,
                seed=SEED,
            )

            best_params[mn] = optuna_result['best_params']

            print(f'[Optuna] Best params for {mn}: {optuna_result["best_params"]}')
            print(f'[Optuna] Best CV F1 for {mn}: {optuna_result["best_value"]:.4f}')

            optuna_records.append({
                'model': mn,
                'best_f1_macro_cv': float(optuna_result['best_value']),
                'n_trials': OPTUNA_TRIALS,
                'cv_folds': OPTUNA_CV_FOLDS,
                'seed': SEED,
                **optuna_result['best_params'],
            })

        pd.DataFrame(optuna_records).to_csv(optuna_csv_path, index=False)
        print(f'[Optuna] Saved summary to {optuna_csv_path}')

        with open(optuna_json_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=4)
        print(f'[Optuna] Saved parameters to {optuna_json_path}')

    results = {m: {'acc': [], 'f1': [], 'mse': []} for m in MODEL_NAMES}

    linear_models, rf_models, svm_models = [], [], []
    xgb_models, vem_models, sem_models = [], [], []
    mlp_models = []

    in_dim = x.shape[1]
    class_order = np.unique(labels)

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(x, y), 1):
        print(f'\n====== Fold {fold} ======')

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        df_train = pd.DataFrame(x_train, columns=feature_names)
        df_train.insert(0, 'label', y_train.astype(int))

        df_test = pd.DataFrame(x_val, columns=feature_names)
        df_test.insert(0, 'label', y_val.astype(int))

        df_fold = pd.concat([df_train, df_test], axis=0, ignore_index=True)

        out_path = CSV_DIR / f'{FOLD_EXPORT_PREFIX}_fold{fold}_rulex_ordered.csv'
        df_fold.to_csv(out_path, index=False)

        path_test = CSV_DIR / f'{FOLD_EXPORT_PREFIX}_fold{fold}_test.csv'
        df_test.to_csv(path_test, index=False)

        print(f'Saved fold {fold} ordered CSV to {out_path} | train={len(df_train)} test={len(df_test)}')

        linear = create_sklearn_model(
            'logistic_regression',
            best_params['logistic_regression'],
            n_classes=n_classes,
            seed=SEED,
        )
        linear.fit(x_train, y_train)
        probs_linear = linear.predict_proba(x_val)
        acc, f1m, mse = compute_metrics(y_val, probs_linear)
        results['Linear']['acc'].append(acc)
        results['Linear']['f1'].append(f1m)
        results['Linear']['mse'].append(mse)
        print(f'LIN | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')
        linear_models.append(linear)

        rf = create_sklearn_model(
            'random_forest',
            best_params['random_forest'],
            n_classes=n_classes,
            seed=SEED,
        )
        rf.fit(x_train, y_train)
        probs_rf = rf.predict_proba(x_val)
        acc, f1m, mse = compute_metrics(y_val, probs_rf)
        results['RF']['acc'].append(acc)
        results['RF']['f1'].append(f1m)
        results['RF']['mse'].append(mse)
        print(f'RF  | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')
        rf_models.append(rf)

        svm = create_sklearn_model(
            'svm',
            best_params['svm'],
            n_classes=n_classes,
            seed=SEED,
        )
        svm.fit(x_train, y_train)
        probs_svm = svm.predict_proba(x_val)
        acc, f1m, mse = compute_metrics(y_val, probs_svm)
        results['SVM']['acc'].append(acc)
        results['SVM']['f1'].append(f1m)
        results['SVM']['mse'].append(mse)
        print(f'SVM | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')
        svm_models.append(svm)

        xgb = create_sklearn_model(
            'xgboost',
            best_params['xgboost'],
            n_classes=n_classes,
            seed=SEED,
        )
        xgb.fit(x_train, y_train)
        probs_xgb = xgb.predict_proba(x_val)
        acc, f1m, mse = compute_metrics(y_val, probs_xgb)
        results['XGB']['acc'].append(acc)
        results['XGB']['f1'].append(f1m)
        results['XGB']['mse'].append(mse)
        print(f'XGB | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')
        xgb_models.append(xgb)

        voting_estimators = []
        for est_name in ENSEMBLES_CFG['voting']['estimators']:
            short_name = 'rf' if est_name == 'random_forest' else 'xgb'
            voting_estimators.append(
                (
                    short_name,
                    create_sklearn_model(
                        est_name,
                        best_params[est_name],
                        n_classes=n_classes,
                        seed=SEED,
                    ),
                )
            )

        vem = VotingClassifier(
            estimators=voting_estimators,
            voting=ENSEMBLES_CFG['voting'].get('voting', 'soft'),
            n_jobs=-1,
        )
        vem.fit(x_train, y_train)
        probs_vem = vem.predict_proba(x_val)
        acc, f1m, mse = compute_metrics(y_val, probs_vem)
        results['VEM']['acc'].append(acc)
        results['VEM']['f1'].append(f1m)
        results['VEM']['mse'].append(mse)
        print(f'VEM | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')
        vem_models.append(vem)

        stacking_estimators = []
        for est_name in ENSEMBLES_CFG['stacking']['estimators']:
            base_model = create_sklearn_model(
                est_name,
                best_params[est_name],
                n_classes=n_classes,
                seed=SEED,
            )
            base_model.fit(x_train, y_train)
            short_name = 'rf' if est_name == 'random_forest' else 'xgb'
            stacking_estimators.append((short_name, base_model))

        final_est_cfg = ENSEMBLES_CFG['stacking']['final_estimator']
        final_estimator = LogisticRegression(
            C=final_est_cfg.get('C', 1.0),
            max_iter=final_est_cfg.get('max_iter', 2000),
            solver=final_est_cfg.get('solver', 'lbfgs'),
            random_state=SEED,
        )

        sem = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=final_estimator,
            cv='prefit',
            stack_method='auto',
            n_jobs=-1,
        )
        sem.fit(x_train, y_train)
        probs_sem = sem.predict_proba(x_val)
        acc, f1m, mse = compute_metrics(y_val, probs_sem)
        results['SEM']['acc'].append(acc)
        results['SEM']['f1'].append(f1m)
        results['SEM']['mse'].append(mse)
        print(f'SEM | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')
        sem_models.append(sem)

        mlp = build_mlp(in_dim, n_classes, device)
        mlp = train_torch_model(
            mlp,
            x_train,
            y_train,
            x_val,
            y_val,
            batch_size=BATCH_SIZE_TRAIN,
            learning_rate=LR_TORCH,
            epochs=EPOCHS,
            device=device,
        )

        mlp.eval()
        with torch.no_grad():
            probs_mlp = torch.softmax(
                mlp(torch.tensor(x_val, dtype=torch.float32).to(device)),
                dim=1,
            ).cpu().numpy()

        acc, f1m, mse = compute_metrics(y_val, probs_mlp)
        results['MLP']['acc'].append(acc)
        results['MLP']['f1'].append(f1m)
        results['MLP']['mse'].append(mse)
        print(f'MLP | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')
        mlp_models.append(copy.deepcopy(mlp))

    print(f'\n================ OVERALL {N_SPLITS}-FOLD ================')
    for model_name, metric_dict in results.items():
        print(
            f'{model_name:>3} | '
            f'ACC={np.mean(metric_dict["acc"]):.4f} '
            f'F1={np.mean(metric_dict["f1"]):.4f} '
            f'MSE={np.mean(metric_dict["mse"]):.6f}'
        )

    safe_store = init_safe_store(MODEL_NAMES)

    x_t = torch.tensor(x, dtype=torch.float32)
    y_labels = labels

    x_images = torch.stack([img for img, _ in dataset])
    x_images_dataset = TensorDataset(x_images)

    cam_model = train_cam_model(
        feature_extractor=resnet,
        images=x_images,
        labels=labels,
        scaler=scaler,
        n_classes=n_classes,
        device=device,
        epochs=CAM_EPOCHS,
        lr=CAM_LR,
        batch_size=CAM_BATCH_SIZE,
        verbose=SAFE_VERBOSE,
    )

    importance_maps = compute_gradcam_maps(
        images=x_images,
        cam_model=cam_model,
        device=device,
        batch_pred=CAM_BATCH_SIZE,
        verbose=SAFE_VERBOSE,
    )

    patch_rankings, patch_meta = precompute_patch_rankings(
        importance_maps=importance_maps,
        patch_size=PATCH_SIZE,
    )

    print('Total patches:', patch_meta['total_patches'])

    idx = 0
    show_heatmap_per_class(
        x_images,
        importance_maps,
        labels,
        class_names,
        n_classes,
        save_path=FIG_DIR / 'gradcam_heatmap_per_class.png',
    )
    show_occlusions_same_idx(
        x_images,
        patch_rankings,
        patch_meta,
        idx=idx,
        save_path=FIG_DIR / 'gradcam_occlusions.png',
    )

    def preprocess(images):
        return extract_features_from_images(
            images,
            feature_extractor=resnet,
            pca=None,
            scaler=scaler,
            device=device,
            batch_size=BATCH_SIZE_IMAGES,
        )

    safe_fig_dir = FIG_DIR / 'safe_folds'
    safe_fig_dir.mkdir(parents=True, exist_ok=True)

    for fold, (linear, rf, svm, xgb, vem, sem, mlp) in enumerate(
        zip(
            linear_models,
            rf_models,
            svm_models,
            xgb_models,
            vem_models,
            sem_models,
            mlp_models,
        ),
        1,
    ):
        fold_dir = safe_fig_dir / f'fold_{fold:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        rga_path = fold_dir / 'rga.png'
        rgr_path = fold_dir / 'rgr.png'
        rge_path = fold_dir / 'rge.png'

        print(f'\n====== SAFE AI METRICS Fold {fold} ======')

        mlp.eval()

        with torch.no_grad():
            prob_base = torch.softmax(mlp(x_t.to(device)), dim=1).cpu().numpy()

        prob_lin = linear.predict_proba(x)
        prob_rf = rf.predict_proba(x)
        prob_svm = svm.predict_proba(x)
        prob_xgb = xgb.predict_proba(x)
        prob_vem = vem.predict_proba(x)
        prob_sem = sem.predict_proba(x)

        prob_base = align_proba_to_class_order(prob_base, class_order, class_order)
        prob_lin = align_proba_to_class_order(prob_lin, linear.classes_, class_order)
        prob_rf = align_proba_to_class_order(prob_rf, rf.classes_, class_order)
        prob_svm = align_proba_to_class_order(prob_svm, svm.classes_, class_order)
        prob_xgb = align_proba_to_class_order(prob_xgb, xgb.classes_, class_order)
        prob_vem = align_proba_to_class_order(prob_vem, vem.classes_, class_order)
        prob_sem = align_proba_to_class_order(prob_sem, sem.classes_, class_order)

        models_rga = {
            'Linear': (prob_lin, class_order),
            'RF': (prob_rf, class_order),
            'SVM': (prob_svm, class_order),
            'XGB': (prob_xgb, class_order),
            'VEM': (prob_vem, class_order),
            'SEM': (prob_sem, class_order),
            'MLP': (prob_base, class_order),
        }

        results_rga = compare_models_rga(
            models_rga,
            y_labels=y_labels,
            n_segments=N_SEGMENTS,
            fig_size=FIG_SIZE,
            verbose=SAFE_VERBOSE,
            save_path=rga_path,
        )

        rga_dict = {m: float(results_rga[m]['rga_full']) for m in MODEL_NAMES}

        for model_name in MODEL_NAMES:
            safe_store[model_name]['rga_curve'].append(np.asarray(results_rga[model_name]['curve_model'], float))
            safe_store[model_name]['rga_full'].append(float(results_rga[model_name]['rga_full']))
            safe_store[model_name]['aurga'].append(float(results_rga[model_name]['aurga_normalized_to_perfect']))

        models_rgr = {
            'Linear': (linear, x, prob_lin, class_order, 'sklearn', None),
            'RF': (rf, x, prob_rf, class_order, 'sklearn', None),
            'SVM': (svm, x, prob_svm, class_order, 'sklearn', None),
            'XGB': (xgb, x, prob_xgb, class_order, 'sklearn', None),
            'VEM': (vem, x, prob_vem, class_order, 'sklearn', None),
            'SEM': (sem, x, prob_sem, class_order, 'sklearn', None),
            'MLP': (mlp, x_t, prob_base, class_order, 'pytorch', device),
        }

        results_rgr = compare_models_rgr(
            models_dict=models_rgr,
            noise_levels=noise_levels,
            class_order=class_order,
            rga_dict=rga_dict,
            fig_size=FIG_SIZE,
            verbose=SAFE_VERBOSE,
            random_seed=SEED,
            save_path=rgr_path,
        )

        for model_name in MODEL_NAMES:
            safe_store[model_name]['rgr_curve'].append(np.asarray(results_rgr[model_name]['rgr_rescaled'], float))
            safe_store[model_name]['aurgr'].append(float(results_rgr[model_name]['aurgr']))

        models_rge = {
            'Linear': (linear, preprocess, class_order, 'sklearn'),
            'RF': (rf, preprocess, class_order, 'sklearn'),
            'SVM': (svm, preprocess, class_order, 'sklearn'),
            'XGB': (xgb, preprocess, class_order, 'sklearn'),
            'VEM': (vem, preprocess, class_order, 'sklearn'),
            'SEM': (sem, preprocess, class_order, 'sklearn'),
            'MLP': (mlp, preprocess, class_order, 'pytorch'),
        }

        results_rge = compare_models_rge(
            models_dict=models_rge,
            images_dataset=x_images_dataset,
            removal_fractions=removal_fractions,
            class_order=class_order,
            occlusion_method=OCCLUSION_METHOD,
            patch_size=PATCH_SIZE,
            batch_size=BATCH_SIZE_SAFE,
            class_weights=None,
            rga_dict=rga_dict,
            device=device,
            verbose=SAFE_VERBOSE,
            random_seed=SEED,
            patch_rankings=patch_rankings,
            patch_meta=patch_meta,
            save_path=rge_path,
        )

        for model_name in MODEL_NAMES:
            safe_store[model_name]['rge_curve'].append(np.asarray(results_rge[model_name]['rge_rescaled'], float))
            safe_store[model_name]['aurge'].append(float(results_rge[model_name]['aurge']))

        fold_rows = []
        for model_name in MODEL_NAMES:
            fold_rows.append({
                'fold': fold,
                'model': model_name,
                'RGA': float(results_rga[model_name]['rga_full']),
                'AURGA': float(results_rga[model_name]['aurga']),
                'AURGR': float(results_rgr[model_name]['aurgr']),
                'AURGE': float(results_rge[model_name]['aurge']),
            })

        pd.DataFrame(fold_rows).to_csv(fold_dir / 'safe_metrics_fold.csv', index=False)

    print_standard_summary(results, MODEL_NAMES, N_SPLITS)

    first_model = MODEL_NAMES[0]
    l_rga = len(safe_store[first_model]['rga_curve'][0])
    x_rga = np.linspace(0, 1, l_rga)

    plot_mean_curve(
        x=x_rga,
        safe_store=safe_store,
        curve_key='rga_curve',
        title=f'RGA (mean across {N_SPLITS} folds)',
        x_label='Fraction of Data Removed',
        y_label='RGA Score',
        save_path=FIG_DIR / 'rga_mean.png',
        model_names=MODEL_NAMES,
    )

    plot_mean_curve(
        x=noise_levels * 100,
        safe_store=safe_store,
        curve_key='rgr_curve',
        title=f'RGR (mean across {N_SPLITS} folds)',
        x_label='Noise Standard Deviation (%)',
        y_label='RGR Score',
        save_path=FIG_DIR / 'rgr_mean.png',
        model_names=MODEL_NAMES,
    )

    plot_mean_curve(
        x=removal_fractions * 100,
        safe_store=safe_store,
        curve_key='rge_curve',
        title=f'RGE (mean across {N_SPLITS} folds)',
        x_label='Occluded Image Area (%)',
        y_label='RGE Score',
        save_path=FIG_DIR / 'rge_mean.png',
        model_names=MODEL_NAMES,
    )

    save_safe_summary(
        results=results,
        safe_store=safe_store,
        model_names=MODEL_NAMES,
        csv_path=CSV_DIR / 'safe_summary_metrics_image.csv',
    )


if __name__ == '__main__':
    main()