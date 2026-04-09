import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from helpers import print_standard_summary, plot_mean_curve, save_safe_summary
from models import (
    MODEL_PARAM_FUNCTIONS,
    build_mlp,
    create_sklearn_model,
    train_torch_model,
)

from safe.rga import compare_models_rga
from safe.rge import compare_models_rge_tabular
from safe.rgr import compare_models_rgr
from safe.utils import align_proba_to_class_order


PROJECT_ROOT = Path(__file__).resolve().parent.parent
config_path = PROJECT_ROOT / 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

GLOBAL_CFG = config.get('global', {})
PATHS_CFG = config['paths']
HMDA_CFG = config['hmda']

FIG_DIR = PROJECT_ROOT / PATHS_CFG['fig_hmda_dir']
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_DIR = PROJECT_ROOT / PATHS_CFG['csv_dir']
CSV_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(HMDA_CFG.get('seed', GLOBAL_CFG.get('seed', 42)))
N_SPLITS = int(HMDA_CFG.get('n_splits', GLOBAL_CFG.get('n_splits', 5)))

DATASET_CFG = HMDA_CFG['dataset']
OPTUNA_CFG = HMDA_CFG['optuna']
TRAIN_CFG = HMDA_CFG['training']
SAFE_CFG = HMDA_CFG['safe']
MODELS_CFG = HMDA_CFG['models']
ENSEMBLES_CFG = HMDA_CFG['ensembles']

RAW_CSV = Path(DATASET_CFG['raw_csv'])
TARGET_COL = DATASET_CFG['target']
CREATE_SAMPLE = bool(DATASET_CFG.get('create_sample', False))
SAMPLE_FRACTION = float(DATASET_CFG.get('sample_fraction', 0.1))
SAMPLE_SIZE = DATASET_CFG.get('sample_size', None)
FOLD_EXPORT_PREFIX = DATASET_CFG.get('fold_export_prefix', 'hmda')

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

RGE_CFG = SAFE_CFG['rge']
RGE_N_STEPS = RGE_CFG.get('n_steps', None)
RGE_MASKING_METHOD = RGE_CFG.get('masking_method', 'greedy')
RGE_BASELINE = RGE_CFG.get('baseline', 'mean')

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


VALUES_TO_REMOVE = {
    'action_taken': [2, 4, 5, 6, 7, 8],
    'loan_type': [3, 4],
    'applicant_race_1': [1, 2, 4, 6, 7],
    'lien_status': [3, 4],
    'applicant_sex': [3, 4],
    'co_applicant_sex': [3, 4],
    'co_applicant_race_1': [1, 2, 4, 6, 7],
    'applicant_ethnicity': [3, 4],
    'co_applicant_ethnicity': [3, 4],
}

COLUMNS_TO_REMOVE = [
    'as_of_year', 'agency_name', 'agency_abbr', 'agency_code', 'property_type_name',
    'property_type', 'owner_occupancy_name', 'owner_occupancy', 'preapproval_name', 'preapproval',
    'state_name', 'state_abbr', 'state_code', 'applicant_race_name_2', 'applicant_race_2',
    'applicant_race_name_3', 'applicant_race_3', 'applicant_race_name_4', 'applicant_race_4',
    'applicant_race_name_5', 'applicant_race_5', 'co_applicant_race_name_2', 'co_applicant_race_2',
    'co_applicant_race_name_3', 'co_applicant_race_3', 'co_applicant_race_name_4', 'co_applicant_race_4',
    'co_applicant_race_name_5', 'co_applicant_race_5', 'purchaser_type_name', 'purchaser_type',
    'denial_reason_name_1', 'denial_reason_1', 'denial_reason_name_2', 'denial_reason_2',
    'denial_reason_name_3', 'denial_reason_3', 'rate_spread', 'hoepa_status_name', 'hoepa_status',
    'edit_status_name', 'edit_status', 'sequence_number', 'application_date_indicator',
    'respondent_id', 'loan_type_name', 'loan_purpose_name', 'action_taken_name', 'msamd_name',
    'county_name', 'applicant_ethnicity_name', 'co_applicant_ethnicity_name',
    'applicant_race_name_1', 'co_applicant_race_name_1', 'applicant_sex_name',
    'co_applicant_sex_name', 'lien_status_name',
]


def load_raw_hmda(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def clean_hmda_binary(df: pd.DataFrame, target_col: str = 'action_taken') -> pd.DataFrame:
    df = df.copy()

    for col, bad_values in VALUES_TO_REMOVE.items():
        if col in df.columns:
            df = df[~df[col].isin(bad_values)]

    cols_to_drop = [col for col in COLUMNS_TO_REMOVE if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = df.dropna(axis='index')

    if target_col not in df.columns:
        raise ValueError(f'Target column \'{target_col}\' not found.')

    df.loc[:, target_col] = df[target_col].map({1: 0, 3: 1})
    df = df.dropna(axis='index')
    df.loc[:, target_col] = df[target_col].astype(int)

    return df


def subsample_stratified(
    df,
    target_col,
    *,
    create_sample,
    sample_fraction=0.1,
    sample_size=None,
    random_state=42,
):
    if not create_sample:
        return df

    if target_col not in df.columns:
        raise ValueError(f'Target column \'{target_col}\' not found for subsample.')

    if sample_size is None:
        sample_size = int(len(df) * float(sample_fraction))

    _, sample_df = train_test_split(
        df,
        test_size=sample_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return sample_df


def compute_metrics(y_true: np.ndarray, probs: np.ndarray):
    preds = np.argmax(probs, axis=1)
    acc_value = accuracy_score(y_true, preds)
    f1_value = f1_score(y_true, preds, average='macro')
    onehot = np.eye(probs.shape[1], dtype=float)[y_true]
    mse_value = mean_squared_error(onehot, probs)
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

            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_tr)
            x_va = scaler.transform(x_va)

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
    print(f'[Optuna] Done {model_name.upper()} | best_f1_macro={study.best_value:.4f} | elapsed={time.time() - t0:.1f}s')

    return {
        'best_params': study.best_trial.params,
        'best_value': float(study.best_value),
        'n_trials': n_trials,
        'cv_folds': cv_folds,
    }


def main():
    raw_csv = RAW_CSV
    if not raw_csv.is_absolute():
        raw_csv = PROJECT_ROOT / raw_csv

    sample_size = SAMPLE_SIZE
    if sample_size is not None:
        sample_size = int(sample_size)

    print('Loading HMDA:', raw_csv)
    df_raw = load_raw_hmda(raw_csv)
    df = clean_hmda_binary(df_raw, target_col=TARGET_COL)
    df = subsample_stratified(
        df,
        TARGET_COL,
        create_sample=CREATE_SAMPLE,
        sample_fraction=SAMPLE_FRACTION,
        sample_size=sample_size,
        random_state=SEED,
    )

    y = df[TARGET_COL].astype(int).to_numpy()
    x_df = df.drop(columns=[TARGET_COL])
    feature_names = x_df.columns.to_list()
    x = x_df.to_numpy(dtype=np.float32)

    n_classes = 2
    class_order = np.array([0, 1], dtype=int)

    print('HMDA cleaned shape:', df.shape)
    print('X shape:', x.shape, '| y shape:', y.shape, '| classes:', sorted(set(y.tolist())))

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

            optuna_records.append({
                'model': mn,
                'best_f1_macro_cv': float(optuna_result['best_value']),
                'n_trials': OPTUNA_TRIALS,
                'cv_folds': OPTUNA_CV_FOLDS,
                'seed': SEED,
                **optuna_result['best_params'],
            })

        pd.DataFrame(optuna_records).to_csv(optuna_csv_path, index=False)
        with open(optuna_json_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=4)

        print(f'[Optuna] Saved summary to {optuna_csv_path}')
        print(f'[Optuna] Saved parameters to {optuna_json_path}')

    k_fold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = {m: {'acc': [], 'f1': [], 'mse': []} for m in MODEL_NAMES}
    safe_store = {
        m: {
            'rga_curve': [],
            'rga_full': [],
            'aurga': [],
            'rgr_curve': [],
            'aurgr': [],
            'rge_curve': [],
            'aurge': [],
        }
        for m in MODEL_NAMES
    }

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(x, y), 1):
        print(f'\n====== Fold {fold} ======')

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        x_train_raw, x_val_raw = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
        x_val = scaler.transform(x_val_raw).astype(np.float32)

        df_train = pd.DataFrame(x_train, columns=feature_names)
        df_train.insert(0, 'label', y_train)

        df_test = pd.DataFrame(x_val, columns=feature_names)
        df_test.insert(0, 'label', y_val)

        df_fold = pd.concat([df_train, df_test], axis=0, ignore_index=True)
        path_ordered = CSV_DIR / f'{FOLD_EXPORT_PREFIX}_fold{fold}_rulex_ordered.csv'
        df_fold.to_csv(path_ordered, index=False)

        path_test = CSV_DIR / f'{FOLD_EXPORT_PREFIX}_fold{fold}_test.csv'
        df_test.to_csv(path_test, index=False)

        print(
            f'Saved fold {fold}:\n'
            f'full: {path_ordered} (train={len(df_train)} test={len(df_test)})\n'
            f'test_only: {path_test}'
        )

        linear = create_sklearn_model(
            'logistic_regression',
            best_params['logistic_regression'],
            n_classes=n_classes,
            seed=SEED,
        )
        linear.fit(x_train, y_train)
        prob_lin = align_proba_to_class_order(linear.predict_proba(x_val), linear.classes_, class_order)

        rf = create_sklearn_model(
            'random_forest',
            best_params['random_forest'],
            n_classes=n_classes,
            seed=SEED,
        )
        rf.fit(x_train, y_train)
        prob_rf = align_proba_to_class_order(rf.predict_proba(x_val), rf.classes_, class_order)

        svm = create_sklearn_model(
            'svm',
            best_params['svm'],
            n_classes=n_classes,
            seed=SEED,
        )
        svm.fit(x_train, y_train)
        prob_svm = align_proba_to_class_order(svm.predict_proba(x_val), svm.classes_, class_order)

        xgb = create_sklearn_model(
            'xgboost',
            best_params['xgboost'],
            n_classes=n_classes,
            seed=SEED,
        )
        xgb.fit(x_train, y_train)
        prob_xgb = align_proba_to_class_order(xgb.predict_proba(x_val), xgb.classes_, class_order)

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
        prob_vem = align_proba_to_class_order(vem.predict_proba(x_val), vem.classes_, class_order)

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
        prob_sem = align_proba_to_class_order(sem.predict_proba(x_val), sem.classes_, class_order)

        mlp = build_mlp(x_train.shape[1], n_classes, device)
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
            prob_mlp = torch.softmax(
                mlp(torch.tensor(x_val, dtype=torch.float32).to(device)),
                dim=1,
            ).cpu().numpy()
        prob_mlp = align_proba_to_class_order(prob_mlp, np.array([0, 1]), class_order)

        for name, probs in [
            ('Linear', prob_lin),
            ('RF', prob_rf),
            ('SVM', prob_svm),
            ('XGB', prob_xgb),
            ('VEM', prob_vem),
            ('SEM', prob_sem),
            ('MLP', prob_mlp),
        ]:
            acc, f1m, mse = compute_metrics(y_val, probs)
            results[name]['acc'].append(acc)
            results[name]['f1'].append(f1m)
            results[name]['mse'].append(mse)
            print(f'{name:>6} | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')

        models_rga = {
            'Linear': (prob_lin, class_order),
            'RF': (prob_rf, class_order),
            'SVM': (prob_svm, class_order),
            'XGB': (prob_xgb, class_order),
            'VEM': (prob_vem, class_order),
            'SEM': (prob_sem, class_order),
            'MLP': (prob_mlp, class_order),
        }
        results_rga = compare_models_rga(
            models_rga,
            y_labels=y_val,
            n_segments=N_SEGMENTS,
            fig_size=FIG_SIZE,
            verbose=SAFE_VERBOSE,
            save_path=FIG_DIR / f'rga_fold{fold}.png',
        )
        rga_dict = {m: float(results_rga[m]['rga_full']) for m in MODEL_NAMES}

        for m in MODEL_NAMES:
            safe_store[m]['rga_curve'].append(np.asarray(results_rga[m]['curve_model'], float))
            safe_store[m]['rga_full'].append(float(results_rga[m]['rga_full']))
            safe_store[m]['aurga'].append(float(results_rga[m]['aurga_normalized_to_perfect']))

        models_rgr = {
            'Linear': (linear, x_val, prob_lin, class_order, 'sklearn', None),
            'RF': (rf, x_val, prob_rf, class_order, 'sklearn', None),
            'SVM': (svm, x_val, prob_svm, class_order, 'sklearn', None),
            'XGB': (xgb, x_val, prob_xgb, class_order, 'sklearn', None),
            'VEM': (vem, x_val, prob_vem, class_order, 'sklearn', None),
            'SEM': (sem, x_val, prob_sem, class_order, 'sklearn', None),
            'MLP': (mlp, x_val, prob_mlp, class_order, 'pytorch', device),
        }
        results_rgr = compare_models_rgr(
            models_dict=models_rgr,
            noise_levels=noise_levels,
            class_order=class_order,
            rga_dict=rga_dict,
            fig_size=FIG_SIZE,
            verbose=SAFE_VERBOSE,
            random_seed=SEED,
            save_path=FIG_DIR / f'rgr_fold{fold}.png',
        )

        for m in MODEL_NAMES:
            safe_store[m]['rgr_curve'].append(np.asarray(results_rgr[m]['rgr_rescaled'], float))
            safe_store[m]['aurgr'].append(float(results_rgr[m]['aurgr']))

        models_rge = {
            'Linear': (linear, x_val, feature_names, prob_lin, linear.classes_, 'sklearn', None),
            'RF': (rf, x_val, feature_names, prob_rf, rf.classes_, 'sklearn', None),
            'SVM': (svm, x_val, feature_names, prob_svm, svm.classes_, 'sklearn', None),
            'XGB': (xgb, x_val, feature_names, prob_xgb, xgb.classes_, 'sklearn', None),
            'VEM': (vem, x_val, feature_names, prob_vem, vem.classes_, 'sklearn', None),
            'SEM': (sem, x_val, feature_names, prob_sem, sem.classes_, 'sklearn', None),
            'MLP': (mlp, x_val, feature_names, prob_mlp, np.array([0, 1]), 'pytorch', device),
        }

        results_rge = compare_models_rge_tabular(
            models_dict=models_rge,
            class_order=class_order,
            rga_dict=rga_dict,
            verbose=SAFE_VERBOSE,
            random_seed=SEED,
            fig_size=FIG_SIZE,
            save_path=FIG_DIR / f'rge_fold{fold}.png',
            masking_method=RGE_MASKING_METHOD,
            baseline=RGE_BASELINE,
            n_steps=RGE_N_STEPS,
        )

        for m in MODEL_NAMES:
            safe_store[m]['rge_curve'].append(np.asarray(results_rge[m]['rge_rescaled'], float))
            safe_store[m]['aurge'].append(float(results_rge[m]['aurge']))

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

    l_rge = len(safe_store[first_model]['rge_curve'][0])
    x_rge = np.linspace(0, 100, l_rge)

    plot_mean_curve(
        x=x_rge,
        safe_store=safe_store,
        curve_key='rge_curve',
        title=f'RGE (mean across {N_SPLITS} folds)',
        x_label='Removed Features (%)',
        y_label='RGE Score',
        save_path=FIG_DIR / 'rge_mean.png',
        model_names=MODEL_NAMES,
    )

    save_safe_summary(
        results=results,
        safe_store=safe_store,
        model_names=MODEL_NAMES,
        csv_path=CSV_DIR / 'safe_summary_metrics_hmda.csv',
    )


if __name__ == '__main__':
    main()