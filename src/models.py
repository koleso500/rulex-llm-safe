import copy
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


def logistic_regression_params(trial):
    return {
        'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'max_iter': trial.suggest_int('max_iter', 500, 4000),
        'solver': 'lbfgs'
    }


def random_forest_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.2, 0.4, 0.6, 0.8, 1.0]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])
    }


def svm_params(trial):
    return {
        'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'degree': trial.suggest_int('degree', 2, 5)
    }


def xgboost_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 3e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }


MODEL_PARAM_FUNCTIONS = {
    'logistic_regression': logistic_regression_params,
    'random_forest': random_forest_params,
    'svm': svm_params,
    'xgboost': xgboost_params
}


def create_sklearn_model(model_name: str, params: dict, *, n_classes: int, seed: int):
    p = dict(params)

    if model_name == 'logistic_regression':
        return LogisticRegression(
            C=p.get('C', 1.0),
            class_weight=p.get('class_weight'),
            max_iter=p.get('max_iter', 2000),
            solver=p.get('solver', 'lbfgs'),
            random_state=seed
        )

    if model_name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            min_samples_split=p['min_samples_split'],
            min_samples_leaf=p['min_samples_leaf'],
            max_features=p['max_features'],
            bootstrap=p['bootstrap'],
            class_weight=p['class_weight'],
            n_jobs=-1,
            random_state=seed
        )

    if model_name == 'svm':
        return SVC(
            C=p.get('C', 1.0),
            kernel=p.get('kernel', 'rbf'),
            gamma=p.get('gamma', 'scale'),
            degree=p.get('degree', 3),
            probability=True,
            random_state=seed
        )

    if model_name == 'xgboost':
        if n_classes > 2:
            p.setdefault('objective', 'multi:softprob')
            p.setdefault('num_class', int(n_classes))
            p.setdefault('eval_metric', 'mlogloss')
        else:
            p.setdefault('objective', 'binary:logistic')
            p.setdefault('eval_metric', 'logloss')

        return XGBClassifier(**p, n_jobs=-1, random_state=seed)

    raise ValueError(f'Unknown model_name: {model_name}')


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.functional.F.gelu(self.fc1(x))
        return self.out(x)


def build_mlp(input_dim, num_classes, device):
    return MLPBaseline(input_dim, num_classes).to(device)


def train_torch_model(mod, x_train, y_train, x_val, y_val, *, batch_size, learning_rate, epochs, device):
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True
    )

    opt = torch.optim.Adam(mod.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        mod.train()
        running = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(mod(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)

        mod.eval()
        with torch.no_grad():
            xv = torch.tensor(x_val, dtype=torch.float32).to(device)
            yv = torch.tensor(y_val, dtype=torch.long).to(device)
            val_loss = loss_fn(mod(xv), yv).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(mod.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    if best_state is not None:
        mod.load_state_dict(best_state)

    return mod