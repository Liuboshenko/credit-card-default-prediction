"""
Train two classification pipelines:
  v1 — LogisticRegression (baseline, no class_weight bias)
  v2 — GradientBoostingClassifier (improved, tuned)

Run:
    python models/train_model.py [--data PATH]
"""
import os, sys, argparse, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_NAMES, TARGET_COLUMN

MODELS_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(os.path.dirname(MODELS_DIR), '..', 'data/UCI_Credit_Card.csv')

TEST_CLIENT = {
    'LIMIT_BAL': 1000, 'SEX': 1, 'EDUCATION': 1, 'MARRIAGE': 1, 'AGE': 34,
    'PAY_0': 2,  'PAY_2': 2,  'PAY_3': -1, 'PAY_4': -1, 'PAY_5': 0, 'PAY_6': 0,
    'BILL_AMT1': 1,  'BILL_AMT2': 1, 'BILL_AMT3': 1,
    'BILL_AMT4': 0,  'BILL_AMT5': 1, 'BILL_AMT6': 0,
    'PAY_AMT1': 0,   'PAY_AMT2': 1,  'PAY_AMT3': 0,
    'PAY_AMT4': 0,   'PAY_AMT5': 0,  'PAY_AMT6': 0,
}


def load_data(path: str):
    df = pd.read_csv(path)
    # убираем выбросы EDUCATION и MARRIAGE (значения вне словаря)
    df = df[df['EDUCATION'].isin([1, 2, 3, 4])]
    df = df[df['MARRIAGE'].isin([1, 2, 3])]
    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]
    return X, y


def report(name, pipeline, X_test, y_test):
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"\n{'='*55}")
    print(f"  {name}")
    print('='*55)
    print(classification_report(y_test, y_pred,
                                target_names=['No Default', 'Default']))
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  F1       : {f1_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
    return {
        'roc_auc':   roc_auc_score(y_test, y_proba),
        'f1':        f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
    }


def check_test_client(v1, v2):
    X = pd.DataFrame([TEST_CLIENT], columns=FEATURE_NAMES)
    p1 = v1.predict_proba(X)[0][1]
    p2 = v2.predict_proba(X)[0][1]
    print(f"\n{'='*55}")
    print("  Тестовый клиент (PAY_0=2, PAY_2=2, мин. счета)")
    print(f"  Реальная default rate в датасете: ~71.8 %")
    print(f"  v1 прогноз : {p1:.1%}  (pred={int(v1.predict(X)[0])})")
    print(f"  v2 прогноз : {p2:.1%}  (pred={int(v2.predict(X)[0])})")
    print('='*55)


def train_and_save(data_path: str):
    print(f"Загрузка данных: {data_path}")
    X, y = load_data(data_path)
    print(f"Размер: {X.shape}, Default rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #  Model v1: LogisticRegression 
    # class_weight=None — без искусственного перекоса,
    # C=0.5 — небольшая регуляризация чтобы снизить Precision/Recall дисбаланс
    pipe_v1 = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ])
    pipe_v1.fit(X_train, y_train)
    m1 = report('Model v1 — LogisticRegression (C=0.5)', pipe_v1, X_test, y_test)

    path_v1 = os.path.join(MODELS_DIR, 'model_v1.pkl')
    joblib.dump(pipe_v1, path_v1)
    print(f"Сохранено: {path_v1}")

    # Model v2: GradientBoostingClassifier
    # n_estimators=500, max_depth=3, learning_rate=0.03 — точнее и не переобучается
    # min_samples_leaf=20 — стабильность на малых выборках
    pipe_v2 = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.7,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])
    pipe_v2.fit(X_train, y_train)
    m2 = report('Model v2 — GradientBoosting (tuned)', pipe_v2, X_test, y_test)

    path_v2 = os.path.join(MODELS_DIR, 'model_v2.pkl')
    joblib.dump(pipe_v2, path_v2)
    print(f"Сохранено: {path_v2}")

    # Сравнение и тестовый клиент
    print(f"\n{'─'*55}")
    print(f"{'Модель':<35} {'ROC-AUC':>8} {'F1':>7} {'Prec':>7} {'Rec':>7}")
    print(f"{'─'*55}")
    print(f"{'v1  LogisticRegression':<35} {m1['roc_auc']:>8.4f} {m1['f1']:>7.4f} {m1['precision']:>7.4f} {m1['recall']:>7.4f}")
    print(f"{'v2  GradientBoosting':<35} {m2['roc_auc']:>8.4f} {m2['f1']:>7.4f} {m2['precision']:>7.4f} {m2['recall']:>7.4f}")

    check_test_client(pipe_v1, pipe_v2)
    return m1, m2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DEFAULT_DATA)
    args = parser.parse_args()
    train_and_save(os.path.abspath(args.data))
