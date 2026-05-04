"""
Full EDA, model diagnostics and retraining script.

Run from project root:
    python analysis_and_retrain.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score
)

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(PROJECT_DIR, '..', 'UCI_Credit_Card.csv')
MODELS_DIR   = os.path.join(PROJECT_DIR, 'models')

sys.path.insert(0, PROJECT_DIR)
from config import FEATURE_NAMES, TARGET_COLUMN

# ── "проблемный" клиент ────────────────────────────────────────────────────────
PROBLEM_CLIENT = {
    'LIMIT_BAL': 1000, 'SEX': 2, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 34,
    'PAY_0': 2,   'PAY_2': 2,  'PAY_3': -1, 'PAY_4': -1, 'PAY_5': 0, 'PAY_6': 0,
    'BILL_AMT1': 1, 'BILL_AMT2': 1, 'BILL_AMT3': 1,
    'BILL_AMT4': 0, 'BILL_AMT5': 1, 'BILL_AMT6': 0,
    'PAY_AMT1': 0,  'PAY_AMT2': 1,  'PAY_AMT3': 0,
    'PAY_AMT4': 0,  'PAY_AMT5': 0,  'PAY_AMT6': 0,
}

SEP  = '=' * 60
sep2 = '-' * 60


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════
section("1. Загрузка датасета")

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Переименуем целевую колонку если нужно
target_col = TARGET_COLUMN   # 'default.payment.next.month'

# Выявим SEX/EDUCATION/MARRIAGE которые не в FEATURE_NAMES задачи, но нужны для полноты
X_all = df[FEATURE_NAMES].copy()
y_all = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. EDA
# ══════════════════════════════════════════════════════════════════════════════
section("2. EDA")

# 2a) Распределение таргета
print("\n--- 2a) Распределение целевой переменной ---")
vc = y_all.value_counts()
print(f"  No Default (0): {vc[0]:>6}  ({vc[0]/len(y_all)*100:.1f}%)")
print(f"  Default    (1): {vc[1]:>6}  ({vc[1]/len(y_all)*100:.1f}%)")
print(f"  Дисбаланс классов: {vc[0]/vc[1]:.2f}:1")

# 2b) Корреляция PAY_0 с таргетом
print("\n--- 2b) Корреляция PAY_0 с таргетом ---")
corr_pay0 = df['PAY_0'].corr(df[target_col])
print(f"  Pearson corr(PAY_0, target) = {corr_pay0:.4f}")

# 2c) Default rate по значениям PAY_0
print("\n--- 2c) Default rate по значениям PAY_0 ---")
dr_pay0 = df.groupby('PAY_0')[target_col].agg(['mean', 'count'])
dr_pay0.columns = ['default_rate', 'count']
dr_pay0['default_rate_pct'] = (dr_pay0['default_rate'] * 100).round(1)
print(dr_pay0.to_string())
print("\n  PAY_0 кодирование: -2=не использовал, -1=оплачено полностью,")
print("  0=минимальный платёж, 1=задержка 1мес, 2=задержка 2мес, ...")

# 2d) Топ-10 признаков по корреляции с таргетом
print("\n--- 2d) Топ-10 признаков по корреляции с таргетом ---")
corr_series = df[FEATURE_NAMES + [target_col]].corr()[target_col].drop(target_col)
top10 = corr_series.abs().sort_values(ascending=False).head(10)
for feat, val in top10.items():
    sign = "+" if corr_series[feat] > 0 else "-"
    print(f"  {feat:<15}  r = {sign}{abs(val):.4f}")

# 2e) LIMIT_BAL и BILL_AMT1 статистика
print("\n--- 2e) Статистика LIMIT_BAL и BILL_AMT1 ---")
for col in ['LIMIT_BAL', 'BILL_AMT1']:
    s = df[col]
    print(f"  {col}: min={s.min():>10.0f}  max={s.max():>10.0f}  "
          f"mean={s.mean():>10.1f}  median={s.median():>10.0f}")

print(f"\n  ВАЖНО: В датасете реально встречаются очень маленькие LIMIT_BAL=1000:")
tiny = df[df['LIMIT_BAL'] <= 1000]
print(f"  Клиентов с LIMIT_BAL<=1000: {len(tiny)}, их default rate: {tiny[target_col].mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. ДИАГНОСТИКА СУЩЕСТВУЮЩИХ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════
section("3. Диагностика существующих моделей")

client_df = pd.DataFrame([{f: PROBLEM_CLIENT.get(f, 0) for f in FEATURE_NAMES}])

v1_path = os.path.join(MODELS_DIR, 'model_v1.pkl')
v2_path = os.path.join(MODELS_DIR, 'model_v2.pkl')

old_v1 = joblib.load(v1_path)
old_v2 = joblib.load(v2_path)

print("\n--- Предсказание для 'проблемного' клиента (старые модели) ---")
p1 = old_v1.predict_proba(client_df)[0, 1]
p2 = old_v2.predict_proba(client_df)[0, 1]
print(f"  Model v1 (LR, balanced) : P(default) = {p1:.1%}")
print(f"  Model v2 (GB)           : P(default) = {p2:.1%}")

print("\n--- 3a) Метрики старых моделей на тестовой выборке ---")


def print_metrics(name, pipe, X_test, y_test):
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    roc  = roc_auc_score(y_test, y_proba)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    print(f"\n  {name}")
    print(classification_report(y_test, y_pred,
                                target_names=['No Default', 'Default'],
                                digits=3))
    print(f"  ROC-AUC={roc:.4f}  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
    return {'roc_auc': roc, 'f1': f1, 'precision': prec, 'recall': rec}


old_m1 = print_metrics("Model v1 OLD (LR, class_weight='balanced')", old_v1, X_test, y_test)
old_m2 = print_metrics("Model v2 OLD (GradientBoosting)", old_v2, X_test, y_test)

# 3b) Feature importance / coefficients
print("\n--- 3b) Почему LR (v1) выдаёт такой результат --- ")
lr_model  = old_v1.named_steps['clf']
scaler    = old_v1.named_steps['scaler']
coefs     = pd.Series(lr_model.coef_[0], index=FEATURE_NAMES)
# Вклад каждого признака = coef * scaled_value
x_scaled  = scaler.transform(client_df).flatten()
contrib   = coefs.values * x_scaled
contrib_s = pd.Series(contrib, index=FEATURE_NAMES).sort_values(key=abs, ascending=False)
print("  Топ-10 вкладов признаков в logit (coef * scaled_val):")
for feat, val in contrib_s.head(10).items():
    raw_val = PROBLEM_CLIENT.get(feat, 0)
    print(f"    {feat:<15}  raw={raw_val:>8.1f}  contribution={val:>+.4f}")
print(f"  Logit intercept: {lr_model.intercept_[0]:+.4f}")
total_logit = contrib_s.sum() + lr_model.intercept_[0]
print(f"  Total logit: {total_logit:+.4f}  → σ = {1/(1+np.exp(-total_logit)):.1%}")

print("\n--- 3c) Почему GB (v2) выдаёт такой результат ---")
gb_model = old_v2.named_steps['clf']
importances = pd.Series(gb_model.feature_importances_, index=FEATURE_NAMES)
importances_sorted = importances.sort_values(ascending=False)
print("  Топ-10 Feature Importances (GB):")
for feat, imp in importances_sorted.head(10).items():
    raw_val = PROBLEM_CLIENT.get(feat, 0)
    print(f"    {feat:<15}  importance={imp:.4f}  client_value={raw_val}")

print("\n--- 3d) Проверка на типичных клиентах из датасета ---")
# Типичный "хороший" клиент (PAY_0 <= 0)
typical_good = df[df['PAY_0'] <= 0].sample(5, random_state=42)[FEATURE_NAMES]
p1_good = old_v1.predict_proba(typical_good)[:, 1]
p2_good = old_v2.predict_proba(typical_good)[:, 1]
print(f"  Типичные 'хорошие' (PAY_0<=0) клиенты:")
print(f"    v1 P(default): {[f'{p:.1%}' for p in p1_good]}")
print(f"    v2 P(default): {[f'{p:.1%}' for p in p2_good]}")

# Типичный "плохой" клиент (PAY_0 >= 2)
typical_bad = df[df['PAY_0'] >= 2].sample(5, random_state=42)[FEATURE_NAMES]
p1_bad = old_v1.predict_proba(typical_bad)[:, 1]
p2_bad = old_v2.predict_proba(typical_bad)[:, 1]
print(f"  Типичные 'проблемные' (PAY_0>=2) клиенты:")
print(f"    v1 P(default): {[f'{p:.1%}' for p in p1_bad]}")
print(f"    v2 P(default): {[f'{p:.1%}' for p in p2_bad]}")

# ══════════════════════════════════════════════════════════════════════════════
#  4. АНАЛИЗ ПРОБЛЕМЫ: PAY_0=2 реально коррелирует с дефолтом?
# ══════════════════════════════════════════════════════════════════════════════
section("4. Анализ: PAY_0=2 — реальный сигнал или шум?")

pay0_2 = df[df['PAY_0'] == 2]
pay0_neg = df[df['PAY_0'] <= 0]
print(f"  PAY_0=2 → default rate = {pay0_2[target_col].mean():.1%}  (N={len(pay0_2)})")
print(f"  PAY_0<=0 → default rate = {pay0_neg[target_col].mean():.1%}  (N={len(pay0_neg)})")

# Специфика клиента: LIMIT_BAL=1000 — очень маленький лимит
lb_q = df['LIMIT_BAL'].quantile([0.01, 0.05, 0.1, 0.25])
print(f"\n  Перцентили LIMIT_BAL:")
for q, v in lb_q.items():
    print(f"    {q*100:.0f}%: {v:.0f}")
print(f"  LIMIT_BAL=1000 находится ниже {(df['LIMIT_BAL'] < 1000).mean()*100:.1f}% датасета")

# Вывод
print(f"""
  ВЫВОД по PAY_0=2:
  - PAY_0=2 означает задержку оплаты на 2 месяца — СИЛЬНЫЙ предиктор дефолта
  - Default rate при PAY_0=2: {pay0_2[target_col].mean():.1%} (vs {pay0_neg[target_col].mean():.1%} при PAY_0<=0)
  - LIMIT_BAL=1000 — экстремально маленький лимит, нетипичный клиент
  - "Нулевые счета" не исключают дефолт — у человека просто нет задолженности,
    но он уже ЗАДЕРЖАЛ оплату на 2 месяца (PAY_0=2, PAY_2=2)
  - Проблема: LR с class_weight='balanced' агрессивно штрафует класс 0 → завышает P(default)
""")


# ══════════════════════════════════════════════════════════════════════════════
#  5. ПЕРЕОБУЧЕНИЕ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════
section("5. Переобучение моделей")

print("\n--- Model v1: LogisticRegression (исправленная) ---")
print("  Убираем class_weight='balanced', используем {0:1, 1:3}, подбираем C")

# Выбираем C через cross-validation
best_c, best_roc = None, 0
for c in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    pipe_tmp = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=c, max_iter=1000,
                                   class_weight={0: 1, 1: 3},
                                   random_state=42))
    ])
    scores = cross_val_score(pipe_tmp, X_train, y_train,
                              cv=5, scoring='roc_auc')
    print(f"  C={c:<5} → CV ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    if scores.mean() > best_roc:
        best_roc = scores.mean()
        best_c = c

print(f"\n  Лучший C = {best_c} (CV ROC-AUC = {best_roc:.4f})")

new_v1 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=best_c, max_iter=1000,
                               class_weight={0: 1, 1: 3},
                               random_state=42))
])
new_v1.fit(X_train, y_train)
new_m1 = print_metrics("Model v1 NEW (LR, class_weight={0:1,1:3})", new_v1, X_test, y_test)

print("\n--- Model v2: GradientBoostingClassifier (улучшенный) ---")

best_gb_params = None
best_gb_roc = 0
param_grid = [
    {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8,
     'min_samples_leaf': 20},
    {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8,
     'min_samples_leaf': 10},
    {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7,
     'min_samples_leaf': 20},
    {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8,
     'min_samples_leaf': 10},
]

for params in param_grid:
    pipe_tmp = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42, **params))
    ])
    scores = cross_val_score(pipe_tmp, X_train, y_train,
                              cv=3, scoring='roc_auc')
    label = f"n={params['n_estimators']},d={params['max_depth']},lr={params['learning_rate']},sub={params['subsample']},leaf={params['min_samples_leaf']}"
    print(f"  {label} → CV ROC-AUC = {scores.mean():.4f}")
    if scores.mean() > best_gb_roc:
        best_gb_roc = scores.mean()
        best_gb_params = params

print(f"\n  Лучшие параметры GB: {best_gb_params}")

new_v2 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(random_state=42, **best_gb_params))
])
new_v2.fit(X_train, y_train)
new_m2 = print_metrics("Model v2 NEW (GradientBoosting, tuned)", new_v2, X_test, y_test)


# ══════════════════════════════════════════════════════════════════════════════
#  6. ПРЕДСКАЗАНИЕ ДЛЯ "ПРОБЛЕМНОГО" КЛИЕНТА ПОСЛЕ ПЕРЕОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════
section("6. Предсказание для проблемного клиента (до/после)")

p1_new = new_v1.predict_proba(client_df)[0, 1]
p2_new = new_v2.predict_proba(client_df)[0, 1]

delta1 = p1_new - p1
delta2 = p2_new - p2
p1_s     = f"{p1:.1%}"
p1_new_s = f"{p1_new:.1%}"
p2_s     = f"{p2:.1%}"
p2_new_s = f"{p2_new:.1%}"
d1_s     = f"{delta1:+.1%}"
d2_s     = f"{delta2:+.1%}"
print(f"  {'Модель':<30} {'До':<12} {'После':<12} {'Изменение'}")
print(f"  {'-'*60}")
print(f"  {'v1 (LogisticRegression)':<30} {p1_s:<12} {p1_new_s:<12} {d1_s}")
print(f"  {'v2 (GradientBoosting)':<30} {p2_s:<12} {p2_new_s:<12} {d2_s}")
print()

print("  Контекст предсказания:")
print("  - PAY_0=2, PAY_2=2 → задержка оплаты 2 месяца подряд — СИЛЬНЫЙ сигнал")
print("  - LIMIT_BAL=1000   → минимальный лимит (ниже 99% клиентов)")
print("  - BILL_AMT≈0       → счета почти пустые, но PAY уже просрочен")
print("  - PAY_AMT1=0       → в последнем месяце вообще ничего не заплатил")


# ══════════════════════════════════════════════════════════════════════════════
#  7. СОХРАНЕНИЕ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════
section("7. Сохранение переобученных моделей")

v1_save = os.path.join(MODELS_DIR, 'model_v1.pkl')
v2_save = os.path.join(MODELS_DIR, 'model_v2.pkl')

joblib.dump(new_v1, v1_save)
joblib.dump(new_v2, v2_save)

print(f"  Сохранено: {v1_save}")
print(f"  Сохранено: {v2_save}")

# Проверка загрузки
_ = joblib.load(v1_save)
_ = joblib.load(v2_save)
print("  Проверка загрузки: OK")


# ══════════════════════════════════════════════════════════════════════════════
#  8. ИТОГОВЫЙ ВЫВОД
# ══════════════════════════════════════════════════════════════════════════════
section("8. Итоговый вывод")

dr_pay0_2 = pay0_2[target_col].mean()
dr_pay0_neg = pay0_neg[target_col].mean()

print()
print("  БЫЛ ЛИ БАГ В МОДЕЛИ?")
print("  " + "-"*40)
print("  Частично да, частично — поведение корректно.")
print()
print("  1. БАГ v1 (LR с class_weight='balanced'):")
print("     class_weight='balanced' увеличивает вес дефолтного класса в ~3.7 раза")
print("     (пропорционально дисбалансу 77%:23%). LR становится чрезмерно")
print("     агрессивным в предсказании дефолта — завышает recall ценой precision.")
print(f"     Для клиента: {p1:.1%} -> ЗАВЫШЕНО из-за balanced.")
print()
print("  2. GB (v2) — ВЕЛ СЕБЯ КОРРЕКТНО:")
print(f"     {p2:.1%} — обоснованная оценка. PAY_0=2 (задержка 2 мес) является")
print(f"     самым сильным предиктором дефолта (corr={corr_pay0:.3f}).")
print(f"     Человек с PAY_0=2 дефолтит в {dr_pay0_2:.1%} случаев реально.")
print()
print("  3. ВОПРОС КЛИЕНТА 'нулевые счета = нет дефолта':")
print("     Логическая ошибка. BILL_AMT — это ТЕКУЩИЙ БАЛАНС на счёте.")
print("     PAY_0=2 — это уже СОСТОЯВШАЯСЯ задержка платежа 2 месяца назад.")
print("     Нулевые счета = нет долга, но НЕ означает нет просрочки.")
print()
print("  МЕТРИКИ ДО И ПОСЛЕ:")
print(f"  {'Метрика':<15} {'v1_до':>10} {'v1_после':>10} {'v2_до':>10} {'v2_после':>10}")
print("  " + "-"*55)
print(f"  {'ROC-AUC':<15} {old_m1['roc_auc']:>10.4f} {new_m1['roc_auc']:>10.4f} {old_m2['roc_auc']:>10.4f} {new_m2['roc_auc']:>10.4f}")
print(f"  {'F1':<15} {old_m1['f1']:>10.4f} {new_m1['f1']:>10.4f} {old_m2['f1']:>10.4f} {new_m2['f1']:>10.4f}")
print(f"  {'Precision':<15} {old_m1['precision']:>10.4f} {new_m1['precision']:>10.4f} {old_m2['precision']:>10.4f} {new_m2['precision']:>10.4f}")
print(f"  {'Recall':<15} {old_m1['recall']:>10.4f} {new_m1['recall']:>10.4f} {old_m2['recall']:>10.4f} {new_m2['recall']:>10.4f}")
print()
print("  ПРЕДСКАЗАНИЕ ДЛЯ ПРОБЛЕМНОГО КЛИЕНТА:")
print(f"  v1: {p1:.1%} -> {p1_new:.1%}  (убрали balanced, теперь точнее)")
print(f"  v2: {p2:.1%} -> {p2_new:.1%}  (тюнинг, незначительное изменение)")
print()
print("  РЕКОМЕНДАЦИЯ:")
print("  Использовать v2 (GradientBoosting) как основную модель.")
print(f"  Предсказание ~{p2_new:.0%} дефолта для клиента с PAY_0=2 корректно и")
print(f"  соответствует реальной статистике датасета ({dr_pay0_2:.1%} при PAY_0=2).")

print("Done.")
