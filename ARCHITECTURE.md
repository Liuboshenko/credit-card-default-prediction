# Архитектура сервиса прогнозирования дефолта

## 1. Монолит vs Микросервисы

### Анализ

| Критерий | Монолит | Микросервисы |
|---|---|---|
| Сложность деплоя | Минимальная (один контейнер) | Требует оркестратора (Docker Compose, K8s) |
| Масштабирование | Вертикальное (больше RAM/CPU) | Горизонтальное (добавить реплики ML-сервиса) |
| Изоляция сбоев | Ошибка роняет весь сервис | Сбой одного компонента не убивает систему |
| Независимый деплой | Невозможен | v2 можно выкатить без остановки v1 |
| Латентность инференса | Минимальная | +сетевой hop |

### Выбор и обоснование

В данном проекте выбрана **микросервисная архитектура** из двух слоёв:

```
Клиент
  │
  ▼
┌──────────────────┐
│  nginx (порт 80) │  ← обратный прокси, TLS-терминация, балансировка
└────────┬─────────┘
         │ proxy_pass
         ▼
┌──────────────────────────────────┐
│  ml-service (Flask/Gunicorn :5000) │  ← инференс обеих версий модели
│  • POST /predict                 │
│  • POST /predict/v1              │
│  • POST /predict/v2              │
│  • POST /ab/predict              │
│  • GET  /ab/stats                │
│  • GET  /health                  │
└──────────────────────────────────┘
         │
         ▼
   logs/app.log  (JSON)
```

**Почему микросервисы, а не монолит:**
1. **Независимое масштабирование**: если нагрузка растёт — добавляются реплики только `ml-service`, nginx остаётся в одном экземпляре.
2. **Separation of concerns**: nginx обрабатывает TLS, кэш статики, rate-limiting; Flask занят только инференсом.
3. **Гибкость A/B-теста**: в будущем v1 и v2 можно вынести в два отдельных сервиса, а nginx/балансировщик будет разделять трафик на уровне HTTP.
4. **Учебный контекст**: проект демонстрирует production-паттерн, который напрямую переносится в облако (ECS, GKE и т.д.).

---

## 2. RabbitMQ — концепт брокера сообщений

### Для чего нужен

В сценарии с высокой нагрузкой или батч-предсказаниями синхронный HTTP-запрос не оптимален. Брокер очередей решает три задачи:

```
┌────────────┐   publish   ┌──────────────┐   consume   ┌─────────────┐
│  API-шлюз  │ ──────────► │   RabbitMQ   │ ──────────► │ ML-Worker   │
│  (принял   │             │   Queue:     │             │ (batch      │
│   запрос)  │             │  predictions │             │  inference) │
└────────────┘             └──────────────┘             └──────┬──────┘
                                                               │ результат
                                                        ┌──────▼──────┐
                                                        │    DB/Cache │
                                                        └─────────────┘
```

| Сценарий | Без брокера | С RabbitMQ |
|---|---|---|
| Батч из 10 000 клиентов | Таймаут HTTP | Все задачи в очередь, N воркеров параллельно |
| Пиковая нагрузка | Падение сервиса | Очередь амортизирует пики |
| Аудит/логирование | Синхронно замедляет ответ | Событие в очередь → async обработка |

**Пример реализации (концепт):**
```python
# publisher (API-шлюз)
import pika, json

def publish_prediction_request(features: dict):
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    channel.queue_declare(queue='predictions', durable=True)
    channel.basic_publish(
        exchange='',
        routing_key='predictions',
        body=json.dumps(features),
        properties=pika.BasicProperties(delivery_mode=2),  # persistent
    )
    connection.close()

# consumer (ML-воркер)
def on_message(ch, method, properties, body):
    features = json.loads(body)
    result = model.predict(features)
    save_to_db(result)
    ch.basic_ack(delivery_tag=method.delivery_tag)
```

---

## 3. Логирование и мониторинг

### Формат логов

Все события пишутся в **JSON** через `app/logger.py`:

```json
{
  "timestamp": "2024-05-01T12:00:00.123Z",
  "level": "INFO",
  "logger": "ml_service",
  "message": "prediction",
  "event": "prediction",
  "model_version": "v1",
  "prediction": 1,
  "probability": 0.7423
}
```

### Куда пишутся логи

| Слой | Место | Формат |
|---|---|---|
| Flask / Gunicorn | `logs/app.log` (JSON) + stdout | JSON |
| Nginx | stdout контейнера (`docker logs`) | Combined |
| Система | Docker log driver (journald/json-file) | JSON |

### ELK-стек в production

```
logs/app.log → Filebeat → Logstash → Elasticsearch → Kibana
```

Kibana dashboards позволяют строить:
- Дашборд по default_rate в разрезе модели (v1 vs v2)
- Алерт при росте вероятности дефолта > порога
- Временны́е ряды латентности инференса

---

## 4. MLOps-инструменты (концепт)

### DVC (Data Version Control)

Решает проблему «какие данные использовались для этого эксперимента»:

```bash
dvc init
dvc add data/UCI_Credit_Card.csv      # версионировать датасет
dvc run -n train \
    -d data/UCI_Credit_Card.csv \
    -d models/train_model.py \
    -o models/model_v1.pkl \
    python models/train_model.py
git add dvc.lock && git commit -m "train v1"
```

При следующем изменении данных `dvc repro` автоматически переобучит модель.

### MLflow

Отслеживание экспериментов без изменения бизнес-логики:

```python
import mlflow

with mlflow.start_run(run_name="logreg_baseline"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metric("roc_auc", 0.772)
    mlflow.log_metric("f1", 0.524)
    mlflow.sklearn.log_model(pipeline_v1, "model")
```

UI `mlflow ui` позволяет визуально сравнивать прогоны, выбирать чемпиона и регистрировать его в Model Registry для последующего деплоя.

---

## 5. Бизнес-метрики

Технические метрики (F1, ROC-AUC) важны для выбора модели, но заказчик мыслит в деньгах:

### Метрика 1: Ожидаемые предотвращённые потери (EPL)

```
EPL = Σ [ P(default_i) × exposure_i ]
    - Σ [ TP_i × LGD_i ]           # кредиты, дефолт по которым предсказан верно
```

Где:
- `exposure_i` — текущий баланс долга клиента i
- `LGD` (Loss Given Default) — доля от баланса, которую банк не вернёт (~40-60 %)
- TP = True Positive (правильно предсказан дефолт → банк не выдал новый кредит)

**Расчёт на выходах модели:**
```python
# df — тестовая выборка с колонками y_true, y_pred, LIMIT_BAL
LGD = 0.45
df['prevented_loss'] = (df['y_pred'] == 1) & (df['y_true'] == 1) * df['LIMIT_BAL'] * LGD
total_epl = df['prevented_loss'].sum()
```

### Метрика 2: Процент одобрения при фиксированном уровне риска

```
ApprovalRate@Risk = |{клиенты с P(default) < threshold}| / N
```

Более строгая модель (меньше FP) позволяет снизить порог отсечения и одобрить больше клиентов, сохраняя допустимый уровень риска. Это прямой доход банка.

---

## 6. ONNX-ML (обзорно)

ONNX (Open Neural Network Exchange) позволяет экспортировать sklearn-модели в переносимый формат, не зависящий от Python:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 23]))]
onnx_model = convert_sklearn(pipeline_v1, initial_types=initial_type)
with open("models/model_v1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Преимущества ONNX:**
- Инференс через `onnxruntime` работает в 2-5x быстрее чем scikit-learn на CPU
- Модель можно запустить в C++, Go, Java-сервисах без Python
- Уменьшает размер Docker-образа (нет scipy/sklearn в runtime)

---

## 7. uWSGI + NGINX в production

| Компонент | Роль |
|---|---|
| **Flask** | Фреймворк для описания маршрутов; _не_ предназначен для production нагрузки |
| **Gunicorn / uWSGI** | WSGI-сервер: управляет пулом Python-воркеров, graceful restart, таймауты |
| **NGINX** | Принимает внешние соединения, терминирует TLS, кэширует, rate-limit, балансирует |

```
Internet → NGINX (TLS, rate-limit) → Gunicorn (workers) → Flask app → ML model
```

Flask в режиме `debug=True` однопоточен и небезопасен. Gunicorn с `--workers 4` обрабатывает 4 запроса одновременно; добавление `--worker-class gevent` переключает на async I/O без изменения кода.
