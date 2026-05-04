import datetime
from flask import Blueprint, request, jsonify, current_app, render_template

from src.preprocessing import validate_and_preprocess

api_bp = Blueprint('api', __name__)


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + 'Z'


@api_bp.route('/', methods=['GET'])
def index():
    """Главная страница — браузерный UI."""
    return render_template('index.html')


def _run_inference(version: str, data: dict):
    features = validate_and_preprocess(data)
    return current_app.model_handler.predict(version, features)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@api_bp.route('/health', methods=['GET'])
def health():
    """
    Health check.

    Response 200:
        {
          "status": "healthy",
          "service": "credit-card-default-prediction",
          "models": ["v1", "v2"],
          "timestamp": "2024-01-01T00:00:00Z"
        }
    """
    return jsonify({
        'status': 'healthy',
        'service': 'credit-card-default-prediction',
        'models': current_app.model_handler.available_versions,
        'timestamp': _utcnow(),
    }), 200


# ---------------------------------------------------------------------------
# POST /predict          — shortcut for model v1
# POST /predict/<version> — explicit version (v1 | v2)
# ---------------------------------------------------------------------------

@api_bp.route('/predict', methods=['POST'])
def predict_default():
    """Predict using default model (v1). Delegates to predict_version."""
    return _predict('v1')


@api_bp.route('/predict/<version>', methods=['POST'])
def predict_version(version: str):
    """
    Predict using an explicit model version.

    URL params:
        version — "v1" (LogisticRegression) or "v2" (GradientBoosting)

    Request body (JSON):
        See FEATURE_NAMES in config.py (23 numeric fields).

    Response 200:
        {
          "prediction": 0 | 1,
          "probability": 0.123,
          "default": false | true,
          "model_version": "v1",
          "timestamp": "..."
        }

    Response 400:
        {"error": "<reason>"}
    """
    return _predict(version)


def _predict(version: str):
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object'}), 400

        if version not in current_app.model_handler.available_versions:
            return jsonify({
                'error': f"Unknown model version '{version}'",
                'available': current_app.model_handler.available_versions,
            }), 400

        prediction, probability = _run_inference(version, data)

        current_app.logger_json.info(
            'prediction',
            extra={
                'event': 'prediction',
                'model_version': version,
                'prediction': prediction,
                'probability': round(probability, 4),
            },
        )

        return jsonify({
            'prediction': prediction,
            'probability': round(probability, 4),
            'default': bool(prediction),
            'model_version': version,
            'timestamp': _utcnow(),
        }), 200

    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        current_app.logger_json.error('unexpected_error', extra={'event': 'error', 'detail': str(exc)})
        return jsonify({'error': 'Internal server error'}), 500


# ---------------------------------------------------------------------------
# POST /ab/predict  — random 50/50 split between v1 and v2
# GET  /ab/stats    — cumulative A/B statistics
# ---------------------------------------------------------------------------

@api_bp.route('/ab/predict', methods=['POST'])
def ab_predict():
    """
    A/B testing endpoint.  Randomly routes to v1 (50 %) or v2 (50 %).

    Response 200 (same shape as /predict, plus ab_group):
        {
          "prediction": 0 | 1,
          "probability": 0.123,
          "default": false,
          "model_version": "v2",
          "ab_group": "v2",
          "timestamp": "..."
        }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object'}), 400

        version = current_app.ab_manager.assign_version()
        prediction, probability = _run_inference(version, data)
        current_app.ab_manager.record(version, prediction, probability)

        current_app.logger_json.info(
            'ab_prediction',
            extra={
                'event': 'ab_prediction',
                'ab_group': version,
                'prediction': prediction,
                'probability': round(probability, 4),
            },
        )

        return jsonify({
            'prediction': prediction,
            'probability': round(probability, 4),
            'default': bool(prediction),
            'model_version': version,
            'ab_group': version,
            'timestamp': _utcnow(),
        }), 200

    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        current_app.logger_json.error('unexpected_error', extra={'event': 'error', 'detail': str(exc)})
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/ab/stats', methods=['GET'])
def ab_stats():
    """
    Cumulative A/B test statistics (in-memory).

    Response 200:
        {
          "ab_stats": {
            "v1": {"requests": 52, "default_rate": 0.21, "avg_probability": 0.19},
            "v2": {"requests": 48, "default_rate": 0.17, "avg_probability": 0.15}
          },
          "timestamp": "..."
        }
    """
    return jsonify({
        'ab_stats': current_app.ab_manager.get_stats(),
        'timestamp': _utcnow(),
    }), 200
