import os
import logging

from flask import Flask

from .model_handler import ModelHandler
from .ab_testing import ABTestingManager
from .logger import setup_logger

logging.getLogger('werkzeug').setLevel(logging.WARNING)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False  # кириллица без unicode-escape

    os.makedirs('logs', exist_ok=True)

    app.logger_json = setup_logger('ml_service', 'logs/app.log')

    model_paths = {
        'v1': os.environ.get('MODEL_V1_PATH', 'models/model_v1.pkl'),
        'v2': os.environ.get('MODEL_V2_PATH', 'models/model_v2.pkl'),
    }
    app.model_handler = ModelHandler(model_paths)

    app.ab_manager = ABTestingManager(versions=['v1', 'v2'], split=0.5)

    from .api import api_bp
    app.register_blueprint(api_bp)

    app.logger_json.info(
        'Service started',
        extra={'event': 'startup', 'loaded_models': app.model_handler.available_versions},
    )
    return app
