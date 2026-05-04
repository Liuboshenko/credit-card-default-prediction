"""
API integration tests.

Run from the project root:
    pytest tests/ -v

Models must be trained before running:
    python models/train_model.py
"""
import sys
import os

# Ensure project root is on path when running via pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app import create_app


VALID_SAMPLE = {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0,
}


@pytest.fixture(scope='module')
def client():
    flask_app = create_app()
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as c:
        yield c


# ── /health ─────────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_200(self, client):
        resp = client.get('/health')
        assert resp.status_code == 200

    def test_body_structure(self, client):
        data = client.get('/health').get_json()
        assert data['status'] == 'healthy'
        assert 'models' in data
        assert 'v1' in data['models']
        assert 'v2' in data['models']
        assert 'timestamp' in data


# ── /predict  (v1) ──────────────────────────────────────────────────────────

class TestPredictV1:
    def test_valid_request(self, client):
        resp = client.post('/predict', json=VALID_SAMPLE)
        assert resp.status_code == 200

    def test_response_fields(self, client):
        data = client.post('/predict', json=VALID_SAMPLE).get_json()
        assert 'prediction' in data
        assert 'probability' in data
        assert 'default' in data
        assert data['model_version'] == 'v1'

    def test_prediction_is_binary(self, client):
        data = client.post('/predict', json=VALID_SAMPLE).get_json()
        assert data['prediction'] in (0, 1)

    def test_probability_in_range(self, client):
        data = client.post('/predict', json=VALID_SAMPLE).get_json()
        assert 0.0 <= data['probability'] <= 1.0

    def test_default_matches_prediction(self, client):
        data = client.post('/predict', json=VALID_SAMPLE).get_json()
        assert data['default'] == bool(data['prediction'])

    def test_missing_feature_returns_400(self, client):
        bad_sample = dict(VALID_SAMPLE)
        del bad_sample['LIMIT_BAL']
        resp = client.post('/predict', json=bad_sample)
        assert resp.status_code == 400
        assert 'error' in resp.get_json()

    def test_non_json_returns_400(self, client):
        resp = client.post('/predict', data='not-json', content_type='text/plain')
        assert resp.status_code == 400

    def test_empty_body_returns_400(self, client):
        resp = client.post('/predict', json={})
        assert resp.status_code == 400


# ── /predict/v2 ─────────────────────────────────────────────────────────────

class TestPredictV2:
    def test_valid_request(self, client):
        resp = client.post('/predict/v2', json=VALID_SAMPLE)
        assert resp.status_code == 200

    def test_model_version_v2(self, client):
        data = client.post('/predict/v2', json=VALID_SAMPLE).get_json()
        assert data['model_version'] == 'v2'

    def test_unknown_version_returns_400(self, client):
        resp = client.post('/predict/v99', json=VALID_SAMPLE)
        assert resp.status_code == 400


# ── /ab/predict ─────────────────────────────────────────────────────────────

class TestABPredict:
    def test_valid_request(self, client):
        resp = client.post('/ab/predict', json=VALID_SAMPLE)
        assert resp.status_code == 200

    def test_ab_group_present(self, client):
        data = client.post('/ab/predict', json=VALID_SAMPLE).get_json()
        assert data['ab_group'] in ('v1', 'v2')

    def test_ab_group_equals_model_version(self, client):
        data = client.post('/ab/predict', json=VALID_SAMPLE).get_json()
        assert data['ab_group'] == data['model_version']

    def test_both_versions_appear_over_many_requests(self, client):
        """Over 100 calls both groups should appear (probabilistic, may rarely fail)."""
        groups = {client.post('/ab/predict', json=VALID_SAMPLE).get_json()['ab_group']
                  for _ in range(100)}
        assert 'v1' in groups
        assert 'v2' in groups


# ── /ab/stats ────────────────────────────────────────────────────────────────

class TestABStats:
    def test_status_200(self, client):
        resp = client.get('/ab/stats')
        assert resp.status_code == 200

    def test_stats_structure(self, client):
        data = client.get('/ab/stats').get_json()
        assert 'ab_stats' in data
        assert 'v1' in data['ab_stats']
        assert 'v2' in data['ab_stats']

    def test_stats_fields(self, client):
        data = client.get('/ab/stats').get_json()
        for version in ('v1', 'v2'):
            s = data['ab_stats'][version]
            assert 'requests' in s
            assert 'default_rate' in s
            assert 'avg_probability' in s
