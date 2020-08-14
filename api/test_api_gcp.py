import os
import tempfile
import requests
import pytest

from flask import Flask

URL_BASE = 'http://localhost:5000/'


# @pytest.fixture
# def client():
#     flask.app.config['TESTING'] = True

#     with flask.app.test_client() as client:
#         yield client

def test_base_route():
    out = requests.get(URL_BASE + 'hello')
    assert out.text == 'oi'

def test_pyequion_api():
    params = {
    "method": "App.create_equilibrium",
    "params": {
            "compounds": ["NaCl"],
            "closingEqType": 0,
            "initial_feed_mass_balance": ["Cl-"]
        }
    }
    out = requests.post(URL_BASE + 'api', json=params)
    assert out.text

if __name__ == "__main__":
    test_pyequion_api()


