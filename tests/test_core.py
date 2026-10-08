"""
Basic tests: run with  python -m pytest -q
They prove the pipeline works end to end and catch the most likely mistakes.
"""

import numpy as np
import pandas as pd
from src import kpi
from src.db import load_csvs_to_db
from src.model import add_features, best_threshold, predict_one


def setup_module(module):                               # pytest runs this once before the tests in this file
    load_csvs_to_db()                                    # make sure factory.db is fresh


def test_oee_between_0_and_1():
    oee = kpi.oee_by_station()
    assert oee["oee"].between(0, 1).all()                # a percentage can never be outside 0-100%


def test_oee_equals_product_of_parts():
    oee = kpi.oee_by_station()
    product = oee["availability"] * oee["performance"] * oee["quality"]
    assert np.allclose(product, oee["oee"], atol=1e-3)   # OEE = A x P x Q (allowing rounding)


def test_line_filter():
    assert set(kpi.oee_by_station("LINE_A")["line_id"]) == {"LINE_A"}   # filter returns only that line


def test_power_feature():
    df = add_features(pd.DataFrame([{"torque_nm": 10, "rpm": 60, "process_temp_k": 310,
                                     "air_temp_k": 300, "tool_wear_min": 5}]))
    assert abs(df["power_w"].iloc[0] - 10 * 2 * np.pi) < 1e-6          # 60 rpm = 1 rev/s = 2*pi rad/s


def test_threshold_prefers_recall_when_misses_are_expensive():
    y = np.array([0, 0, 0, 1])
    proba = np.array([0.1, 0.2, 0.4, 0.45])
    assert best_threshold(y, proba, cost_missed=100, cost_false_alarm=1) <= 0.45   # must still catch the failure


def test_predict_one_returns_probability():
    out = predict_one({"machine_type": "M", "air_temp_k": 300, "process_temp_k": 310,
                       "rpm": 1500, "torque_nm": 40, "tool_wear_min": 50})
    assert 0 <= out["failure_probability"] <= 1
