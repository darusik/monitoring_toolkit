# MIT License MIT
#
# Copyright © 2025 Daryna Oliynyk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
#  persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS 
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from monitoring_toolkit.detectors.confidence_detector import ConfidenceDetector
from monitoring_toolkit.detectors.result import DetectionResult
from monitoring_toolkit.utils.query import Query


# --- Entropy mode ---
def test_entropy_high_entropy_triggers_alert():
    detector = ConfidenceDetector(config={"mode": "entropy", "threshold": 0.5})
    uniform_probs = [0.25, 0.25, 0.25, 0.25]  # max entropy
    query = Query(input_data="Hello world!", model_output=uniform_probs)
    result = detector.process(query)
    assert result.is_suspicious

def test_entropy_low_entropy_not_suspicious():
    detector = ConfidenceDetector(config={"mode": "entropy", "threshold": 0.5})
    confident_probs = [0.95, 0.03, 0.02]
    query = Query(input_data="Hello world!", model_output=confident_probs)
    result = detector.process(query)
    assert not result.is_suspicious

# --- Max confidence mode ---
def test_max_confidence_low_confidence_is_suspicious():
    detector = ConfidenceDetector(config={"mode": "max_confidence", "threshold": 0.3})
    probs = [0.4, 0.3, 0.3]
    query = Query(input_data="Hello world!", model_output=probs)
    result = detector.process(query)
    assert result.is_suspicious

def test_max_confidence_high_confidence_not_suspicious():
    detector = ConfidenceDetector(config={"mode": "max_confidence", "threshold": 0.3})
    probs = [0.9, 0.05, 0.05]
    query = Query(input_data="Hello world!", model_output=probs)
    result = detector.process(query)
    assert not result.is_suspicious

# --- Margin mode ---
def test_margin_small_margin_is_suspicious():
    detector = ConfidenceDetector(config={"mode": "margin", "threshold": 0.3})
    probs = [0.4, 0.39, 0.21]
    query = Query(input_data="Hello world!", model_output=probs)
    result = detector.process(query)
    assert result.is_suspicious

def test_margin_large_margin_not_suspicious():
    detector = ConfidenceDetector(config={"mode": "margin", "threshold": 0.3})
    probs = [0.9, 0.05, 0.05]
    query = Query(input_data="Hello world!", model_output=probs)
    result = detector.process(query)
    assert not result.is_suspicious

# --- Batch processing ---
def test_process_batch():
    detector = ConfidenceDetector(config={"mode": "entropy", "threshold": 0.5})
    queries = [
        Query(input_data="Hello world!", model_output=[0.25, 0.25, 0.25, 0.25]),
        Query(input_data="Goodbye world!", model_output=[0.9, 0.05, 0.05])
    ]
    results = detector.process_batch(queries)
    num_results = 0
    for result in results:
        num_results += 1
        assert isinstance(result, DetectionResult)
        if num_results == 1:
            assert result.is_suspicious
        elif num_results == 2:
            assert not result.is_suspicious
    assert num_results == 2


# --- Get state ---
def test_get_state():
    detector = ConfidenceDetector(config={"mode": "entropy", "threshold": 0.5})
    query = Query(input_data="Hello world!", model_output=[0.25, 0.25, 0.25, 0.25])
    detector.process(query)
    state = detector.get_state()
    assert state["last_score"] - 1.38629436071989054 < 10e-10 
    assert state["avg_score"] - 1.38629436071989054 < 10e-10
    assert state["num_scores"] == 1
