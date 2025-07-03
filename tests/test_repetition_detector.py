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

import numpy as np
import torch
from PIL import Image
import pandas as pd

from monitoring_toolkit.detectors.repetition_detector import ImageRepetitionDetector, TabularRepetitionDetector, TextRepetitionDetector
from monitoring_toolkit.detectors.result import DetectionResult
from monitoring_toolkit.utils.query import Query

# --- Image tests ---
def test_image_detector_with_pil():
    detector = ImageRepetitionDetector(config={"threshold": 4, "max_history_size": 10})
    img = Image.new('RGB', (224, 224), color='white')
    req = Query(input_data=img)
    result = detector.process(req)
    result = detector.process(req)
    assert not result.is_suspicious

def test_image_detector_with_numpy():
    detector = ImageRepetitionDetector(config={"threshold": 2, "max_history_size": 10})
    img = np.ones((224, 224, 3), dtype='uint8')
    req = Query(input_data=img)
    result = detector.process(req)
    result = detector.process(req)
    assert result.is_suspicious
    assert result.confidence == 1.0

def test_image_detector_with_torch():
    detector = ImageRepetitionDetector(config={"threshold": 2})
    img = torch.ones((3, 224, 224), dtype=torch.uint8)
    req = Query(input_data=img)
    result = detector.process(req)
    assert not result.is_suspicious

# --- Text tests ---
def test_text_detector_with_str():
    detector = TextRepetitionDetector(config={"threshold": 2})
    text = "Hello world!"
    req = Query(input_data=text)
    result = detector.process(req)
    result = detector.process(req)
    assert result.is_suspicious

def test_text_detector_with_list():
    detector = TextRepetitionDetector(config={"threshold": 2})
    text = ["Hello", "world!"]
    req = Query(input_data=text)
    result = detector.process(req)
    assert not result.is_suspicious

# --- Tabular tests ---
def test_tabular_detector_with_dict():
    detector = TabularRepetitionDetector(config={"threshold": 2})
    row = {"age": 25, "income": 30000}
    req = Query(input_data=row)
    result = detector.process(req)
    result = detector.process(req)
    assert result.is_suspicious

def test_tabular_detector_with_pd_series():
    detector = TabularRepetitionDetector(config={"threshold": 2})
    row = pd.Series({"age": 25, "income": 30000})
    req = Query(input_data=row)
    result = detector.process(req)
    result = detector.process(req)
    assert result.is_suspicious

# --- Batch tests ---
def test_process_batch():
    detector = TextRepetitionDetector(config={"threshold": 2})
    queries = [Query(input_data="test") for _ in range(3)]
    results = detector.process_batch(queries)
    num_results = 0
    for result in results:
        num_results += 1
        assert isinstance(result, DetectionResult)
        if num_results > 1:
            assert result.is_suspicious
        else:
            assert not result.is_suspicious
    assert num_results == 3

# --Reset and State Tests--
def test_reset_state():
    detector = TextRepetitionDetector(config={"threshold": 2})
    text = "Hello world!"
    req = Query(input_data=text)
    detector.process(req)
    detector.reset_state()
    state = detector.get_state()
    assert state["current_history_size"] == 0
    assert state["unique_inputs"] == 0
    assert state["max_repetition"] == 0
    assert state["repetition_counts"] == {}

def test_get_state():
    detector = TextRepetitionDetector(config={"threshold": 2})
    text = "Hello world!"
    req = Query(input_data=text)
    detector.process(req)
    state = detector.get_state()
    assert state["current_history_size"] == 1
    assert state["unique_inputs"] == 1
    assert state["max_repetition"] == 1
    assert state["repetition_counts"] == {'7509e5bda0c762d2bac7f90d758b5b2263fa01ccbc542ab5e3df163be08e6ca9': 1}