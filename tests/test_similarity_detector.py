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
import pandas as pd
from PIL import Image


from detectors.similarity_detector import ImageSimilarityDetector, TabularSimilarityDetector, TextSimilarityDetector    
from detectors.result import DetectionResult
from utils.query import Query


# --- Image tests ---
def test_image_detector_with_pil_single_query():
    detector = ImageSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
    img = Image.new('RGB', (224, 224), color='white')
    req = Query(input_data=img)
    result = detector.process(req)
    assert not result.is_suspicious

def test_image_detector_with_numpy_hwc_same_query_twice():
    detector = ImageSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
    img = np.ones((224, 224, 3), dtype='uint8')
    req = Query(input_data=img)
    result1 = detector.process(req)
    result2 = detector.process(req)
    assert not result1.is_suspicious
    assert result2.is_suspicious
    assert result2.confidence == 1.0

def test_image_detector_with_numpy_chw_two_different_queries():
    detector = ImageSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
    img1 = np.ones((3, 224, 224), dtype='uint8')
    img2 = np.random.randint(0, 10, size=(3, 224, 224), dtype='uint8')
    req1 = Query(input_data=img1)
    req2 = Query(input_data=img2)
    result1 = detector.process(req1)
    result2 = detector.process(req2)
    assert not result1.is_suspicious
    assert not result2.is_suspicious


def test_image_detector_with_torch_two_not_similar_enough_queries():
    detector = ImageSimilarityDetector(config={"threshold": 0.999, "max_history_size": 10})
    img1 = torch.ones((3, 224, 224), dtype=torch.float)
    img2 = 0.85*torch.ones((3, 224, 224), dtype=torch.float)
    req1 = Query(input_data=img1)
    req2 = Query(input_data=img2)
    result1 = detector.process(req1)
    result2 = detector.process(req2)
    assert not result1.is_suspicious
    assert not result2.is_suspicious
    assert result2.confidence < 1.0  # Should be similar but not identical
    assert result2.confidence > 0.9  # Should be above threshold


# --- Text tests ---
def test_text_detector_with_str_same_query_twice():
    detector = TextSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
    text = "Hello world!"
    req = Query(input_data=text)
    result1 = detector.process(req)
    result2 = detector.process(req)
    assert not result1.is_suspicious
    assert result2.is_suspicious

def test_text_detector_with_list_with_different_queries():
    detector = TextSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
    text1 = ["Hello", "world!"]
    text2 = ["Goodbye", "world!"]
    req1 = Query(input_data=text1)
    req2 = Query(input_data=text2)
    result1 = detector.process(req1)
    result2 = detector.process(req2)
    assert not result1.is_suspicious
    assert not result2.is_suspicious

# --- Tabular tests ---
def test_tabular_detector_with_dict_similar_queries():
    detector = TabularSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
    row1 = {"age": 25, "income": 30000}
    row2 = {"age": 24, "income": 30001}
    req1 = Query(input_data=row1)
    req2 = Query(input_data=row2)
    result1 = detector.process(req1)
    result2 = detector.process(req2)
    assert not result1.is_suspicious
    assert result2.is_suspicious
    assert result2.confidence > 0.9  # Should be above threshold

def test_tabular_detector_with_pd_series_different_queries():
    detector = TabularSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
    row1 = pd.Series({"age": 25, "income": 30000})
    row2 = pd.Series({"length": 50, "width": 100})
    req1 = Query(input_data=row1)
    req2 = Query(input_data=row2)
    result1 = detector.process(req1)
    result2 = detector.process(req2)
    assert not result1.is_suspicious
    assert not result2.is_suspicious

# --- Batch tests ---
def test_process_batch():
    detector = TextSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
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
    detector = TextSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
    text = "Hello world!"
    req = Query(input_data=text)
    detector.process(req)
    detector.reset_state()
    state = detector.get_state()
    assert state["current_history_size"] == 0
    assert state["latest_incident_similarity_score"] is None
    assert state["latest_incident_closest_index"] is None
    assert state["latest_incident_closest_embedding"] is None

def test_get_state():
    detector = TextSimilarityDetector(config={"threshold": 0.9, "max_history_size": 10})
    text = "Hello world!"
    req = Query(input_data=text)
    detector.process(req)
    detector.process(req)  # Process the same query again to create incident
    state = detector.get_state(include_embedding=True)
    assert state["current_history_size"] == 2
    assert state["latest_incident_similarity_score"] == 1.0  # Should be identical
    assert state["latest_incident_closest_index"] == 0  # Index of the first query
    assert state["latest_incident_closest_embedding"] is not None  # Should have a valid embedding