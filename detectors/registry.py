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
"""
This module defines a registry for various detectors, allowing dynamic retrieval and instantiation of detector classes.
"""


import inspect

from detectors.base_detector import BaseDetector
from detectors.confidence_detector import ConfidenceDetector
from detectors.repetition_detector import (   RepetitionDetector,
    ImageRepetitionDetector,
    TabularRepetitionDetector,
    TextRepetitionDetector
)
from detectors.similarity_detector import (   SimilarityDetector,
    ImageSimilarityDetector,
    TabularSimilarityDetector,
    TextSimilarityDetector
)

DETECTOR_REGISTRY = {
    "confidence": {
        "class": ConfidenceDetector,
        "modality": "any",
        "base": "confidence_detector"
    },
    "repetition": {
        "class": RepetitionDetector,
        "modality": "any",
        "base": "repetition_detector"
    },
    "image_repetition": {
        "class": ImageRepetitionDetector,
        "modality": "image",
        "base": "repetition_detector"
    },
    "tabular_repetition": {
        "class": TabularRepetitionDetector,
        "modality": "tabular",
        "base": "repetition_detector"
    },
    "text_repetition": {
        "class": TextRepetitionDetector,
        "modality": "text",
        "base": "repetition_detector"
    },
    "similarity": {
        "class": SimilarityDetector,
        "modality": "any",
        "base": "similarity_detector"
    },
    "image_similarity": {
        "class": ImageSimilarityDetector,
        "modality": "image",
        "base": "similarity_detector"
    },
    "tabular_similarity": {
        "class": TabularSimilarityDetector,
        "modality": "tabular",
        "base": "similarity_detector"
    },
    "text_similarity": {
        "class": TextSimilarityDetector,
        "modality": "text",
        "base": "similarity_detector"
    }
}

def _validate_detector_class(cls):
    """
    Validate that the given class is a subclass of BaseDetector.

    Args:
        cls (type): The class to validate.
    
    Raises:
        ValueError: If the class does not meet the requirements.
    """
    if not (inspect.isclass(cls) and issubclass(cls, BaseDetector)):
        raise TypeError(f"Registered detector must be a subclass of BaseDetector. Got: {cls}")

def get_detector(name: str, config: dict = None, **kwargs):
    """
    Retrieve a detector class from the registry by its name.

    Args:
        name (str): The name of the detector.
        config (dict, optional): Configuration parameters for the detector.
        **kwargs: Additional keyword arguments for the detector initialization.

    Returns:
        Detector: An instance of the requested detector class.
    """
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Detector '{name}' is not registered.")
    
    detector_info = DETECTOR_REGISTRY[name]
    detector_class = detector_info["class"]
    _validate_detector_class(detector_class)

    return detector_class(config=config, **kwargs) if config else detector_class()

def list_detectors(modality: str = None, base: str = None):
    """
    List registered detectors filtered by modality or base class type.

    Args:
        modality (str, optional): One of "image", "text", "tabular", or "any".
        base (str, optional): Detector base type (e.g., "similarity_detector").

    Returns:
        dict: Matching detector entries with metadata.
    """
    return {
        name: info 
        for name, info in DETECTOR_REGISTRY.items()
        if (modality is None or info["modality"] == modality) 
        and (base is None or info["base"] == base)
    }

def load_from_config(config: dict):
    """
    Instantiate a detector from a config dictionary with a 'name' field.

    Args:
        config (dict): Configuration dictionary with a detector name.

    Returns:
        Detector: An instance of the requested detector class.
    """
    name = config.pop("name", None)
    if name is None:
        raise ValueError("Configuration must contain a 'name' field.")

    return get_detector(name, config=config)