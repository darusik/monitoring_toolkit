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
This module defines a repetition detector that identifies repeated queries based on hash values of their input data.
"""

import imagehash
import logging
import torch
import numpy as np

from collections import Counter, deque
from hashlib import sha256
from PIL import Image
from typing import Callable, Any

from detectors.base_detector import BaseDetector
from detectors.result import DetectionResult
from utils.query import Query

logger = logging.getLogger(__name__)

class RepetitionDetector(BaseDetector):
    """ 
    A base class for detectors that identify repeated queries.
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.history = deque(maxlen=self.config.get('max_history_size', 100))
        self.threshold = self.config.get('threshold', 5)
        self.repetition_counts = Counter()

    def _hashed_query(self, query: Query) -> Query:
        """
        Updates the query with a hash based on its input data.
        Has to be overridden in subclasses depending on the query modality.
        """
        raise NotImplementedError("Subclasses must implement _hashed_query method.")

    def _update_state(self, query: Query):
        """
        Update the state of the detector with the given query hash.
        This method maintains a history of the last N queries and updates
        the repetition counts for each query.
        """

        # Remove the oldest query if the history is full    
        if len(self.history) == self.history.maxlen:
            oldest_hash = self.history.popleft()
            self.repetition_counts.subtract([oldest_hash])
            if self.repetition_counts[oldest_hash] <= 0:              
                del self.repetition_counts[oldest_hash]
            logger.info("Evicting query hash due to max history size: %s", oldest_hash[:8])

        # Add the new query to the history and update counts
        self.history.append(query.hash)
        self.repetition_counts.update([query.hash])

    def _compute_score(self, query: Query) -> int:
        """
        Score is based on the repetition count of the query.
        """
        return self.repetition_counts[query.hash]


    def _make_prediction(self, score: int) -> DetectionResult: 
        """
        Convert the score into a final prediction result.
        """
        is_suspicious = score >= self.threshold
        if is_suspicious:
             logger.info("Repetition threshold reached: %d >= %d", score, self.threshold)

        return DetectionResult(
            is_suspicious=is_suspicious,
            confidence=min(score / self.threshold, 1.0),
            reason=f"Query repeated {score} times."
        )

    def process(self, query: Query) -> DetectionResult:
        """
        Process the query by:
        - Computing a hash
        - Updating internal state
        - Scoring the query
        - Returning a DetectionResult
        """

        try:
            logger.debug("Processing query with input data type: %s", type(query.input_data).__name__)
            query = self._hashed_query(query)
            logger.debug("Query hash: %s", query.hash[:8])

            self._update_state(query)
            logger.debug("Added query to history, current history size: %d", len(self.history))

            score = self._compute_score(query)
            logger.debug("Computed score for query: %d", score)

            result = self._make_prediction(score)
            if result.is_suspicious:
                logger.warning("Suspicious query detected! Confidence in being suspicious: %.4f", result.confidence)
            else:
                logger.info("Query is not suspicious. Confidence in being suspicious: %.4f", result.confidence)
            return result
        except Exception as e:
            logger.error("Failed to process query: %s", str(e), exc_info=True)
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                reason=f"Exception during processing: {type(e).__name__}",
                metadata={"error": str(e), "stage": "process"}
            )


    def get_state(self):
        """
        Get the current state of the detector.
        This method returns the size of the history, the number of unique inputs,
        and the maximum repetition count.
        """
        return {
            "current_history_size": len(self.history),
            "unique_inputs": len(self.repetition_counts),
            "repetition_counts": dict(self.repetition_counts),
            "max_repetition": max(self.repetition_counts.values(), default=0)
        }

    def reset_state(self):
        """
        Reset the state of the detector.
        This method clears the history and resets the repetition counts.
        """
        self.history.clear()
        self.repetition_counts.clear()

class ImageRepetitionDetector(RepetitionDetector):
    """
    A detector for repeated image queries.
    """
    def _hashed_query(
            self, query: Query, hashing_func: Callable[[Any], str] = lambda x: str(imagehash.phash(x))) -> Query:
        """
        Hash the image data to ensure consistent representation.
        This method supports three types of input data: PIL Image, numpy array, and torch tensor.
        It converts the input data to a PIL Image and then computes the hash.
        """
        data = query.input_data

        if isinstance(data, Image.Image):
            pil_image = data
        
        elif isinstance(data, np.ndarray):
            pil_image = Image.fromarray(data.astype('uint8'))

        elif isinstance(data, torch.Tensor):
            if data.ndim == 3 and data.shape[0] in (1, 3):
                pil_image = Image.fromarray(data.permute(1, 2, 0).cpu().numpy())
            else:
                logger.error(f"Unsupported tensor shape for image hashing: {data.shape}.")
                raise ValueError(f"Unsupported tensor shape for image hashing: {data.shape}.")
        else:
            logger.error(f"Unsupported data type for image hashing: {type(data)}.")
            raise ValueError(f"Unsupported data type for image hashing: {type(data)}.")

        # Compute the hash of the image
        query.hash = hashing_func(pil_image)
        logger.debug("Computed hash for query: %s", query.hash[:8])
        return query


class TabularRepetitionDetector(RepetitionDetector):
    """
    A detector for repeated tabular data queries.
    """
    def _hashed_query(self, query: Query, hashing_func: Callable[[Any], str] = lambda x: sha256(x.encode('utf-8')).hexdigest()) -> Query:
        """
        Hash the tabular data to ensure consistent representation.
        This method assumes that the input data is either a dictionary or has a to_dict method that returns a dictionary.
        """
        data = query.input_data

        if isinstance(data, dict):
            items = sorted(data.items())

        elif hasattr(data, "to_dict") and callable(data.to_dict):
            items = sorted(data.to_dict().items())

        else:
            logger.error(f"Unsupported data type for tabular hashing: {type(data)}.")
            raise ValueError(f"Unsupported data type for tabular hashing: {type(data)}.")
        
        str_data = str(items)
        query.hash = hashing_func(str_data)
        logger.debug("Computed hash for query: %s", query.hash[:8])
        return query

class TextRepetitionDetector(RepetitionDetector):
    """
    A detector for repeated text queries.
    """
    def _hashed_query(self, query: Query, hashing_func: Callable[[Any], str] = lambda x: sha256(x.encode('utf-8')).hexdigest()) -> Query:
        """
        Hash the text data to ensure consistent representation.
        This method assumes that the input data is a string or a list of strings.
        """
        data = query.input_data

        if isinstance(data, list):
            text = ' '.join(data)

        elif isinstance(data, str):
            text = data

        else:
            logger.error(f"Unsupported data type for text hashing: {type(data)}.")
            raise ValueError(f"Unsupported data type for text hashing: {type(data)}.")
        
        text = text.strip().lower()
        query.hash = hashing_func(text)
        logger.debug("Computed hash for query: %s", query.hash[:8])
        return query