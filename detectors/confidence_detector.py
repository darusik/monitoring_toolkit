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
This module defines the ConfidenceDetector class, which computes suspiciousness scores based on model confidence scores.
"""

import logging
import numpy as np

from collections import deque
from typing import Iterable

from detectors.base_detector import BaseDetector
from detectors.result import DetectionResult
from utils.query import Query

logger = logging.getLogger(__name__)

class ConfidenceDetector(BaseDetector):
    """
    A detector that computes suspiciousness scores based on the confidence
    of the model's predictions. It can operate in different modes:
    - 'entropy': High entropy indicates suspiciousness.
    - 'max_confidence': Low maximum confidence indicates suspiciousness.
    - 'margin': Low margin between the top two probabilities indicates suspiciousness.
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.mode = self.config.get('mode', 'entropy')
        assert self.mode in ('entropy', 'max_confidence', 'margin'), f'Invalid mode: {self.mode}'
        self.threshold = self.config.get('threshold', 0.5)
        self.history = deque(maxlen=self.config.get('max_history_size', 100)) 

    def _compute_entropy(self, probabilities: Iterable[float]) -> float:
        """
        Compute the entropy of the given probabilities.
        """
        # Normalize the probabilities if they don't sum to 1
        probability_sum = np.sum(probabilities)
        if np.abs(probability_sum - 1) > 1e-3:
            logger.warning("Probabilities do not sum to 1 (sum=%.4f); normalizing.", probability_sum)
            probabilities = probabilities / probability_sum

        return -np.sum(probabilities * np.log(probabilities + 1e-10))

    def _compute_score(self, query: Query):
        """
        Compute a suspiciousness score based on the current query.
        This method assumes that the query has a model_output attribute
        containing the model's output probabilities (not logits!).
        """
        if query.model_output is None or len(query.model_output) == 0:
            logger.error("Query is missing model_output or it is empty.")
            raise ValueError("Query must include non-empty model_output for confidence-based detection.")

        probabilities = np.array(query.model_output)
        logger.debug("Processing query with model output: %s", probabilities)
        if self.mode == 'entropy':
            # high entropy is suspicious 
            suspicious_score = self._compute_entropy(probabilities)
        elif self.mode == 'max_confidence':
            # low max confidence is suspicious
            max_confidence = np.max(probabilities)
            suspicious_score = 1 - max_confidence
        elif self.mode == 'margin':
            # low margin between top two probabilities is suspicious
            sorted_probs = np.sort(probabilities)[::-1]
            margin = sorted_probs[0] - sorted_probs[1]
            suspicious_score = 1 - margin
        else:
            logger.error("Invalid mode for computing score: %s", self.mode)
            raise ValueError(f"Invalid mode: {self.mode}")
        logger.debug("Computed suspicious score: %.4f in mode: %s", suspicious_score, self.mode)
        return suspicious_score

    def _record_score(self, score: float):
        """
        Store or log the score for introspection, metrics, or dashboards.
        """
        self.history.append(score)

    def _make_prediction(self, score: float) -> DetectionResult:
        """
        Convert a suspiciousness score into a final prediction result.
        """
        is_suspicious = score >= self.threshold
        if is_suspicious:
            logger.info("Suspicious score detected: %.4f >= %.4f", score, self.threshold)

        return DetectionResult(
            is_suspicious=is_suspicious,
            confidence=score,
            reason=f"{self.mode} mode: suspicious score = {score:.4f}, threshold = {self.threshold}",
            metadata=self.get_state() if self.history else {}
        )

    def get_state(self):
        """
        Get the current state of the detector.
        This method returns the last scores and their average.
        """
        if self.history:
            return {
                "last_score": self.history[-1],
                "avg_score": sum(self.history) / len(self.history),
                "num_scores": len(self.history)
            }
        return {}