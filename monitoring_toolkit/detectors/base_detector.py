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
This module defines the BaseDetector class, which serves as an abstract base class for all detectors.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from monitoring_toolkit.detectors.result import DetectionResult
from monitoring_toolkit.utils.query import Query

import logging

logger = logging.getLogger(__name__)

class BaseDetector(ABC):
    """Base class for all detectors.
    Subclasses must implement the _compute_score and _make_prediction methods.
    """
    def __init__(self, config=None):
        self.config = config or {}
        log_level = self.config.get("log_level", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    def _update_state(self, query: Query):
        """
        Optional: Override if the detector maintains internal state
        (e.g., stores recent queries, tracks counters).
        """
        pass

    @abstractmethod
    def _compute_score(self, query: Query) -> float:
        """
        Must be overridden.

        Compute a suspiciousness score based on the current query.
        Returns:
            float - Higher means more suspicious.
        """
        raise NotImplementedError

    def _record_score(self, score: float):
        """
        Optional: Store or log the score for introspection, metrics, or dashboards.
        """
        pass

    @abstractmethod
    def _make_prediction(self, score: float) -> DetectionResult:
        """
        Must be overridden.
        
        Convert a suspiciousness score into a final prediction result.
        Returns:
            DetectionResult:
                is_suspicious (bool)
                confidence (float)
                reason (str)
                metadata (optional dict)
        """
        raise NotImplementedError

    def process(self, query: Query) -> DetectionResult:
        """
        Process the query and update the detector state.
        """
        try:
            logger.debug("Processing query of type: %s", type(query.input_data).__name__)

            self._update_state(query)
            score = self._compute_score(query)
            logger.debug("Computed score: %.4f", score)

            self._record_score(score)
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
                metadata={"error": str(e)}
            )

    def process_batch(self, queries: Iterable[Query]) -> Iterator[DetectionResult]:
        """
        Process a batch of queries and update the detector state.
        """
        for query in queries:
            yield self.process(query)


    def get_state(self) -> dict:
        """
        Optional: Get the current state of the detector.
        """
        return {}

    def reset_state(self):
        """
        Optional: Reset the detector state.
        """
        pass

    