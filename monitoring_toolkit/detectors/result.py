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
This module defines the DetectionResult class, which encapsulates the result of a detection operation.
"""

from dataclasses import asdict, dataclass, field

@dataclass
class DetectionResult:
    is_suspicious: bool
    confidence: float
    reason: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Post-initialization processing to ensure that the attributes are of the correct type.
        """
        self.is_suspicious = bool(self.is_suspicious)
        self.confidence = float(self.confidence)
        self.reason = str(self.reason)
        self.metadata = dict(self.metadata)

    def dict(self) -> dict:
        """
        Convert the DetectionResult instance to a dictionary.
        """
        return asdict(self)