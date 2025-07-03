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
This module defines the Query class, which represents a single query to the model.
"""

from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Query:
    """
    Represents a single query to the model. 

    Attributes:
        input_data (any): The input data for the query, can be an image, text, or tabular data.
        model_output (any, optional): The output from the model, such as logits or probabilities.
        hash (str, optional): A unique hash representing the query, computed from the input data by the detector.
        embedding (any, optional): An embedding representation of the query, computed from the input data by the detector.
        metadata (dict, optional): Additional metadata about the query, such as user ID, session ID, etc.
        timestamp (datetime, optional): The time when the query was created. Defaults to the current time.
    """
    input_data: any
    model_output: any = None
    hash: str = field(init=False, default=None)
    embedding: any = field(init=False, default=None)
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        assert self.input_data is not None, "input_data must not be None"