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
This module defines a similarity-based detector that identifies suspicious queries based on similarity of their embeddings.
"""

import logging
from collections import deque

from monitoring_toolkit.detectors.base_detector import BaseDetector
from monitoring_toolkit.detectors.result import DetectionResult
from monitoring_toolkit.utils.query import Query

import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np


logger = logging.getLogger(__name__)

class SimilarityDetector(BaseDetector):
    """
    A base class for similarity-based detectors.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.max_history_size = self.config.get('max_history_size', 100)
        self.threshold = self.config.get('threshold', 0.9)
        self.embedding_history = deque(maxlen=self.max_history_size)
        self.latest_incident = None
        self._latest_similarity_info = None

    def _embed(self, query: Query):
        raise NotImplementedError("No default embed function provided.")

    def _similarity(self, embedding1, embedding2):
        """
        Default similarity function: cosine similarity.
        """
        embedding1, embedding2 = np.asarray(embedding1, dtype=np.float64).flatten(), np.asarray(embedding2, dtype=np.float64).flatten()
        norm1, norm2 = np.linalg.norm(embedding1), np.linalg.norm(embedding2)
        return np.dot(embedding1, embedding2) / (norm1 * norm2) if norm1 and norm2 else 0.0

    def _record_incident(self, query, embedding):
        """
        Record the latest suspicious query match.
        """
        self.latest_incident = {
            'query': query,
            'embedding': embedding,
            'similarity_info': self._latest_similarity_info
        }

    def _update_state(self, query: Query):
        """
        Add query embedding to the history. 
        """
        self.embedding_history.append(query.embedding)

    def _compute_score(self, query: Query) -> float:
        """
        Compute a suspiciousness score based on the current query embedding.
        This method checks the similarity of the current embedding with the history.
        """
        if not self.embedding_history:
            logger.info('No previous embeddings to compare with. Returning score of 0.0.')
            return 0.0

        similarity_scores = [self._similarity(query.embedding, past_embedding) for past_embedding in self.embedding_history]
        max_score = max(similarity_scores)
        most_similar_index = similarity_scores.index(max_score)
        most_similar_embedding = self.embedding_history[most_similar_index]

        logger.debug("Most similar embedding found at index %d with score %.4f", most_similar_index, max_score)

        self._latest_similarity_info = {
            'most_similar_index': most_similar_index,
            'most_similar_embedding': most_similar_embedding,
            'similarity_score': max_score
        }
        return max_score

    def _make_prediction(self, score: float) -> DetectionResult:
        """
        Convert a suspiciousness score into a final prediction result.
        """
        is_suspicious = score >= self.threshold
        metadata = self._latest_similarity_info or {}

        return DetectionResult(
            is_suspicious=is_suspicious,
            confidence=score,
            reason=f"Similarity score: {score:.4f}, threshold: {self.threshold:.4f}.",
            metadata=metadata
        )

    def process(self, query: Query) -> DetectionResult:
        """
        Process the query by:
        - Computing the embedding
        - Finding the most similar embedding
        - Updating internal state
        - Returning a DetectionResult
        """
        try:
            logger.debug("Processing query with input data type: %s", type(query.input_data).__name__)
            embedding = self._embed(query)
            logger.debug("Computed embedding for the current query with shape: %s", np.asarray(embedding).shape if isinstance(embedding, np.ndarray) else type(embedding))
            query.embedding = embedding

            score = self._compute_score(query)
            logger.debug("Computed similarity score: %.4f", score)

            self._update_state(query)
            logger.debug("Updated state with new embedding. History size: %d", len(self.embedding_history))
            result = self._make_prediction(score)
            if result.is_suspicious:
                self._record_incident(query, embedding)
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

    def get_state(self, include_embedding=False):
        """
        Get the current state of the detector.
        This method returns the size of the history,
        and the latest incident if any.
        """
        return {
            "current_history_size": len(self.embedding_history),
            "threshold": self.threshold,
            "latest_incident_similarity_score": self.latest_incident['similarity_info']['similarity_score'] if self.latest_incident else None,
            "latest_incident_closest_index": self.latest_incident['similarity_info']['most_similar_index'] if self.latest_incident else None,
            "latest_incident_closest_embedding": self.latest_incident['similarity_info']['most_similar_embedding'] if self.latest_incident and include_embedding else None,
        }

    def reset_state(self):
        """
        Reset the state of the detector.
        This method clears the history and resets the latest incident.
        """
        self.embedding_history.clear()
        self.latest_incident = None


class ImageSimilarityDetector(SimilarityDetector):
    """
    A detector for image queries based on similarity.
    """
    def __init__(self, config=None, model=None, transform=None, model_layer=-2):
        """
        Initialize the ImageSimilarityDetector with optional:
        - custom embedding function
        - custom similarity function
        - PyTorch model for feature extraction
        - image transform (e.g., normalization)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model:
            self.model = self._truncate_model(model, model_layer)
        else:
            self.model = self._default_model(layer_index=model_layer)
        self.model.eval().to(self.device)

        self.transform = transform or T.Compose([
                                                    T.Resize((224, 224)),  # Resize to match ResNet input size
                                                    T.ToTensor(),           # Convert PIL image to tensor
                                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
                                                ])
        super().__init__(config=config)

    def _truncate_model(self, model, layer_index):
        return torch.nn.Sequential(*list(model.children())[:layer_index])

    def _default_model(self, layer_index=-2):
        """
        Load a default model (ResNet-18) and truncate at the specified layer.
        """
        model = resnet18(weights='DEFAULT')
        return self._truncate_model(model, layer_index).eval().to(self.device)

    def _embed(self, query: Query):
        """
        Extracts an embedding from a query using a CNN model.
        Supports various image input types (tensor, array, PIL).
        If the query already has an embedding, it returns that directly.
        """
        data = query.input_data
            
        if isinstance(data, torch.Tensor):
            if data.ndim == 3 and data.shape[0] in (1, 3): 
                image = data.unsqueeze(0).to(self.device)
            else:
                raise ValueError(f"Unexpected tensor shape for image input: {data.shape}")
            
        elif isinstance(data, np.ndarray):
            if data.ndim == 3:
                if data.shape[2] == 3:  # likely HWC (common)
                    pil_image = Image.fromarray(data.astype('uint8'), mode='RGB')
                elif data.shape[0] == 3:  # possibly CHW, channel-first
                    pil_image = Image.fromarray(np.transpose(data, (1, 2, 0)).astype('uint8'), mode='RGB')
                else:
                    raise ValueError(f"Unsupported channel dimension in array: shape={data.shape}")
            elif data.ndim == 2:
                # Grayscale image
                pil_image = Image.fromarray(data.astype('uint8'), mode='L')
            else:
                raise ValueError(f"Unsupported ndarray shape for image: {data.shape}")
            image = self.transform(pil_image).unsqueeze(0).to(self.device)

        elif isinstance(data, Image.Image):
            image = self.transform(data).unsqueeze(0).to(self.device)
        else:
            logger.error("Unsupported input data type in _embed: %s", type(data))
            raise ValueError(f"Unsupported input data type: {type(data)}. Expected torch tensor, numpy ndarray, or PIL image.")
        
        with torch.no_grad():
            embedding = self.model(image).squeeze().cpu().numpy()
        return embedding

       
class TabularSimilarityDetector(SimilarityDetector):
    def __init__(self, config=None, encoder=None):
        """
        Initialize the TabularSimilarityDetector.

        Parameters:
        - config (dict, optional): Configuration dictionary for the detector.
        - embed (callable, optional): Custom embedding function. Defaults to `_embed`.
        - similarity (callable, optional): Custom similarity function. Defaults to `_similarity`.
        - encoder (object, optional): Optional encoder for transforming tabular data into embeddings.
          This can be a learned model (e.g., PCA, autoencoder) or a feature transformer.
        """
        self.encoder = encoder  # Optional: can be a learned model or PCA
        super().__init__(config=config)

    def _embed(self, query: Query):

        data = query.input_data

        if isinstance(data, dict):
            features = np.array([v for _, v in sorted(data.items())])
        elif hasattr(data, 'to_dict') and callable(data.to_dict):
            features = np.array([v for _, v in sorted(data.to_dict().items())])
        else:
            logger.error("Unsupported input data type in _embed: %s", type(data))
            raise ValueError(f"Unsupported data type for computing embedding: {type(data)}.")

        if self.encoder:
            return np.asarray(self.encoder.transform([features])[0])
        return features
    

class TextSimilarityDetector(SimilarityDetector):
    def __init__(self, config=None, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the TextSimilarityDetector.
        Parameters:
        - config (dict, optional): Configuration dictionary for the detector.
        - embed (callable, optional): Custom embedding function. Defaults to `_embed`.
        - similarity (callable, optional): Custom similarity function. Defaults to `_cosine_similarity`.
        - model_name (str, optional): Name of the pre-trained SentenceTransformer model to use.
          Defaults to 'all-MiniLM-L6-v2'.
        """
        self.model = SentenceTransformer(model_name)
        super().__init__(config=config)

    def _embed(self, query: Query):

        data = query.input_data

        if isinstance(data, list):
            text = ' '.join(data)

        elif isinstance(data, str):
            text = data

        else:
            logger.error("Unsupported input data type in _embed: %s", type(data))
            raise ValueError(f"Unsupported data type for computing embedding: {type(data)}.")
        
        return np.asarray(self.model.encode(text))
