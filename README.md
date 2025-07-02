# Monitoring Toolkit for ML Models

Monitoring Toolkit is a modular, extensible Python library for detecting suspicious or anomalous inputs in machine learning systems. It includes general-purpose detectors for repetitions, confidence-based anomalies, and similarity-based analysis across multiple modalities (text, image, tabular).

This library is designed to be easy to extend, model-agnostic, and suitable for real-time or batch pipelines.



## Features

-  Modular design via `BaseDetector` class
-  Plug-and-play detectors (similarity, repetition, confidence-based)
-  Support for image, text, and tabular inputs
-  Batch processing and internal state tracking
-  Easy integration with external systems


## Installation

```bash
git clone https://github.com/darusik/monitoring_toolkit.git
cd monitoring_toolkit
pip install -e .
```


## Available Detectors

The toolkit includes a range of built-in detectors, organized by type and modality.

| Detector Name        | Description                                                      | Modality  |
|----------------------|------------------------------------------------------------------|-----------|
| `confidence`         | Suspiciousness based on model confidence (entropy, margin, etc.) | Any |
| `repetition`         | Tracks repetition of identical queries via hashing               | Any |
| `image_repetition`   | Hash-based detection of repeated image inputs                    | Image |
| `tabular_repetition` | Hash-based detection of repeated tabular samples                 | Tabular |
| `text_repetition`    | Hash-based detection of repeated text queries                    | Text |
| `similarity`         | Generic embedding similarity-based detector                      | Any |
| `image_similarity`   | Detects visually similar images using CNN embeddings             | Image |
| `tabular_similarity` | Detects similar tabular samples via embedding space              | Tabular |
| `text_similarity`    | Detects similar queries via SentenceTransformer embeddings       | Text |

You can also retrieve the list through code:

```python
from detectors import list_detectors
print(list_detectors())  # or list_detectors(modality="text") to retrieve detectors feasible for text data
```

## Quick Start
The example below instantiates a similarity-based detector for image data and uses it to process a `Query`. The default `"image_similarity"` detector computes the cosine similarity between embedding of the current query and embeddings of past queries. The default embedding is computed as the output of the penultimate layer of ResNet18. If the similarity exceeds the `threshold`, the query is marked as suspicious. For each processed query, the detector returns its detection result (`result.is_suspicious`), which is `True` if the query is suspicious and `False` if not. Alongside, the detector also returns the confidence that the query is suspicious (`result.confidence`) and argumentation of the detection result (`result.reason`). Additionaly, classifiers can also return relevant metadata (`result.metadata`). 
```python
from detectors import get_detector
from utils.query import Query

detector = get_detector("image_similarity", config={"threshold": 0.95})

query = Query(input_data=your_image, modality="image")
result = detector.process(query)

print(result.is_suspicious, result.confidence, result.reason)
```
For a more detailed example see: ```examples/similarity_detection_demo.ipynb```


## Extending the Toolkit
This toolkit is designed to be flexible and easily extensible, making it simple to plug in your own detection logic or reuse components.


### Add a New Detector

1. Inherit from `BaseDetector` or one of the provided base classes to implement a custom logic:
   ```python
   from detectors.base_detector import BaseDetector

   class CustomDetector(BaseDetector):
       def _compute_score(self, query):
           ...
       def _make_prediction(self, score):
           ...
    ```
2. Register in the central registry to load the custom detector dynamically:
   ```python
   from detectors.registry import DETECTOR_REGISTRY

    DETECTOR_REGISTRY["custom_detector"] = {
        "class": CustomDetector,
        "modality": "text",
        "base": "custom"
    }
    ```

3. Instantiate the custom detector:
   ```python
   from detectors import get_detector

   detector = get_detector("custom_detector", config={...})
   ```

### Customization of Exsisting Detectors

Below we briefly summarize configurations and customizations supported by basic classes `ConfidenceDetector`, `RepetitionDetector`, and `SimilarityDetector`.


#### `ConfidenceDetector`

 - Supports three modes: *'entropy'*, *'max_confidence'*, and *'margin'*
 - Configurable parameters: *'threshold'* and *'max_history_size'* 

#### `RepetitionDetector`
- Supports custom hashing of queries `_hashed_query` that has to be implemented in all subclasses
- Has three subcalsses `ImageRepetitionDetector`, `TabularRepetitionDetector`, and `TextRepetitionDetector` with default hashing for corresponding modalities and support for pluggable custom hashing functions
- Configurable parameters: *'threshold'* and *'max_history_size'* 

#### `SimilarityDetector`

- Supports custom embedding `_embed` and similarity `_similarity` logic
- Has three subcalsses `ImageSimilarityDetector`, `TabularSimilarityDetector`, and `TextSimilarityDetector` with default embedding for corresponding modalities and support for pluggable model-based embeddings
- Configurable parameters: *'threshold'* and *'max_history_size'* 


### Aknowledgement 

The development of this toolkit was sponsored by [Netidee](https://www.netidee.at/) within the project [Monitaur](https://www.netidee.at/monitaur).

