# import torch
# import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

# from torchvision.io import read_image
from torchvision import transforms
from torchvision.datasets.utils import download_url

from monitoring_toolkit.detectors import get_detector
from monitoring_toolkit.utils.query import Query

# Get a test image
ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "dog.jpg"
DOG_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"

if not IMAGE_PATH.exists():
    print(f"Downloading test image to {IMAGE_PATH}")
    download_url(DOG_URL, str(ROOT), filename="dog.jpg")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(IMAGE_PATH).convert("RGB")
original_tensor = transform(image)

# Create a slightly modified version (blurred for demo purposes)
modified_image = image.filter(ImageFilter.GaussianBlur(radius=1))
modified_tensor = transform(modified_image)

# Prepare queries
original_query = Query(input_data=original_tensor)
modified_query = Query(input_data=modified_tensor)

# Initialize similarity detector
detector = get_detector("image_similarity", config={
    "threshold": 0.9,
    "max_history_size": 10
})

# Process queries with the detector
print("Processing original image...")
result1 = detector.process(original_query)
print(f"Result: \n query is suspicious: {result1.is_suspicious}, \n confidence that query is suspicious: {result1.confidence}, \n reason: {result1.reason}")

print("\nProcessing slightly modified image...")
result2 = detector.process(modified_query)
print(f"Result: \n query is suspicious: {result2.is_suspicious}, \n confidence that query is suspicious: {result2.confidence}, \n reason: {result2.reason}")
