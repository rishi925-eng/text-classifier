---
title: Civic Issues Classifier
emoji: 🏛️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Civic Issues Text Classifier API 🏛️

A FastAPI application that classifies civic issues into three categories using DistilBERT:
- **Streetlight** issues 🚦
- **Garbage** issues 🗑️  
- **Potholes** issues 🕳️

## 🚀 Live API

This Space provides a REST API for civic issue classification. The API is automatically deployed and accessible at:

```
https://YOUR_USERNAME-civic-issues-classifier.hf.space
```

## 📡 API Endpoints

- `GET /` - API information and welcome message
- `GET /health` - Health check and model status
- `POST /predict` - Classify a single text input
- `POST /predict/batch` - Classify multiple texts at once
- `GET /examples` - Get example predictions with sample texts
- `GET /docs` - Interactive Swagger API documentation

## 💻 Usage Examples

### Single Text Classification

```python
import requests

response = requests.post(
    "https://YOUR_USERNAME-civic-issues-classifier.hf.space/predict",
    json={"text": "The streetlight near my house is not working"}
)
print(response.json())
# Output: {"text": "The streetlight near my house is not working", "predicted_category": "streetlight", "confidence": 0.9234}
```

### Batch Classification

```python
import requests

texts = [
    "The streetlight is broken",
    "Garbage dumped on roadside", 
    "Big pothole causing accidents"
]

response = requests.post(
    "https://YOUR_USERNAME-civic-issues-classifier.hf.space/predict/batch",
    json=texts
)
print(response.json())
```

### cURL Examples

```bash
# Health check
curl https://YOUR_USERNAME-civic-issues-classifier.hf.space/health

# Single prediction
curl -X POST "https://YOUR_USERNAME-civic-issues-classifier.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The streetlight is broken"}'

# Get examples
curl https://YOUR_USERNAME-civic-issues-classifier.hf.space/examples
```

## 🌍 Multilingual Support

The model supports both English and Hindi text:

```python
# English
{"text": "The streetlight is not working", "predicted_category": "streetlight"}

# Hindi
{"text": "batti kharab hai", "predicted_category": "streetlight"}

# Mixed
{"text": "road me gadhha hai", "predicted_category": "potholes"}
```

## 🔧 Model Details

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Training Data**: Civic issues dataset with English/Hindi text
- **Categories**: 3 classes (streetlight, garbage, potholes)
- **Framework**: Transformers + PyTorch
- **API**: FastAPI with automatic documentation

## 📊 Response Format

```json
{
  "text": "input text here",
  "predicted_category": "streetlight|garbage|potholes", 
  "confidence": 0.8765
}
```

## 🚀 Quick Start

1. **Health Check**: `GET /health`
2. **Interactive Docs**: Visit `/docs` for Swagger UI
3. **Try Examples**: `GET /examples` for sample predictions
4. **Make Predictions**: `POST /predict` with your text

## 📖 Interactive Documentation

Visit the `/docs` endpoint for full interactive API documentation with example requests and responses.

---

**Built with FastAPI ⚡ | Deployed on Hugging Face Spaces 🤗**