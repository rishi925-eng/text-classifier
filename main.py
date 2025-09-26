from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from typing import Dict, Any

app = FastAPI(
    title="Text Classification API",
    description="API for classifying civic issues into categories: streetlight, garbage, and potholes",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: float

@app.on_event("startup")
async def load_model():
    """Load the trained model and tokenizer on startup"""
    global model, tokenizer
    
    model_path = "model/saved_model"
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise Exception(f"Model not found at {model_path}. Please train the model first.")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ Model and tokenizer loaded successfully")
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def classify_issue(text: str) -> Dict[str, Any]:
    """Classify a text input into one of the civic issue categories"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = probs.argmax().item()
    
    predicted_label = model.config.id2label[predicted_class_id]
    confidence = probs[0][predicted_class_id].item()
    
    return {
        "text": text,
        "predicted_category": predicted_label,
        "confidence": round(confidence, 4)
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Text Classification API for Civic Issues",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/classify",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and tokenizer is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.post("/classify", response_model=PredictionResponse)
async def classify_text(input_data: TextInput):
    """
    Classify text into civic issue categories
    
    Categories:
    - streetlight: Issues related to street lighting
    - garbage: Issues related to waste management
    - potholes: Issues related to road conditions
    """
    try:
        result = classify_issue(input_data.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get available classification categories"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "categories": list(model.config.id2label.values()),
        "description": {
            "streetlight": "Issues related to street lighting problems",
            "garbage": "Issues related to waste management and cleanliness",
            "potholes": "Issues related to road conditions and potholes"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)