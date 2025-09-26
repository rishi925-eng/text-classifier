from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os
import uvicorn

app = FastAPI(title="Civic Issues Text Classifier", description="API for classifying civic issues into categories")

# Global variables for model and tokenizer
model = None
tokenizer = None
id2label = None
label2id = None

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: float

def train_model():
    """Train the model if it doesn't exist"""
    global model, tokenizer, id2label, label2id
    
    print("Training model...")
    
    # Load dataset
    df = pd.read_csv("data/textdata.csv")
    
    # Allowed labels
    allowed_labels = ["streetlight", "garbage", "potholes"]
    
    # Filter and prepare data
    df = df[df["label"].isin(allowed_labels)].reset_index(drop=True)
    
    # Create mappings
    label2id = {label: idx for idx, label in enumerate(allowed_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Convert labels to IDs
    df["label_id"] = df["label"].map(label2id)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))
    test_dataset = Dataset.from_pandas(test_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    
    # Training arguments - optimized for deployment
    training_args = TrainingArguments(
        output_dir="./model/results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,  # Reduced for faster training
        weight_decay=0.01,
        logging_dir="./model/logs",
        logging_steps=50,
        report_to=None,
        push_to_hub=False,
        dataloader_pin_memory=False,
        save_steps=500,
        eval_steps=500,
    )
    
    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    
    # Save model
    os.makedirs("model/saved_model", exist_ok=True)
    model.save_pretrained("model/saved_model")
    tokenizer.save_pretrained("model/saved_model")
    
    print("✅ Model training completed and saved!")

def load_model():
    """Load the trained model"""
    global model, tokenizer, id2label, label2id
    
    model_path = "model/saved_model"
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        id2label = model.config.id2label
        label2id = model.config.label2id
        print("✅ Model loaded successfully!")
    else:
        print("No existing model found. Training new model...")
        train_model()

def classify_issue(text: str):
    """Classify a single text input"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = probs.argmax().item()
    
    predicted_label = id2label[predicted_class_id]
    confidence = probs[0][predicted_class_id].item()
    
    return {
        "text": text,
        "predicted_category": predicted_label,
        "confidence": round(confidence, 4)
    }

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Civic Issues Text Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Classify a single text",
            "/predict/batch": "POST - Classify multiple texts",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "categories": ["streetlight", "garbage", "potholes"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: TextInput):
    """Predict category for a single text input"""
    try:
        result = classify_issue(input_data.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    """Predict categories for multiple text inputs"""
    try:
        results = []
        for text in texts:
            result = classify_issue(text)
            results.append(result)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/examples")
async def get_examples():
    """Get example texts for testing the API"""
    examples = [
        "The streetlight near my home is broken",
        "There is garbage dumped on the roadside", 
        "The road has a big pothole causing accidents",
        "road me gadhha hai",
        "streetlight khrab hai",
        "kachra faila hua hai"
    ]
    
    try:
        predictions = []
        for text in examples:
            result = classify_issue(text)
            predictions.append(result)
        return {"examples": predictions}
    except Exception as e:
        return {"examples": examples, "note": "Model not ready for predictions yet"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
