import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os

def train_model():
    """Train the text classification model and save it"""
    
    print("🚀 Starting model training...")
    
    # Load dataset
    df = pd.read_csv("data/textdata.csv")
    
    # Allowed labels (text form)
    allowed_labels = ["streetlight", "garbage", "potholes"]
    
    # Filter only rows with allowed labels and keep text labels in df
    df = df[df["label"].isin(allowed_labels)].reset_index(drop=True)
    
    # Create mapping for training (used internally)
    label2id = {label: idx for idx, label in enumerate(allowed_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Convert label column to integer IDs for training
    df["label_id"] = df["label"].map(label2id)
    
    print("Label mapping:", label2id)
    print("Loaded dataset shape:", df.shape)
    
    # Split the data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]])
    
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(allowed_labels),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Rename label_id to labels
    train_dataset = train_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")
    
    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    
    # Train the model
    print("🏋️ Training model...")
    trainer.train()
    
    # Create model directory if it doesn't exist
    model_dir = "model/saved_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    print(f"✅ Model saved in {model_dir}")
    
    # Test the model with example predictions
    def classify_issue(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = probs.argmax().item()
        predicted_label = model.config.id2label[predicted_class_id]
        return {"text": text, "predicted_category": predicted_label, "confidence": probs[0][predicted_class_id].item()}
    
    # Test examples
    examples = [
        "The streetlight near my home is broken",
        "There is garbage dumped on the roadside",
        "The road has a big pothole causing accidents",
        "road me gadhha hai",
        "streetlight khrab hai",
        "kachra faila hua hai"
    ]
    
    print("\n🧪 Testing trained model:")
    for text in examples:
        result = classify_issue(text)
        print(f"Text: '{text}' -> Category: {result['predicted_category']} (Confidence: {result['confidence']:.4f})")

if __name__ == "__main__":
    train_model()