import gradio as gr
import requests
import json

# Your FastAPI backend URL (will be the same Hugging Face Space)
API_URL = "http://localhost:7860"

def predict_text(text):
    """Predict civic issue category for input text"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            return f"Category: {result['predicted_category']}\nConfidence: {result['confidence']:.4f}"
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_examples():
    """Get example predictions"""
    try:
        response = requests.get(f"{API_URL}/examples", timeout=30)
        if response.status_code == 200:
            examples = response.json()["examples"]
            return json.dumps(examples, indent=2)
        else:
            return "Error loading examples"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Civic Issues Classifier") as demo:
    gr.Markdown("# 🏛️ Civic Issues Text Classifier")
    gr.Markdown("Classify civic complaints into: **Streetlight**, **Garbage**, or **Potholes**")
    
    with gr.Tab("Single Prediction"):
        text_input = gr.Textbox(
            label="Enter complaint text",
            placeholder="e.g., The streetlight near my house is not working",
            lines=3
        )
        predict_btn = gr.Button("Classify Issue", variant="primary")
        result_output = gr.Textbox(label="Prediction Result", lines=3)
        
        predict_btn.click(predict_text, inputs=text_input, outputs=result_output)
        
        # Example inputs
        gr.Examples(
            examples=[
                ["The streetlight is broken and needs repair"],
                ["There is garbage dumped on the roadside"],
                ["Big pothole causing traffic problems"],
                ["road me gadhha hai"],
                ["streetlight khrab hai"],
                ["kachra faila hua hai"]
            ],
            inputs=text_input
        )
    
    with gr.Tab("API Examples"):
        examples_btn = gr.Button("Load Examples", variant="secondary")
        examples_output = gr.Code(label="Example Predictions", language="json")
        
        examples_btn.click(get_examples, outputs=examples_output)
    
    with gr.Tab("API Documentation"):
        gr.Markdown("""
        ## FastAPI Endpoints
        
        - `GET /` - API information
        - `GET /health` - Health check  
        - `POST /predict` - Single text prediction
        - `POST /predict/batch` - Batch predictions
        - `GET /examples` - Example predictions
        - `GET /docs` - Interactive API docs
        
        ## Usage Examples
        
        ### Python
        ```python
        import requests
        
        # Single prediction
        response = requests.post(
            "https://your-space.hf.space/predict",
            json={"text": "The streetlight is broken"}
        )
        print(response.json())
        ```
        
        ### cURL
        ```bash
        curl -X POST "https://your-space.hf.space/predict" \\
          -H "Content-Type: application/json" \\
          -d '{"text": "Garbage on the road"}'
        ```
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)