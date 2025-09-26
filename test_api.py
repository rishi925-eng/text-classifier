#!/usr/bin/env python3
"""
Test script for the FastAPI application
"""

import requests
import json

def test_api(base_url):
    print(f"Testing API at: {base_url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test prediction endpoint
    try:
        test_data = {"text": "The streetlight is broken"}
        response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        print(f"✅ Prediction test: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")

if __name__ == "__main__":
    # Replace with your Railway URL when deployed
    test_url = "https://your-app-url.railway.app"
    test_api(test_url)