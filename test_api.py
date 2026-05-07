"""
Quick test script for the Deepfake Detection API
"""
import requests
import time

API_URL = "http://localhost:5000"

# Wait for server to be ready
print("Waiting for server...")
time.sleep(3)

# Test health endpoint
print("\n1. Testing health endpoint...")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"✓ Server is healthy: {response.json()}")
except Exception as e:
    print(f"✗ Health check failed: {e}")
    exit(1)

# Test model info
print("\n2. Getting model info...")
try:
    response = requests.get(f"{API_URL}/model-info")
    info = response.json()
    print(f"✓ Model info:")
    print(f"  - Type: {info.get('model_type', 'unknown')}")
    print(f"  - Input shape: {info.get('input_shape', 'unknown')}")
except Exception as e:
    print(f"✗ Model info failed: {e}")

# Test detection with a real image
print("\n3. Testing detection with a REAL image...")
test_image_path = r"image-detection-module\training_data\Test\Real\real_1091.jpg"
try:
    with open(test_image_path, 'rb') as f:
        files = {'image': ('test_real.jpg', f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/detect", files=files)
        result = response.json()
        
    if response.status_code == 200:
        print(f"✓ Detection successful!")
        print(f"  - Full response: {result}")
        detection = result.get('detection', {})
        print(f"  - Prediction: {detection.get('prediction', 'unknown')}")
        print(f"  - Label: {detection.get('label', 'unknown')}")
        print(f"  - Confidence: {detection.get('confidence', 0):.2%}")
        print(f"  - Probabilities: {detection.get('probabilities', [])}")
    else:
        print(f"✗ Detection failed: {result.get('error', 'unknown')}")
except Exception as e:
    print(f"✗ Detection test failed: {e}")

# Test detection with a fake image
print("\n4. Testing detection with a FAKE image...")
test_image_path = r"image-detection-module\training_data\Test\Fake\fake_1091.jpg"
try:
    with open(test_image_path, 'rb') as f:
        files = {'image': ('test_fake.jpg', f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/detect", files=files)
        result = response.json()
        
    if response.status_code == 200:
        print(f"✓ Detection successful!")
        print(f"  - Full response: {result}")
        detection = result.get('detection', {})
        print(f"  - Prediction: {detection.get('prediction', 'unknown')}")
        print(f"  - Label: {detection.get('label', 'unknown')}")
        print(f"  - Confidence: {detection.get('confidence', 0):.2%}")
        print(f"  - Probabilities: {detection.get('probabilities', [])}")
    else:
        print(f"✗ Detection failed: {result.get('error', 'unknown')}")
except Exception as e:
    print(f"✗ Detection test failed: {e}")

print("\n" + "="*80)
print("API test complete!")
print("="*80)
