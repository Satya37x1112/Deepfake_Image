"""
Flask API for Deepfake Detection
Provides REST API endpoints for image upload and detection.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import models in order of preference
USE_MODEL_TYPE = None
# Prefer transfer learning model first, then regular CNN
try:
    from model.fast_train import FastDeepfakeDetector
    from model.model_cnn import DeepfakeCNNDetector
    USE_MODEL_TYPE = 'tensorflow_cnn'
except ImportError:
    try:
        from model.model_cnn import DeepfakeCNNDetector
        USE_MODEL_TYPE = 'tensorflow_cnn'
    except ImportError:
        try:
            from model.model_simple_cnn import SimpleCNNDeepfakeDetector
            USE_MODEL_TYPE = 'simple_cnn'
        except ImportError:
            try:
                from model.model import DeepfakeDetector
                USE_MODEL_TYPE = 'legacy'
            except ImportError:
                print("ERROR: No model classes could be imported!")
                USE_MODEL_TYPE = None

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Model paths (try in order of preference)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Model path candidates
TF_CNN_MODEL = os.path.join(BASE_DIR, 'saved-models', 'deepfake_cnn_model.h5')
IMPROVED_MODEL_PATH = os.path.join(BASE_DIR, 'saved-models', 'deepfake_detector_improved.pkl')
CNN_MODEL_PATH = os.path.join(BASE_DIR, 'saved-models', 'deepfake_detector_cnn.pkl')
OLD_MODEL_PATH = os.path.join(BASE_DIR, 'saved-models', 'deepfake_detector.pkl')

# Determine which model to use
# Priority:
# 1) TensorFlow/Keras CNN model (.h5)
# 2) Pickle-based RandomForest models (.pkl)
MODEL_PATH = None

if os.path.exists(TF_CNN_MODEL):
    MODEL_PATH = TF_CNN_MODEL
    USE_MODEL_TYPE = 'tensorflow_cnn'
elif os.path.exists(IMPROVED_MODEL_PATH):
    MODEL_PATH = IMPROVED_MODEL_PATH
elif os.path.exists(CNN_MODEL_PATH):
    MODEL_PATH = CNN_MODEL_PATH
else:
    MODEL_PATH = OLD_MODEL_PATH

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model at startup
print("=" * 80)
print("DEEPFAKE DETECTION API - INITIALIZING")
print("=" * 80)
print(f"Script location: {__file__}")
print(f"Model type: {USE_MODEL_TYPE}")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")

try:
    print("Loading model... (this may take 10-20 seconds)")
    if USE_MODEL_TYPE == 'tensorflow_cnn':
        # TensorFlow/Keras CNN model
        # Check which model exists
        fast_model_path = os.path.join(BASE_DIR, 'saved-models', 'deepfake_fast.h5')
        print(f"Checking for fast model at: {fast_model_path}")
        print(f"Fast model exists: {os.path.exists(fast_model_path)}")
        if os.path.exists(fast_model_path):
            # Use FastDeepfakeDetector for transfer learning model (96x96)
            print("Using TensorFlow/Keras Transfer Learning model (MobileNetV2)")
            print(f"Loading model from: {fast_model_path}")
            detector = FastDeepfakeDetector(input_shape=(96, 96, 3))
            detector.load_model_file(fast_model_path)
            MODEL_PATH = fast_model_path  # Update MODEL_PATH for consistency
            print(f"✓ Fast transfer learning model loaded (input shape: {detector.input_shape})")
        else:
            # Use regular CNN detector (96x96 by default)
            print("Using TensorFlow/Keras Custom CNN model")
            print(f"Loading model from: {MODEL_PATH}")
            detector = DeepfakeCNNDetector(input_shape=(96, 96, 3))
            detector.load_model_file(MODEL_PATH)
            print(f"✓ TensorFlow model loaded (input shape: {detector.input_shape})")
    elif USE_MODEL_TYPE == 'simple_cnn':
        # RandomForest feature-based model
        detector = SimpleCNNDeepfakeDetector()
        print("Using RandomForest model (Enhanced deep features)")
        detector.load(MODEL_PATH)
    else:
        # Legacy model
        detector = DeepfakeDetector()
        print("Using legacy model (hand-crafted features + Random Forest)")
        detector.load(MODEL_PATH)
    MODEL_LOADED = True
    print("✓ Model loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"✗ Failed to load model: {e}")
    print(f"Traceback: {e.__class__.__name__}")
    print("  API will return error responses until model is available.")

print("=" * 80)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """Serve the frontend"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, images)"""
    return send_from_directory(app.static_folder, path)


@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Deepfake Detection API',
        'version': '1.0.0',
        'description': 'Random Forest-based deepfake image detection service',
        'model_loaded': MODEL_LOADED,
        'endpoints': {
            'POST /detect': 'Upload image for deepfake detection',
            'GET /health': 'Check API health status',
            'GET /stats': 'Get detection statistics',
            'GET /model-info': 'Get model information'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'degraded',
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded',
            'model_loaded': False
        }), 503
    
    try:
        # Provide model-specific info depending on loaded detector
        info = {'model_loaded': True, 'timestamp': datetime.now().isoformat()}

        if USE_MODEL_TYPE == 'tensorflow_cnn':
            # TensorFlow/Keras CNN model
            info.update({
                'model_type': 'TensorFlow/Keras CNN',
                'input_shape': detector.input_shape,
                'num_classes': detector.num_classes,
                'is_trained': detector.is_trained,
                'label_map': detector.label_map
            })
        else:
            # RandomForest-based models
            try:
                info.update({
                    'model_type': 'Random Forest Classifier',
                    'n_estimators': getattr(detector.classifier, 'n_estimators', None),
                    'max_depth': getattr(detector.classifier, 'max_depth', None),
                    'using_pca': getattr(detector, 'use_pca', False),
                    'is_trained': getattr(detector, 'is_trained', True),
                    'label_map': getattr(detector, 'label_map', {})
                })

                if getattr(detector, 'use_pca', False) and getattr(detector, 'pca', None):
                    info['pca_components'] = detector.pca.n_components
                    info['pca_explained_variance'] = float(
                        detector.pca.explained_variance_ratio_.sum()
                    )
            except Exception:
                info['model_type'] = 'Unknown'

        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get model info: {str(e)}'
        }), 500


@app.route('/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint.
    Upload an image and get deepfake detection results.
    """
    # Check if model is loaded
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded. Please ensure the model file exists.',
            'model_path': MODEL_PATH,
            'suggestion': 'Train a model using train.py first'
        }), 503
    
    # Check if file is in request
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided',
            'expected_field': 'image'
        }), 400
    
    file = request.files['image']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        }), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'allowed_types': list(ALLOWED_EXTENSIONS),
            'received': file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'unknown'
        }), 400
    
    try:
        # Secure filename and save
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Perform detection
        result = detector.predict(filepath)
        
        # Add metadata
        result['filename'] = filename
        result['upload_id'] = unique_filename.split('_')[0]
        result['timestamp'] = datetime.now().isoformat()
        result['file_size_kb'] = round(os.path.getsize(filepath) / 1024, 2)
        
        # Clean up uploaded file (optional - comment out to keep uploads)
        try:
            os.remove(filepath)
        except:
            pass
        
        # Return results
        return jsonify({
            'success': True,
            'detection': result,
            'message': f'Image classified as {result["prediction"]} with {result["confidence"]:.2f}% confidence'
        }), 200
        
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'success': False
        }), 500


@app.route('/stats', methods=['GET'])
def stats():
    """
    Get detection statistics from upload folder.
    (This is a simple implementation - in production, use a database)
    """
    try:
        upload_count = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                           if allowed_file(f)])
        
        return jsonify({
            'total_uploads': upload_count,
            'model_loaded': MODEL_LOADED,
            'uptime_info': 'Statistics tracking is basic in this version',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get stats: {str(e)}'
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'max_size_mb': MAX_FILE_SIZE / (1024 * 1024),
        'message': f'Maximum file size is {MAX_FILE_SIZE / (1024 * 1024)}MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': {
            'GET /': 'API information',
            'POST /detect': 'Upload image for detection',
            'GET /health': 'Health check',
            'GET /model-info': 'Model information',
            'GET /stats': 'Detection statistics'
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    try:
        print("\n" + "=" * 80)
        print("STARTING DEEPFAKE DETECTION WEB APPLICATION")
        print("=" * 80)
        print("Frontend: http://localhost:5000")
        print("\nAPI Endpoints:")
        print("  POST /detect      - Upload image for detection")
        print("  GET  /health      - Health check")
        print("  GET  /model-info  - Model information")
        print("  GET  /stats       - Statistics")
        print("  GET  /api         - API information")
        print("\nModel Status:", "✓ LOADED" if MODEL_LOADED else "✗ NOT LOADED")
        print("=" * 80 + "\n")
        
        # Use waitress for more stable server
        from waitress import serve
        import os
        port = int(os.environ.get('PORT', 5000))
        print(f"✓ Server starting with Waitress on port {port}...")
        serve(app, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"\n✗ ERROR: Server failed to start!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
