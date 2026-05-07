# Deepfake Image Detection - TensorFlow CNN Model# Deepfake Image Detector - Complete Guide



Fresh, clean implementation compatible with Python 3.13 using TensorFlow/Keras CNN for deepfake detection.## 🎯 Overview

This is a deepfake detection system with automatic handling of mislabeled training data. The system includes:

## 🎯 Features- **Dataset utilities** to detect and fix mislabeled folders

- **Training script** with automatic label correction

- **TensorFlow/Keras CNN model** with proper architecture (Conv2D, BatchNorm, Dropout)- **Flask web server** with modern UI for image analysis

- **Input size:** 150x150x3 RGB images

- **Data augmentation** for better generalization## 🚨 Important: Mislabeled Dataset Issue

- **Flask REST API** for easy integration

- **Automatic label mapping** detection for swapped datasets**Problem:** The training dataset folders are swapped:

- **Progress tracking** and checkpointing during training- `training_data/Train/Real/` folder contains **FAKE** images

- **Early stopping** and learning rate scheduling- `training_data/Train/Fake/` folder contains **REAL** images



## 📋 Requirements**Solutions:** This project handles this automatically in 3 ways:



- Python 3.13 (or 3.10+)### Method 1: Dynamic Correction (Recommended)

- TensorFlow 2.15+Labels are corrected during training without moving files.

- See `requirements.txt` for full dependencies```bash

python train.py --fix-method dynamic

## 🚀 Quick Start```



### 1. Install Dependencies### Method 2: Swap Folders

Physically swap folder names (creates backup).

```powershell```bash

pip install -r requirements.txtpython image-detection-module/utils/dataset_utils.py --action swap --dataset training_data/Train

``````



### 2. Prepare Dataset### Method 3: Auto (Default)

Automatically detects and applies the best method.

Organize your training data:```bash

```python train.py --fix-method auto

training_data/```

├── Train/

│   ├── Real/      # Real images## 📋 Requirements

│   └── Fake/      # Fake/deepfake images

└── Test/Install dependencies:

    ├── Real/```bash

    └── Fake/pip install flask waitress opencv-python numpy scikit-learn tqdm

``````



### 3. Train the Model## 🚀 Quick Start



**Quick test run (small subset):**### Step 1: Check Dataset Status

```powershell```bash

python image-detection-module/model/train.py --epochs=10 --batch_size=32python image-detection-module/utils/dataset_utils.py --action stats --dataset training_data/Train

``````



**Full training (140k images):**### Step 2: Mark Dataset as Swapped (if needed)

```powershell```bash

python image-detection-module/model/train.py --epochs=50 --batch_size=32 --validation_split=0.2python image-detection-module/utils/dataset_utils.py --action mark_swapped --dataset training_data/Train

``````



**Training options:**### Step 3: Train Model

- `--train_dir`: Path to training directory (default: `../training_data/Train`)```bash

- `--epochs`: Number of epochs (default: 50)# Using all 140,000 images with dynamic label correction

- `--batch_size`: Batch size (default: 32)python train.py --fix-method dynamic --use-checkpoint

- `--validation_split`: Validation fraction (default: 0.2)

- `--save_path`: Model save path (default: `../../saved-models/deepfake_cnn_model.h5`)# Or train with subset for testing

- `--input_size`: Input image size (default: 150)python train.py --fix-method dynamic --max-samples 5000 --use-checkpoint

```

**Training output:**

- Model will be saved to `saved-models/deepfake_cnn_model.h5`### Step 4: Start Web Server

- Best model (highest validation accuracy) is automatically saved```bash

- Early stopping if validation loss doesn't improve for 10 epochspython app.py --model saved-models/deepfake_detector_improved.pkl

- Learning rate reduces on plateau```



### 4. Run InferenceThen open: **http://localhost:5000**



**Python script:**## 🛠️ Dataset Utilities

```python

from image_detection_module.model.model_cnn import DeepfakeCNNDetector### Check dataset statistics:

```bash

# Load trained modelpython image-detection-module/utils/dataset_utils.py --action stats

detector = DeepfakeCNNDetector()```

detector.load_model_file('saved-models/deepfake_cnn_model.h5')

### Verify dataset integrity:

# Predict single image```bash

result = detector.predict('path/to/test_image.jpg')python image-detection-module/utils/dataset_utils.py --action verify

print(result)```

# Output: {

#   'prediction': 'real',  # or 'fake'### Swap folder names (with backup):

#   'label': 0,```bash

#   'confidence': 87.5,python image-detection-module/utils/dataset_utils.py --action swap

#   'probabilities': {'real': 87.5, 'fake': 12.5}```

# }

### Swap without backup (use with caution):

# Batch prediction```bash

images = ['image1.jpg', 'image2.jpg', 'image3.jpg']python image-detection-module/utils/dataset_utils.py --action swap --no-backup

results = detector.batch_predict(images)```

```

## 🎓 Training Options

### 5. Start Flask API Server

### Basic training:

```powershell```bash

python image-detection-module/api/app.pypython train.py

``````



Server will start at `http://localhost:5000`### Training with custom parameters:

```bash

**API Endpoints:**python train.py \

  --dataset training_data/Train \

- `GET /health` - Health check  --fix-method dynamic \

- `GET /model-info` - Model information  --max-samples 50000 \

- `GET /api` - API information  --n-estimators 500 \

- `POST /detect` - Upload image for detection  --max-depth 25 \

  --test-split 0.2 \

**Test the API:**  --use-checkpoint \

```powershell  --output saved-models/my_model.pkl

# Health check```

Invoke-RestMethod -Uri 'http://localhost:5000/health' | ConvertTo-Json

### Training parameters:

# Model info- `--dataset`: Path to dataset folder (default: `training_data/Train`)

Invoke-RestMethod -Uri 'http://localhost:5000/model-info' | ConvertTo-Json- `--fix-method`: How to handle mislabeled folders (`auto`/`swap`/`dynamic`)

- `--max-samples`: Max samples per class (None = all images)

# Detect (upload image)- `--n-estimators`: Number of Random Forest trees (default: 500)

$response = Invoke-RestMethod -Uri 'http://localhost:5000/detect' -Method Post -Form @{ image = Get-Item 'test_image.jpg' }- `--max-depth`: Max tree depth (default: 25)

$response | ConvertTo-Json -Depth 5- `--test-split`: Test set ratio (default: 0.2)

```- `--use-checkpoint`: Enable checkpointing every 5000 images

- `--output`: Output model path

## 📊 Model Architecture

## 🌐 Web Server Usage

```

Input: (150, 150, 3)### Start server:

↓```bash

Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)python app.py

↓```

Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)

↓### With custom settings:

Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)```bash

↓python app.py \

Conv2D(256) → BatchNorm → MaxPool → Dropout(0.3)  --model saved-models/deepfake_detector_improved.pkl \

↓  --host 0.0.0.0 \

Flatten  --port 8080

↓```

Dense(512) → BatchNorm → Dropout(0.5)

↓### Server endpoints:

Dense(256) → Dropout(0.4)- `GET /` - Main web interface

↓- `POST /predict` - Prediction API (upload image)

Dense(2, softmax)- `GET /health` - Health check

```- `GET /model-info` - Model information



**Features:**### API Example (curl):

- 4 convolutional blocks with increasing filters (32→256)```bash

- Batch normalization for training stabilitycurl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict

- Dropout for regularization (0.25-0.5)```

- Adam optimizer with learning rate 0.0001

- Sparse categorical crossentropy loss### API Response:

```json

## 🔧 Data Augmentation{

  "success": true,

Training uses real-time augmentation:  "prediction": "real",

- Rotation: ±20°  "confidence": 78.5,

- Width/height shift: ±20%  "probabilities": {

- Zoom: ±15%    "real": 78.5,

- Shear: ±15%    "fake": 21.5

- Horizontal flip  }

- Rescaling to [0, 1]}

```

Validation uses only rescaling (no augmentation).

## 🔧 Model Architecture

## 📈 Training Tips

**Features (71 dimensions):**

**For best results:**- RGB/HSV/LAB color statistics

1. Use GPU if available (TensorFlow will auto-detect)- Texture analysis (Gabor-like, local variance)

2. Start with smaller batch size if you run out of memory- Edge detection (Canny, Sobel)

3. Monitor validation accuracy - stop if it plateaus- Frequency domain (DCT for GAN artifacts)

4. Expected training time on 140k images:- Noise estimation

   - GPU (RTX 3060+): 2-4 hours for 50 epochs- Symmetry analysis

   - CPU: 10-20 hours for 50 epochs

**Classifier:**

**If training is slow:**- Random Forest with 500 trees

- Reduce `--batch_size` to 16 or 8- Max depth: 25

- Reduce `--input_size` to 128 or 100- Regularization to prevent overfitting

- Use fewer `--epochs` (30-40 may be sufficient)

## 📊 Expected Performance

## 🐛 Troubleshooting

With the full 140,000 image dataset:

**Model not loading:**- **Training accuracy:** ~92%

- Check that `saved-models/deepfake_cnn_model.h5` exists- **Test accuracy:** ~71%

- Verify TensorFlow version is 2.15+- **Real image detection:** ~71%

- **Fake image detection:** ~70%

**Out of memory during training:**

- Reduce batch size: `--batch_size=16`## 🐛 Troubleshooting

- Reduce input size: `--input_size=128`

- Close other applications### Model predicting incorrectly?

The model is trained to handle the swapped labels. If predictions seem inverted:

**Low accuracy:**```bash

- Train for more epochs (50-100)# Check if dataset is marked as swapped

- Check dataset is balanced (equal real/fake counts)python image-detection-module/utils/dataset_utils.py --action stats

- Verify images are correctly labeled

- Try increasing model capacity or using transfer learning# If not marked, mark it

python image-detection-module/utils/dataset_utils.py --action mark_swapped

**Import errors:**```

- Run `pip install -r requirements.txt`

- Ensure Python 3.10+ is being used### Training taking too long?

- Check all modules are in `image-detection-module/model/`Use checkpointing and limit samples:

```bash

## 📁 Project Structurepython train.py --max-samples 10000 --use-checkpoint

```

```

Deepfake_image_detector/### Import errors?

├── image-detection-module/Make sure you're running from the project root:

│   ├── model/```bash

│   │   ├── model_cnn.py          # Main CNN detector classcd /path/to/Deepfake_image_detector

│   │   ├── train.py               # Training scriptpython app.py

│   │   └── dataset_utils.py       # Dataset utilities (optional)```

│   ├── api/

│   │   └── app.py                 # Flask REST API## 📁 Project Structure

│   └── frontend/

│       ├── index.html```

│       └── style.cssDeepfake_image_detector/

├── saved-models/├── app.py                              # Flask web server (NEW)

│   └── deepfake_cnn_model.h5     # Trained model (after training)├── train.py                            # Training script with auto-correction (NEW)

├── training_data/├── image-detection-module/

│   └── Train/│   ├── model/

│       ├── Real/│   │   ├── model_simple_cnn.py        # Model with label correction

│       └── Fake/│   │   └── train_simple.py            # Original training script

├── requirements.txt│   ├── utils/

└── README.md│   │   └── dataset_utils.py           # Dataset fixing utilities (NEW)

```│   └── api/

│       └── app.py                      # Original API server

## 🎓 Next Steps├── training_data/

│   └── Train/

1. **Train your model** with the full dataset│       ├── Real/                       # Actually contains FAKE images!

2. **Validate** predictions match image content│       ├── Fake/                       # Actually contains REAL images!

3. **Deploy** the Flask API for production use│       └── .dataset_config.json        # Auto-generated config

4. **Monitor** model performance and retrain if needed├── saved-models/

│   └── deepfake_detector_improved.pkl

## 📝 License└── README.md                           # This file (NEW)

```

MIT License - feel free to use for your projects.

## 🎯 Next Steps

---

1. **Mark your dataset:**

**Need help?** Check the code comments in `model_cnn.py` and `train.py` for detailed documentation.   ```bash

   python image-detection-module/utils/dataset_utils.py --action mark_swapped
   ```

2. **Train the model:**
   ```bash
   python train.py --fix-method dynamic --use-checkpoint
   ```

3. **Start the server:**
   ```bash
   python app.py
   ```

4. **Test it out:**
   - Open http://localhost:5000
   - Upload a real image → should say "REAL"
   - Upload a fake image → should say "FAKE"

## 📝 Notes

- The model in `model_simple_cnn.py` has been updated to handle label correction at prediction time
- Always use the `dataset_utils.py` or `train.py` scripts to ensure proper label handling
- Backups are created automatically when swapping folders
- Checkpoints save progress every 5000 images during training

## 🤝 Support

If you encounter issues:
1. Check dataset statistics: `python image-detection-module/utils/dataset_utils.py --action stats`
2. Verify dataset integrity: `python image-detection-module/utils/dataset_utils.py --action verify`
3. Check server health: `curl http://localhost:5000/health`

---

**Made with ❤️ for accurate deepfake detection**
