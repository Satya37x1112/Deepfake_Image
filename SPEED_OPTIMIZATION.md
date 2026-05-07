# Speed Optimization Guide

## 🚀 Training is Slow? Here's How to Speed It Up

### Current Optimizations Applied

1. **Smaller Input Size**: 96x96 instead of 150x150
   - 2.4x fewer pixels to process
   - Faster data loading and augmentation

2. **Larger Batch Size**: 64 instead of 32
   - Better GPU utilization
   - Fewer training iterations per epoch

3. **Mixed Precision (FP16)**: Automatic if GPU supports it
   - 2x faster matrix operations
   - Lower memory usage

4. **GPU Memory Growth**: Prevents OOM errors
   - Allocates GPU memory as needed

5. **Fewer Epochs**: 20 instead of 50 for quick testing
   - Can increase later if needed

### Training Speed Comparison

**Original settings** (150x150, batch=32):
- CPU: ~15-20 hours for 140k images
- GPU: ~3-5 hours

**Optimized settings** (96x96, batch=64):
- CPU: ~8-12 hours  
- GPU: ~1-2 hours ✓

**Ultra-fast test** (96x96, batch=64, 2k samples):
- CPU: ~20-30 minutes
- GPU: ~5-10 minutes ✓

### Quick Training Options

#### Option 1: Quick Full Dataset (Recommended)
```powershell
python image-detection-module\model\quick_train.py
```
- Input: 96x96
- Batch: 64
- Epochs: 20
- Time: 1-2 hours (GPU) or 8-12 hours (CPU)

#### Option 2: Ultra-Fast Test (2000 samples)
```powershell
python image-detection-module\model\train.py --input_size=96 --batch_size=64 --epochs=15 --max_samples=1000
```
- Uses 1000 samples per class (2000 total)
- Time: 5-10 minutes (GPU) or 20-30 minutes (CPU)
- Good for testing if everything works

#### Option 3: Custom Settings
```powershell
python image-detection-module\model\train.py `
  --input_size=96 `
  --batch_size=64 `
  --epochs=30 `
  --validation_split=0.2
```

### Input Size vs Accuracy Trade-off

| Input Size | Speed  | Accuracy | Recommended For |
|-----------|--------|----------|-----------------|
| 96x96     | Fast   | Good     | Development, testing |
| 128x128   | Medium | Better   | Balanced option |
| 150x150   | Slow   | Best     | Production model |
| 224x224   | Slowest| Best     | Maximum accuracy |

### If Training is STILL Slow

1. **Check GPU is being used:**
   ```powershell
   python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
   ```

2. **Reduce batch size if out of memory:**
   ```powershell
   python image-detection-module\model\quick_train.py
   # Then manually edit quick_train.py and change batch_size to 32 or 16
   ```

3. **Use even smaller subset for testing:**
   ```powershell
   python image-detection-module\model\train.py --max_samples=500 --epochs=10
   ```

4. **Install GPU-optimized TensorFlow:**
   ```powershell
   pip install tensorflow[and-cuda]
   ```

### Current Training Status

Run this to check progress:
```powershell
# The quick_train.py is running in the background
# It will show progress with epoch numbers and ETA
```

### Expected Output

You should see:
- Epoch 1/20 progress bar
- Training accuracy increasing
- Validation accuracy tracked
- Model automatically saved when validation improves

### After Training Completes

Model will be saved to: `saved-models/deepfake_cnn_quick.h5`

Test it:
```python
from image_detection_module.model.model_cnn import DeepfakeCNNDetector

detector = DeepfakeCNNDetector(input_shape=(96, 96, 3))
detector.load_model_file('saved-models/deepfake_cnn_quick.h5')
result = detector.predict('test_image.jpg')
print(result)
```

Or start the API:
```powershell
python image-detection-module\api\app.py
```

---

**Pro tip:** Start with ultra-fast test mode to verify everything works, then run full training overnight or while you're away.
