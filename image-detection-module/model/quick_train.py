"""
Quick Train - Fast training for testing/development
Uses smaller subset and optimized settings for speed
"""

import os
import sys

# Add model directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_cnn import DeepfakeCNNDetector

def quick_train():
    """Quick training with optimized settings"""
    
    print("=" * 80)
    print("QUICK TRAINING MODE - Optimized for Speed")
    print("=" * 80)
    print("Settings:")
    print("  - Input size: 96x96 (faster)")
    print("  - Batch size: 64 (larger batches)")
    print("  - Epochs: 20 (fewer epochs)")
    print("  - Subset: Using all available data")
    print("=" * 80 + "\n")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.normpath(os.path.join(script_dir, '../training_data/Train'))
    save_path = os.path.normpath(os.path.join(script_dir, '../../saved-models/deepfake_cnn_quick.h5'))
    
    # Initialize detector with smaller input size for speed
    detector = DeepfakeCNNDetector(
        input_shape=(96, 96, 3),  # Smaller = faster
        num_classes=2
    )
    
    # Build model
    detector.build_model()
    
    # Train with optimized settings
    try:
        history = detector.train(
            train_dir=train_dir,
            epochs=20,  # Fewer epochs
            batch_size=64,  # Larger batch size
            validation_split=0.2,
            save_path=save_path
        )
        
        print("\n" + "=" * 80)
        print("✓ QUICK TRAINING COMPLETED")
        print("=" * 80)
        print(f"Model saved to: {save_path}")
        print("\nTo test predictions:")
        print("  from model_cnn import DeepfakeCNNDetector")
        print(f"  detector = DeepfakeCNNDetector(input_shape=(96, 96, 3))")
        print(f"  detector.load_model_file('{save_path}')")
        print("  result = detector.predict('test_image.jpg')")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_train()
