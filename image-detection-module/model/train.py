"""
Training Script for Deepfake CNN Model
Compatible with Python 3.13, TensorFlow 2.x

Trains a CNN model on 140k+ images with proper dataset handling,
label mapping detection, and progress tracking.
"""

import os
import sys
import argparse
from pathlib import Path

# Add model directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_cnn import DeepfakeCNNDetector


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Train CNN Deepfake Detector'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='../training_data/Train',
        help='Path to training directory containing Real/ and Fake/ folders'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training (increase for faster training if GPU available)'
    )
    parser.add_argument(
        '--validation_split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='../../saved-models/deepfake_cnn_model.h5',
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=128,
        help='Input image size (smaller = faster training, e.g., 96, 128, 150)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Max samples per class for quick testing (e.g., 1000 for fast test)'
    )
    
    args = parser.parse_args()
    
    # Convert paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.normpath(os.path.join(script_dir, args.train_dir))
    save_path = os.path.normpath(os.path.join(script_dir, args.save_path))
    
    print("=" * 80)
    print("DEEPFAKE CNN TRAINING")
    print("=" * 80)
    print(f"Training directory: {train_dir}")
    print(f"Save path: {save_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Validation split: {args.validation_split}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    print("=" * 80 + "\n")
    
    # Verify training directory exists
    if not os.path.exists(train_dir):
        print(f"ERROR: Training directory not found: {train_dir}")
        print("Please ensure the directory contains 'Real' and 'Fake' subdirectories")
        return
    
    # Check for Real and Fake folders
    real_dir = os.path.join(train_dir, 'Real')
    fake_dir = os.path.join(train_dir, 'Fake')
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"ERROR: Training directory must contain 'Real' and 'Fake' subdirectories")
        print(f"  Real folder: {real_dir} - {'Found' if os.path.exists(real_dir) else 'NOT FOUND'}")
        print(f"  Fake folder: {fake_dir} - {'Found' if os.path.exists(fake_dir) else 'NOT FOUND'}")
        return
    
    # Count images
    real_images = len([f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    fake_images = len([f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Handle max_samples for quick testing
    if args.max_samples:
        print(f"⚠️  QUICK TEST MODE: Using only {args.max_samples} samples per class")
        # Create temporary subset directories (symbolic would be ideal, but copy for compatibility)
        import shutil
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='deepfake_subset_')
        temp_real = os.path.join(temp_dir, 'Real')
        temp_fake = os.path.join(temp_dir, 'Fake')
        os.makedirs(temp_real, exist_ok=True)
        os.makedirs(temp_fake, exist_ok=True)
        
        # Copy subset of files
        import random
        real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        random.seed(42)
        real_subset = random.sample(real_files, min(args.max_samples, len(real_files)))
        fake_subset = random.sample(fake_files, min(args.max_samples, len(fake_files)))
        
        print(f"  Copying {len(real_subset)} real images...")
        for src in real_subset:
            shutil.copy2(src, temp_real)
        
        print(f"  Copying {len(fake_subset)} fake images...")
        for src in fake_subset:
            shutil.copy2(src, temp_fake)
        
        # Use temp directory as training directory
        train_dir = temp_dir
        real_images = len(real_subset)
        fake_images = len(fake_subset)
    
    print(f"\nDataset information:")
    print(f"  Real images: {real_images:,}")
    print(f"  Fake images: {fake_images:,}")
    print(f"  Total: {real_images + fake_images:,}")
    print()
    
    # Initialize detector
    detector = DeepfakeCNNDetector(
        input_shape=(args.input_size, args.input_size, 3),
        num_classes=2
    )
    
    # Build and show model architecture
    detector.build_model()
    
    # Train model
    print("\nStarting training...")
    print("This may take several hours depending on dataset size and hardware.")
    print("Progress will be shown below.\n")
    
    try:
        history = detector.train(
            train_dir=train_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            save_path=save_path
        )
        
        print("\n" + "=" * 80)
        print("✓ TRAINING SUCCESSFUL")
        print("=" * 80)
        print(f"Model saved to: {save_path}")
        print(f"\nTo use this model:")
        print(f"  from model_cnn import DeepfakeCNNDetector")
        print(f"  detector = DeepfakeCNNDetector()")
        print(f"  detector.load_model_file('{save_path}')")
        print(f"  result = detector.predict('path/to/image.jpg')")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial progress may have been saved by ModelCheckpoint callback.")
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
