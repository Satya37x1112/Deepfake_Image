"""
Ultra-Fast Deepfake Detector using Transfer Learning
Uses pre-trained MobileNetV2 - trains in 5-10 minutes with better accuracy!
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# GPU optimization
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"✓ GPU detected: {len(physical_devices)} device(s)")


class FastDeepfakeDetector:
    """Transfer learning model - 10x faster than training from scratch"""
    
    def __init__(self, input_shape=(96, 96, 3)):
        self.input_shape = input_shape
        self.model = None
        self.label_map = {'Fake': 0, 'Real': 1}
        
    def build_model(self):
        """Build model with pre-trained MobileNetV2"""
        print("Loading pre-trained MobileNetV2 (trained on ImageNet)...")
        
        # Load pre-trained MobileNetV2 without top layers
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model (faster training)
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(2, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with higher learning rate (faster convergence)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Model built with transfer learning")
        print(f"  Trainable params: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        return self.model
    
    def train(self, train_dir, epochs=10, batch_size=32, max_samples=None):
        """Fast training with transfer learning"""
        
        if self.model is None:
            self.build_model()
        
        # Prepare subset if requested
        if max_samples:
            print(f"\n⚡ FAST MODE: Using {max_samples} samples per class")
            train_dir = self._create_subset(train_dir, max_samples)
        
        # Data generators with light augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=15,
            horizontal_flip=True
        )
        
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            subset='training',
            shuffle=True
        )
        
        val_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation',
            shuffle=False
        )
        
        self.label_map = train_gen.class_indices
        
        print(f"\n{'='*80}")
        print("FAST TRAINING (Transfer Learning)")
        print(f"{'='*80}")
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Epochs: {epochs}")
        print(f"{'='*80}\n")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'saved-models/deepfake_fast.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n{'='*80}")
        print("✓ TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
        print(f"Model saved: saved-models/deepfake_fast.h5")
        print(f"{'='*80}")
        
        return history
    
    def _create_subset(self, train_dir, max_samples):
        """Create small subset for ultra-fast training"""
        import shutil
        import tempfile
        import random
        
        temp_dir = tempfile.mkdtemp(prefix='deepfake_fast_')
        
        for class_name in ['Real', 'Fake']:
            src_dir = os.path.join(train_dir, class_name)
            dst_dir = os.path.join(temp_dir, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            
            files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            selected = random.sample(files, min(max_samples, len(files)))
            
            for f in selected:
                shutil.copy2(os.path.join(src_dir, f), dst_dir)
        
        return temp_dir
    
    def load_model_file(self, path):
        """Load trained model"""
        self.model = tf.keras.models.load_model(path)
        print(f"✓ Model loaded from: {path}")
    
    def predict(self, image_path):
        """Predict single image"""
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        
        img = load_img(image_path, target_size=self.input_shape[:2])
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = self.model.predict(img_array, verbose=0)[0]
        pred_class = np.argmax(preds)
        
        inv_map = {v: k for k, v in self.label_map.items()}
        label = inv_map.get(pred_class, 'unknown')
        label = 'real' if label == 'Real' else 'fake'
        
        return {
            'prediction': label,
            'label': int(pred_class),
            'confidence': float(preds[pred_class] * 100),
            'probabilities': {
                'real': float(preds[self.label_map.get('Real', 1)] * 100),
                'fake': float(preds[self.label_map.get('Fake', 0)] * 100)
            }
        }


def main():
    """Ultra-fast training"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='../training_data/Train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=2000, 
                       help='Samples per class (2000 = 5-10min, 10000 = 20-30min)')
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.normpath(os.path.join(script_dir, args.train_dir))
    
    print("\n" + "="*80)
    print("⚡ ULTRA-FAST DEEPFAKE DETECTOR (Transfer Learning)")
    print("="*80)
    print("Using pre-trained MobileNetV2 - 10x faster than training from scratch!")
    print(f"Training on {args.max_samples*2:,} images ({args.max_samples} per class)")
    print(f"Expected time: 5-10 minutes")
    print("="*80 + "\n")
    
    # Create detector
    detector = FastDeepfakeDetector(input_shape=(96, 96, 3))
    
    # Train
    detector.train(
        train_dir=train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    print("\n✓ Ready to use!")
    print("\nTest it:")
    print("  detector = FastDeepfakeDetector()")
    print("  detector.load_model_file('saved-models/deepfake_fast.h5')")
    print("  print(detector.predict('test.jpg'))")


if __name__ == "__main__":
    main()
