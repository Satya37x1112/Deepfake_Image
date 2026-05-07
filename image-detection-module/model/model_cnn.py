"""
Deepfake Image Detection CNN Model
Compatible with Python 3.13, TensorFlow 2.x, Keras

Binary classification model to detect deepfake images using CNN architecture.
- Input: (150, 150, 3) RGB images
- Output: Probability for 'real' (0) and 'fake' (1)
- Uses Conv2D, MaxPooling, Dropout, BatchNormalization for robustness
- Includes data augmentation pipeline for 140k+ images
- GPU optimization with mixed precision for faster training
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model

# Enable GPU memory growth to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✓ GPU detected: {len(physical_devices)} device(s)")
    except:
        pass

# Enable mixed precision for faster training on modern GPUs
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✓ Mixed precision (FP16) enabled for faster training")
except:
    pass
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, 
    Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path


class DeepfakeCNNDetector:
    """
    CNN-based Deepfake Detector using TensorFlow/Keras
    """
    
    def __init__(self, input_shape=(150, 150, 3), num_classes=2):
        """
        Initialize the detector
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes (2 for binary: real/fake)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.label_map = {'Real': 0, 'Fake': 1}  # Folder name -> numeric label
        self.is_trained = False
        
    def build_model(self):
        """
        Build CNN architecture
        
        Architecture:
        - 3 Convolutional blocks (Conv2D + BatchNorm + MaxPool + Dropout)
        - Flatten + Dense layers with dropout
        - Softmax output for binary classification
        """
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=self.input_shape, name='conv1'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), name='pool1'),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), name='pool2'),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), name='pool3'),
            Dropout(0.25),
            
            # Block 4 (deeper)
            Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), name='pool4'),
            Dropout(0.3),
            
            # Fully connected layers
            Flatten(),
            Dense(512, activation='relu', name='fc1'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', name='fc2'),
            Dropout(0.4),
            Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("=" * 80)
        print("MODEL ARCHITECTURE")
        print("=" * 80)
        self.model.summary()
        print("=" * 80)
        
        return model
    
    def create_data_generators(self, train_dir, validation_split=0.2, 
                               batch_size=32, augment=True):
        """
        Create data generators for training and validation
        
        Args:
            train_dir: Path to training directory containing Real/ and Fake/ folders
            validation_split: Fraction of data to use for validation
            batch_size: Batch size for training
            augment: Whether to apply data augmentation
            
        Returns:
            train_generator, validation_generator
        """
        if augment:
            # Training data with augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            # Without augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',  # For sparse_categorical_crossentropy
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Validation generator
        validation_generator = val_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        # Store class indices (folder_name -> label)
        self.label_map = train_generator.class_indices
        print(f"\n✓ Data generators created")
        print(f"  Class mapping: {self.label_map}")
        print(f"  Training samples: {train_generator.samples}")
        print(f"  Validation samples: {validation_generator.samples}")
        print(f"  Batch size: {batch_size}")
        
        return train_generator, validation_generator
    
    def train(self, train_dir, epochs=50, batch_size=32, 
              validation_split=0.2, save_path='saved-models/deepfake_cnn_model.h5'):
        """
        Train the model
        
        Args:
            train_dir: Path to training directory
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split fraction
            save_path: Path to save the best model
            
        Returns:
            History object
        """
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(
            train_dir, 
            validation_split=validation_split,
            batch_size=batch_size,
            augment=True
        )
        
        # Callbacks
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("\n" + "=" * 80)
        print("TRAINING DEEPFAKE CNN MODEL")
        print("=" * 80)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Model will be saved to: {save_path}")
        print("=" * 80 + "\n")
        
        # Train model
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETED")
        print("=" * 80)
        print(f"Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
        print(f"Best model saved to: {save_path}")
        print("=" * 80 + "\n")
        
        return history
    
    def load_model_file(self, model_path):
        """
        Load a trained model from file
        
        Args:
            model_path: Path to saved model (.h5 file)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = load_model(model_path)
        self.is_trained = True
        
        print(f"✓ Model loaded from: {model_path}")
        return self.model
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        img = load_img(image_path, target_size=self.input_shape[:2])
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path):
        """
        Predict whether an image is real or fake
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Prediction result with probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Load a trained model first.")
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get predicted class
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class] * 100)
        
        # Map to human-readable labels
        inv_map = {v: k for k, v in self.label_map.items()}
        predicted_label = inv_map.get(predicted_class, 'unknown')
        predicted_label = 'real' if predicted_label == 'Real' else 'fake'
        
        # Build result dictionary
        result = {
            'prediction': predicted_label,
            'label': int(predicted_class),
            'confidence': confidence,
            'probabilities': {
                'real': float(predictions[self.label_map.get('Real', 0)] * 100),
                'fake': float(predictions[self.label_map.get('Fake', 1)] * 100)
            }
        }
        
        return result
    
    def batch_predict(self, image_paths):
        """
        Predict multiple images at once
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results


if __name__ == "__main__":
    print("=" * 80)
    print("DEEPFAKE CNN DETECTOR - TensorFlow/Keras Implementation")
    print("Compatible with Python 3.13")
    print("=" * 80)
    
    # Example usage
    detector = DeepfakeCNNDetector(input_shape=(150, 150, 3))
    print("\n✓ Detector initialized")
    print(f"  Input shape: {detector.input_shape}")
    print(f"  Number of classes: {detector.num_classes}")
    
    # Build model to show architecture
    detector.build_model()
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print("\n1. Training:")
    print("   detector = DeepfakeCNNDetector()")
    print("   detector.train(")
    print("       train_dir='image-detection-module/training_data/Train',")
    print("       epochs=50,")
    print("       batch_size=32")
    print("   )")
    print("\n2. Prediction:")
    print("   detector.load_model_file('saved-models/deepfake_cnn_model.h5')")
    print("   result = detector.predict('path/to/image.jpg')")
    print("   print(result)")
    print("=" * 80)
