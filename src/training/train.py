"""Letter Detection and Classification System using a CNN.
This script trains a CNN on individual letters and predicts multiple letters in an image."""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json
from ..segment import segment_letters as external_segment_letters

class LetterDetectionCNN:
    def __init__(self, img_size=64, num_classes=27):
        """
        Initializes the letter detection system.
        
        Args:
            img_size: Size of the input images (img_size x img_size).
            num_classes: Number of classes (26 for A-Z + 1 for '?').
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.label_map = {}
        
    def build_model(self):
        """
        Builds the CNN architecture.
        Optimized for letter recognition.
        """
        model = models.Sequential([
            # Convolutional Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_size, self.img_size, 1),
                         padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Convolutional Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Convolutional Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model built successfully!")
        print(model.summary())
        
        return model
    
    def load_training_data(self, dataset_path, augment_per_image=100):
        """
        Loads the dataset of individual letters (1 image per letter)
        and generates augmented versions to create a robust dataset.
        
        Expected structure:
        dataset_path/
            A.png
            B.png
            C.png
            ...
        
        Args:
            augment_per_image: Number of augmented images to generate per original image.
        """
        images = []
        labels = []
        
        print(f"Loading dataset from: {dataset_path}")
        print(f"Generating {augment_per_image} variations per original image")
        
        # Load all images (format: LETTER.png)
        for img_file in sorted(os.listdir(dataset_path)):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                # Extract letter from filename
                raw_label = img_file.split('.')[0].upper()
                label = '?' if raw_label == 'HIDDEN' else raw_label
                
                # Check if it's a valid letter (A-Z) or '?'
                if (len(label) == 1 and label.isalpha()) or label == '?':
                    if label not in self.label_map:
                        self.label_map[label] = len(self.label_map)
                    
                    img_path = os.path.join(dataset_path, img_file)
                    img = self._load_and_preprocess_image(img_path)
                    
                    if img is not None:
                        # Add original image
                        images.append(img)
                        labels.append(self.label_map[label])
                        
                        # Generate augmented versions
                        augmented = self._augment_image(img, augment_per_image)
                        images.extend(augmented)
                        labels.extend([self.label_map[label]] * len(augmented))
                        
                        print(f"  Loaded: {label} - Total: {len(augmented) + 1} images")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"\nTotal images loaded: {len(images)}")
        print(f"Classes found ({len(self.label_map)}): {sorted(self.label_map.keys())}")
        print(f"Image shape: {images.shape}")
        print(f"Distribution per class: ~{len(images) // len(self.label_map)} images/class")
        
        return images, labels
    
    def _augment_image(self, img, num_augmentations):
        """
        Generates multiple augmented versions of a single image.
        Applies heavy random transformations to create variety.
        """
        augmented_images = []
        
        # Remove channel dimension for processing
        img_2d = img[:, :, 0]
        
        for _ in range(num_augmentations):
            aug_img = img_2d.copy()
            
            # 1. Random rotation (-20 to +20 degrees)
            angle = np.random.uniform(-20, 20)
            center = (self.img_size // 2, self.img_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (self.img_size, self.img_size),
                                     borderValue=1.0)
            
            # 2. Random translation
            tx = np.random.uniform(-0.15, 0.15) * self.img_size
            ty = np.random.uniform(-0.15, 0.15) * self.img_size
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            aug_img = cv2.warpAffine(aug_img, M, (self.img_size, self.img_size),
                                     borderValue=1.0)
            
            # 3. Random scaling (zoom)
            scale = np.random.uniform(0.8, 1.2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            aug_img = cv2.warpAffine(aug_img, M, (self.img_size, self.img_size),
                                     borderValue=1.0)
            
            # 4. Light perspective distortion
            if np.random.random() > 0.5:
                pts1 = np.float32([[0, 0], [self.img_size, 0], 
                                   [0, self.img_size], [self.img_size, self.img_size]])
                offset = self.img_size * 0.1
                pts2 = pts1 + np.random.uniform(-offset, offset, pts1.shape)
                pts2 = np.clip(pts2, 0, self.img_size)
                try:
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    aug_img = cv2.warpPerspective(aug_img, M, (self.img_size, self.img_size),
                                                 borderValue=1.0)
                except:
                    pass  # If it fails, keep the image without transformation
            
            # 5. Add Gaussian noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.05, aug_img.shape)
                aug_img = np.clip(aug_img + noise, 0, 1)
            
            # 6. Brightness adjustment
            brightness = np.random.uniform(0.7, 1.3)
            aug_img = np.clip(aug_img * brightness, 0, 1)
            
            # 7. Random blur (simulates different image qualities)
            if np.random.random() > 0.7:
                ksize = np.random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)
            
            # 8. Erosion/Dilation (simulates different thicknesses)
            if np.random.random() > 0.6:
                kernel = np.ones((2, 2), np.uint8)
                if np.random.random() > 0.5:
                    aug_img = cv2.erode(aug_img, kernel, iterations=1)
                else:
                    aug_img = cv2.dilate(aug_img, kernel, iterations=1)
            
            # Add channel dimension back
            aug_img = np.expand_dims(aug_img, axis=-1)
            augmented_images.append(aug_img)
        
        return augmented_images
    
    def _load_and_preprocess_image(self, img_path):
        """Loads and preprocesses a single image."""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            
            return img
        except Exception as e:
            print(f"Erro ao carregar {img_path}: {e}")
            return None
    
    def train(self, dataset_path, epochs=50, batch_size=32, validation_split=0.2, 
              augment_per_image=100):
        """
        Trains the model with the individual letter dataset.
        
        Args:
            augment_per_image: Number of variations to generate per original image.
        """
        X, y = self.load_training_data(dataset_path, augment_per_image)
        
        y_categorical = keras.utils.to_categorical(y, self.num_classes)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"\nTraining data: {X_train.shape}")
        print(f"Validation data: {X_val.shape}")
        
        # Additional light data augmentation during training
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='constant',
            cval=1.0
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                '../model/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        print("\nStarting training...")
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self._plot_training_history(history)
        
        return history
    
    def _plot_training_history(self, history):
        """Plots accuracy and loss graphs during training."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history.history['accuracy'], label='Treino')
        ax1.plot(history.history['val_accuracy'], label='Validação')
        ax1.set_title('Acurácia do Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Acurácia')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history.history['loss'], label='Treino')
        ax2.plot(history.history['val_loss'], label='Validação')
        ax2.set_title('Perda do Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Perda')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('../training_history.png', dpi=300, bbox_inches='tight')
        print("\nGraph saved as '../training_history.png'")
        plt.show()
    
    def predict_multiple_letters(self, image_path, confidence_threshold=0.5):
        """
        Predicts all letters in an image with multiple letters.
        """
        segmented_letters, original_img = external_segment_letters(image_path, debug=False)

        letters = []
        for ld in segmented_letters:
            img_gray = ld.get('image')

            if img_gray is None:
                if 'image_binary' in ld and ld['image_binary'] is not None:
                    img_gray = ld['image_binary']
                elif 'image_color' in ld and ld['image_color'] is not None:
                    try:
                        img_gray = cv2.cvtColor(ld['image_color'], cv2.COLOR_BGR2GRAY)
                    except Exception:
                        img_gray = None

            if img_gray is None:
                continue

            letters.append({
                'image': img_gray,
                'position': ld.get('position', (ld.get('x'), ld.get('y'), ld.get('w'), ld.get('h'))),
                'original_region': ld
            })

        print(f"\nFound {len(letters)} letters in the image")

        predictions = []
        reverse_label_map = {v: k for k, v in self.label_map.items()}

        for i, letter_data in enumerate(letters):
            img = cv2.resize(letter_data['image'], (self.img_size, self.img_size))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)

            prediction = self.model.predict(img, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]

            if confidence >= confidence_threshold:
                predicted_letter = reverse_label_map.get(predicted_class, '?')
                predictions.append({
                    'letter': predicted_letter,
                    'confidence': float(confidence),
                    'position': letter_data['position'],
                    'index': i
                })
        
        self._visualize_predictions(original_img, predictions)
        
        word = ''.join([p['letter'] for p in predictions])
        print(f"\nDetected word: {word}")
        print(f"Average confidence: {np.mean([p['confidence'] for p in predictions]):.2%}")
        
        return predictions, word
    
    def _visualize_predictions(self, img, predictions):
        """Visualizes the predictions on the original image."""
        img_copy = img.copy()
        
        for pred in predictions:
            x, y, w, h = pred['position']
            
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            label = f"{pred['letter']} ({pred['confidence']:.2%})"
            cv2.putText(
                img_copy, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        """ # Salvar e mostrar
        cv2.imwrite('../prediction_result.png', img_copy)
        print("Resultado salvo como '../prediction_result.png'")
        
        # Mostrar
        plt.figure(figsize=(15, 5))
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Detecção e Classificação de Letras')
        plt.tight_layout()
        plt.savefig('../prediction_visualization.png', dpi=300, bbox_inches='tight')
        plt.show() """
    
    def save_model(self, filepath='../model/letter_detector_model.h5'):
        """Saves the trained model."""
        self.model.save(filepath)
        
        with open('../model/label_map.json', 'w') as f:
            json.dump(self.label_map, f)
        
        print(f"\nModel saved to: {filepath}")
        print(f"Label map saved to: ../model/label_map.json")
    
    def load_model(self, filepath='../model/letter_detector_model.h5', label_map_path='../model/label_map.json'):
        """Loads a trained model."""
        self.model = keras.models.load_model(filepath)
        
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
        print(f"Model loaded from: {filepath}")


if __name__ == "__main__":
    detector = LetterDetectionCNN(img_size=64, num_classes=27)
    
    """ detector.build_model() """
    
    dataset_path = r"../data/dataset"
    
    """ 
    print("\n" + "="*60)
    print("INICIANDO TREINAMENTO")
    print("="*60)
    
    history = detector.train(
        dataset_path=dataset_path,
        epochs=50,
        batch_size=32,
        validation_split=0.2
    )

     """
    detector.load_model('../model/letter_detector_final.h5')

    detector.model.predict(np.zeros((1, detector.img_size, detector.img_size, 1)))

    if "?" not in detector.label_map:
        detector.label_map["?"] = len(detector.label_map)
        print("Classe '?' adicionada ao label_map:", detector.label_map)

    for layer in detector.model.layers[:-1]:
        layer.trainable = False

    from tensorflow.keras import layers, models

    x = detector.model.layers[-2].output
    new_output = layers.Dense(27, activation='softmax', name='new_predictions')(x)

    detector.model = models.Model(inputs=detector.model.inputs, outputs=new_output)

    detector.model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = detector.train(
        dataset_path=dataset_path,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    
    detector.save_model('../model/letter_detector_with_question.h5')

    """ 
    # FAZER PREDIÇÕES
    print("\n" + "="*60)
    print("FAZENDO PREDIÇÕES")
    print("="*60)
    
    test_image_path = r"C:\Users\Guilherme\Downloads\Nova pasta (2)\treino\1.png"
    predictions, word = detector.predict_multiple_letters(
        test_image_path,
        confidence_threshold=0.5
    )
    
    print("\n" + "="*60)
    print("RESULTADOS DETALHADOS")
    print("="*60)
    for i, pred in enumerate(predictions):
        print(f"Letra {i+1}: {pred['letter']} (Confiança: {pred['confidence']:.2%})")
 """
