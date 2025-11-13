"""
Sistema de Detecção e Classificação de Letras usando CNN
Treina uma CNN com letras individuais e prediz múltiplas letras em uma imagem
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json
from segmentar import segment_letters as external_segment_letters, get_tile_color_images

class LetterDetectionCNN:
    def __init__(self, img_size=64, num_classes=26):
        """
        Inicializa o sistema de detecção de letras
        
        Args:
            img_size: Tamanho das imagens de entrada (img_size x img_size)
            num_classes: Número de classes (26 para A-Z)
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.label_map = {}
        
    def build_model(self):
        """
        Constrói a arquitetura da CNN
        Arquitetura otimizada para reconhecimento de letras
        """
        model = models.Sequential([
            # Bloco Convolucional 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_size, self.img_size, 1),
                         padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloco Convolucional 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloco Convolucional 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Camadas Densas
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
        print("Modelo construído com sucesso!")
        print(model.summary())
        
        return model
    
    def load_training_data(self, dataset_path, augment_per_image=100):
        """
        Carrega o dataset de letras individuais (1 imagem por letra)
        e gera versões aumentadas para criar um dataset robusto
        
        Estrutura esperada:
        dataset_path/
            A.png
            B.png
            C.png
            ...
        
        Args:
            augment_per_image: Número de imagens aumentadas a gerar por imagem original
        """
        images = []
        labels = []
        
        print(f"Carregando dataset de: {dataset_path}")
        print(f"Gerando {augment_per_image} variações por imagem original")
        
        # Carregar todas as imagens (formato: LETRA.png)
        for img_file in sorted(os.listdir(dataset_path)):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                # Extrair letra do nome do arquivo
                label = img_file.split('.')[0].upper()
                
                # Verificar se é uma letra válida (A-Z)
                if len(label) == 1 and label.isalpha():
                    if label not in self.label_map:
                        self.label_map[label] = len(self.label_map)
                    
                    img_path = os.path.join(dataset_path, img_file)
                    img = self._load_and_preprocess_image(img_path)
                    
                    if img is not None:
                        # Adicionar imagem original
                        images.append(img)
                        labels.append(self.label_map[label])
                        
                        # Gerar versões aumentadas
                        augmented = self._augment_image(img, augment_per_image)
                        images.extend(augmented)
                        labels.extend([self.label_map[label]] * len(augmented))
                        
                        print(f"  Carregada: {label} - Total: {len(augmented) + 1} imagens")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"\nTotal de imagens carregadas: {len(images)}")
        print(f"Classes encontradas ({len(self.label_map)}): {sorted(self.label_map.keys())}")
        print(f"Shape das imagens: {images.shape}")
        print(f"Distribuição por classe: ~{len(images) // len(self.label_map)} imagens/classe")
        
        return images, labels
    
    def _augment_image(self, img, num_augmentations):
        """
        Gera múltiplas versões aumentadas de uma única imagem
        Aplica transformações aleatórias pesadas para criar variedade
        """
        augmented_images = []
        
        # Remover dimensão do canal para processamento
        img_2d = img[:, :, 0]
        
        for _ in range(num_augmentations):
            # Copiar imagem
            aug_img = img_2d.copy()
            
            # 1. Rotação aleatória (-20 a +20 graus)
            angle = np.random.uniform(-20, 20)
            center = (self.img_size // 2, self.img_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (self.img_size, self.img_size),
                                     borderValue=1.0)
            
            # 2. Translação aleatória
            tx = np.random.uniform(-0.15, 0.15) * self.img_size
            ty = np.random.uniform(-0.15, 0.15) * self.img_size
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            aug_img = cv2.warpAffine(aug_img, M, (self.img_size, self.img_size),
                                     borderValue=1.0)
            
            # 3. Escala aleatória (zoom)
            scale = np.random.uniform(0.8, 1.2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            aug_img = cv2.warpAffine(aug_img, M, (self.img_size, self.img_size),
                                     borderValue=1.0)
            
            # 4. Distorção de perspectiva leve
            if np.random.random() > 0.5:
                pts1 = np.float32([[0, 0], [self.img_size, 0], 
                                   [0, self.img_size], [self.img_size, self.img_size]])
                offset = self.img_size * 0.1
                pts2 = pts1 + np.random.uniform(-offset, offset, pts1.shape)
                # Garantir que os pontos estejam dentro dos limites
                pts2 = np.clip(pts2, 0, self.img_size)
                try:
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    aug_img = cv2.warpPerspective(aug_img, M, (self.img_size, self.img_size),
                                                 borderValue=1.0)
                except:
                    pass  # Se falhar, manter a imagem sem transformação
            
            # 5. Adicionar ruído gaussiano
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.05, aug_img.shape)
                aug_img = aug_img + noise
                aug_img = np.clip(aug_img, 0, 1)
            
            # 6. Ajuste de brilho
            brightness = np.random.uniform(0.7, 1.3)
            aug_img = aug_img * brightness
            aug_img = np.clip(aug_img, 0, 1)
            
            # 7. Blur aleatório (simula diferentes qualidades de imagem)
            if np.random.random() > 0.7:
                ksize = np.random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)
            
            # 8. Erosão/Dilatação (simula diferentes espessuras)
            if np.random.random() > 0.6:
                kernel = np.ones((2, 2), np.uint8)
                if np.random.random() > 0.5:
                    aug_img = cv2.erode(aug_img, kernel, iterations=1)
                else:
                    aug_img = cv2.dilate(aug_img, kernel, iterations=1)
            
            # Adicionar dimensão do canal de volta
            aug_img = np.expand_dims(aug_img, axis=-1)
            augmented_images.append(aug_img)
        
        return augmented_images
    
    def _load_and_preprocess_image(self, img_path):
        """Carrega e preprocessa uma imagem individual"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Redimensionar
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalizar
            img = img.astype('float32') / 255.0
            
            # Adicionar dimensão do canal
            img = np.expand_dims(img, axis=-1)
            
            return img
        except Exception as e:
            print(f"Erro ao carregar {img_path}: {e}")
            return None
    
    def train(self, dataset_path, epochs=50, batch_size=32, validation_split=0.2, 
              augment_per_image=100):
        """
        Treina o modelo com o dataset de letras individuais
        
        Args:
            augment_per_image: Número de variações a gerar por imagem original
        """
        # Carregar dados (já com augmentation)
        X, y = self.load_training_data(dataset_path, augment_per_image)
        
        # Converter labels para one-hot encoding
        y_categorical = keras.utils.to_categorical(y, self.num_classes)
        
        # Dividir em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42, stratify=y_categorical
        )
        
        print(f"\nDados de treino: {X_train.shape}")
        print(f"Dados de validação: {X_val.shape}")
        
        # Data Augmentation adicional durante o treinamento (mais leve)
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='constant',
            cval=1.0
        )
        
        # Callbacks
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
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Treinar
        print("\nIniciando treinamento...")
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Plotar resultados
        self._plot_training_history(history)
        
        return history
    
    def _plot_training_history(self, history):
        """Plota gráficos de acurácia e perda durante o treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Acurácia
        ax1.plot(history.history['accuracy'], label='Treino')
        ax1.plot(history.history['val_accuracy'], label='Validação')
        ax1.set_title('Acurácia do Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Acurácia')
        ax1.legend()
        ax1.grid(True)
        
        # Perda
        ax2.plot(history.history['loss'], label='Treino')
        ax2.plot(history.history['val_loss'], label='Validação')
        ax2.set_title('Perda do Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Perda')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("\nGráfico salvo como 'training_history.png'")
        plt.show()
    
    def segment_letters(self, image_path):
        """
        Segmenta letras individuais de uma imagem com múltiplas letras
        Usa projeção vertical e detecção de contornos
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarização adaptativa
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar e ordenar contornos
        letter_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar contornos muito pequenos (ruído)
            if w > 15 and h > 15:
                letter_regions.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'contour': contour
                })
        
        # Ordenar da esquerda para direita
        letter_regions = sorted(letter_regions, key=lambda r: r['x'])
        
        # Extrair imagens das letras
        segmented_letters = []
        for region in letter_regions:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Adicionar padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(gray.shape[1], x + w + padding)
            y2 = min(gray.shape[0], y + h + padding)
            
            letter_img = gray[y1:y2, x1:x2]
            
            segmented_letters.append({
                'image': letter_img,
                'position': (x, y, w, h),
                'original_region': region
            })
        
        return segmented_letters, img
    
    def predict_multiple_letters(self, image_path, confidence_threshold=0.5):
        """
        Prediz todas as letras em uma imagem com múltiplas letras
        """
        # Segmentar letras usando a implementação em `segmentar.py`
        segmented_letters, original_img = external_segment_letters(image_path, debug=False)

        # Converter para o formato esperado (lista de dicts com 'image' e 'position')
        letters = []
        for ld in segmented_letters:
            # Preferir imagem em tons de cinza já fornecida pelo `segmentar`
            img_gray = ld.get('image')

            # Se não houver imagem em gray, tentar usar binária ou converter a color
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

        print(f"\nEncontradas {len(letters)} letras na imagem")

        # Predizer cada letra
        predictions = []
        reverse_label_map = {v: k for k, v in self.label_map.items()}

        for i, letter_data in enumerate(letters):
            # Preprocessar
            img = cv2.resize(letter_data['image'], (self.img_size, self.img_size))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)

            # Predizer
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
        
        # Visualizar resultados
        self._visualize_predictions(original_img, predictions)
        
        # Retornar palavra completa
        word = ''.join([p['letter'] for p in predictions])
        print(f"\nPalavra detectada: {word}")
        print(f"Confiança média: {np.mean([p['confidence'] for p in predictions]):.2%}")
        
        return predictions, word
    
    def _visualize_predictions(self, img, predictions):
        """Visualiza as predições na imagem original"""
        img_copy = img.copy()
        
        for pred in predictions:
            x, y, w, h = pred['position']
            
            # Desenhar retângulo
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Adicionar texto
            label = f"{pred['letter']} ({pred['confidence']:.2%})"
            cv2.putText(
                img_copy, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        """ # Salvar e mostrar
        cv2.imwrite('prediction_result.png', img_copy)
        print("Resultado salvo como 'prediction_result.png'")
        
        # Mostrar
        plt.figure(figsize=(15, 5))
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Detecção e Classificação de Letras')
        plt.tight_layout()
        plt.savefig('prediction_visualization.png', dpi=300, bbox_inches='tight')
        plt.show() """
    
    def save_model(self, filepath='letter_detector_model.h5'):
        """Salva o modelo treinado"""
        self.model.save(filepath)
        
        # Salvar mapeamento de labels
        with open('label_map.json', 'w') as f:
            json.dump(self.label_map, f)
        
        print(f"\nModelo salvo em: {filepath}")
        print(f"Mapeamento de labels salvo em: label_map.json")
    
    def load_model(self, filepath='letter_detector_model.h5', label_map_path='label_map.json'):
        """Carrega um modelo treinado"""
        self.model = keras.models.load_model(filepath)
        
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
        print(f"Modelo carregado de: {filepath}")


# Exemplo de uso
if __name__ == "__main__":
    # Inicializar o sistema
    detector = LetterDetectionCNN(img_size=64, num_classes=26)
    
    # Construir o modelo
    detector.build_model()
    
    # TREINAR O MODELO
    # Substitua pelo caminho do seu dataset de letras individuais
    dataset_path = r"c:\Users\Guilherme\Downloads\Nova pasta (2)\treino\dataset"
    
    """ 
    print("\n" + "="*60)
    print("INICIANDO TREINAMENTO")
    print("="*60) """
    
    # Treinar
    """ history = detector.train(
        dataset_path=dataset_path,
        epochs=50,
        batch_size=32,
        validation_split=0.2
    ) """
    
    # Salvar modelo
    detector.load_model('letter_detector_final.h5')
    
    # FAZER PREDIÇÕES
    print("\n" + "="*60)
    print("FAZENDO PREDIÇÕES")
    print("="*60)
    
    # Testar com imagem de múltiplas letras
    test_image_path = r"C:\Users\Guilherme\Downloads\Nova pasta (2)\treino\1.png"
    predictions, word = detector.predict_multiple_letters(
        test_image_path,
        confidence_threshold=0.5
    )
    
    # Exibir resultados detalhados
    print("\n" + "="*60)
    print("RESULTADOS DETALHADOS")
    print("="*60)
    for i, pred in enumerate(predictions):
        print(f"Letra {i+1}: {pred['letter']} (Confiança: {pred['confidence']:.2%})")