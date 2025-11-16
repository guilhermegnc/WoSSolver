#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Words on Stream - Anagram Generator (final version with CNN)

Improvements applied:
- Use of pathlib, logging, and better exception handling
- Dictionary pre-processing for faster search (word counters, size index)
- Letter detection with CNN (convolutional neural networks)
- Tile segmentation using computer vision (segmentar.py)
- Image processing in a separate thread to avoid UI freezing
- Buttons to save results, copy to clipboard, and clear
- Clearer status messages and protection against multiple clicks

Additions in this version:
- Explicit support for the '?' symbol (hidden game letter)
- Testing all A-Z letters when '?' is present to generate
  probabilities and list possible words per letter

Requirements:
- Python 3.8+
- pip install pillow numpy onnxruntime opencv-python

Required pre-trained models:
- letter_detector_final.onnx (ONNX model)
- label_map.json (class map)

"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from collections import Counter, defaultdict
from typing import List, Set, Optional, Tuple, Dict
import re
import urllib.request
import os
import io
import sys
import logging
import json
import numpy as np
import cv2
from pathlib import Path
from threading import Thread
import time

# Imports for image processing
try:
    from PIL import Image, ImageTk, ImageGrab
    PILLOW_AVAILABLE = True
except Exception:
    PILLOW_AVAILABLE = False

# Imports for CNN
try:
    import onnxruntime as ort
    CNN_AVAILABLE = True
except Exception:
    CNN_AVAILABLE = False

# Imports for segmentation
try:
    from segment import segment_letters, get_tile_color_images
    SEGMENTATION_AVAILABLE = True
except Exception:
    SEGMENTATION_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TrieNode:
    """Node for the Trie structure for efficient prefix searching."""
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_word = False
        self.full_word: Optional[str] = None

# Equivalence maps between base vowels/consonants and accented variants.
# Used to allow unaccented letters to form words containing
# accented vowels (and vice versa).
ACCENT_EQUIV: Dict[str, List[str]] = {
    'A': ['A', 'Á', 'À', 'Â', 'Ã', 'Ä'],
    'E': ['E', 'É', 'È', 'Ê', 'Ë'],
    'I': ['I', 'Í', 'Ì', 'Î', 'Ï'],
    'O': ['O', 'Ó', 'Ò', 'Ô', 'Õ', 'Ö'],
    'U': ['U', 'Ú', 'Ù', 'Û', 'Ü'],
    'C': ['C', 'Ç'],
}

# Inverse map: variant -> base
VARIANT_TO_BASE: Dict[str, str] = {}
for base, variants in ACCENT_EQUIV.items():
    for v in variants:
        VARIANT_TO_BASE[v] = base


class LetterDetectorCNN:
    """Wrapper to load and use the ONNX letter detection model."""
    
    def __init__(self, model_path='letter_detector_final.onnx', label_map_path='label_map.json'):
        self.session = None
        self.label_map = {}
        self.img_size = 64
        self.input_name: Optional[str] = None
        
        try:
            # Load the ONNX model
            if Path(model_path).exists():
                self.session = ort.InferenceSession(model_path)
                # Get the input name from the model
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"ONNX model loaded: {model_path}")
                logger.info(f"Model input name: {self.input_name}")
            else:
                logger.warning(f"Model not found at: {model_path}")
                
            # Load label map
            if Path(label_map_path).exists():
                with open(label_map_path, 'r') as f:
                    self.label_map = json.load(f)
                logger.info(f"Label map loaded: {len(self.label_map)} classes")
            else:
                logger.warning(f"Label map not found at: {label_map_path}")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
    
    def is_ready(self) -> bool:
        """Checks if the model has been loaded correctly."""
        return self.session is not None and self.input_name is not None and len(self.label_map) > 0
    
    def predict_letter(self, tile_image: np.ndarray, confidence_threshold: float = 0.3) -> Optional[Tuple[str, float]]:
        """Predicts the letter of an individual tile.
        
        Args:
            tile_image: Tile image (BGR or grayscale)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple (letter, confidence) or None if below threshold
        """
        if not self.is_ready():
            return None
        
        try:
            # Convert to grayscale if necessary
            if len(tile_image.shape) == 3:
                gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = tile_image
            
            # Resize to model's expected size
            resized = cv2.resize(gray, (self.img_size, self.img_size))
            
            # Normalize (0-1)
            normalized = resized.astype('float32') / 255.0
            
            # Add batch and channel dimensions
            input_data = np.expand_dims(normalized, axis=(0, -1))  # (1, 64, 64, 1)
            
            # Prediction with ONNX Runtime
            # The input must be a dictionary where keys are input names
            result = self.session.run(None, {self.input_name: input_data})
            
            # The output is a list of numpy arrays
            prediction = result[0][0]
            confidence = float(np.max(prediction))
            class_idx = int(np.argmax(prediction))
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                logger.debug(f"Prediction below threshold: {confidence:.2%}")
                return None
            
            # Get corresponding letter
            reverse_map = {v: k for k, v in self.label_map.items()}
            if class_idx in reverse_map:
                letter = reverse_map[class_idx]
                logger.debug(f"Predicted letter: {letter} ({confidence:.2%})")
                return (letter, confidence)
            else:
                logger.warning(f"Class {class_idx} not found in label map")
                return None
        except Exception as e:
            logger.error(f"Error in ONNX prediction: {e}")
            return None


class WordsStreamSolver:
    """Anagram solver for Words on Stream (improved with Trie + Recursive Pruning)

    Strategy:
      - Loads dictionary in uppercase
      - Builds a Trie for efficient prefix searching
      - Uses recursive search with pruning: eliminates impossible branches based on available letters
      - Much faster than brute force, especially with large dictionaries
    """

    def __init__(self, dictionary_path: str | Path = "palavras_pt.txt"):
        self.dictionary_path = Path(dictionary_path)
        self.words: Set[str] = set()
        self.trie = TrieNode()  # Trie structure for fast searching
        self._load_dictionary()

    def _download_dictionary(self) -> Set[str]:
        """Tries to download public PT-BR dictionaries. Returns a set of words."""
        urls = [
            "https://www.ime.usp.br/~pf/dicios/br-utf8.txt",
            #"https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt",
        ]

        for url in urls:
            try:
                logger.info(f"Attempting to download dictionary: {url}")
                response = urllib.request.urlopen(url, timeout=10)
                content = response.read().decode('utf-8', errors='ignore')
                words = set(line.strip().upper() for line in content.splitlines() if line.strip())

                # save locally
                try:
                    self.dictionary_path.write_text('\n'.join(sorted(words)), encoding='utf-8')
                except Exception as e:
                    logger.warning(f"Could not save dictionary locally: {e}")

                return words
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
                continue

        return set()

    def _basic_dictionary(self) -> Set[str]:
        """Minimum embedded dictionary, in uppercase."""
        basic_words = """
        CASA GATO CACHORRO BOLA CAMA MESA CADEIRA LIVRO AGUA CAFE LEITE
        PRATO GARFO FACA COLHER PORTA JANELA CHAVE CARRO MOTO BIKE TREM
        AVIAO NAVIO BARCO PRAIA MAR SOL LUA ESTRELA CEU NUVEM CHUVA VENTO
        FOGO TERRA PEDRA AREIA GRAMA FLOR ROSA PLANTA ARVORE FOLHA GALHO
        RAIZ FRUTA MACA BANANA LARANJA UVA MELAO ABACAXI MANGA PERA LIMAO
        PESSOA GENTE HOMEM MULHER MENINO MENINA BEBE FILHO FILHA PAI MAE
        IRMAO IRMA AVO TIO TIA PRIMO AMIGO AMIGA NOME IDADE CORPO CABECA
        OLHO NARIZ BOCA ORELHA MANO BRACO PERNA DEDO UNHA PELE OSSO
        SANGUE CORACAO CEREBRO MENTE ALMA VIDA MORTE TEMPO DIA NOITE
        MANHA TARDE ANO MES SEMANA HORA MINUTO SEGUNDO HOJE AMANHA ONTEM
        ESCOLA SALA AULA PROFESSOR ALUNO PROVA NOTA LIVRO CADERNO LAPIS
        CANETA BORRACHA REGUA MOCHILA TRABALHO EMPREGO CHEFE SALARIO
        DINHEIRO REAL MOEDA BANCO LOJA MERCADO COMPRA VENDA PRECO BARATO
        CARO COMIDA BEBIDA PRATO LANCHE JANTAR ALMOCO CAFE MANHA COZINHA
        GELADEIRA FOGAO FORNO MICROONDAS PANELA TIGELA COPO GARRAFA
        CIDADE BAIRRO RUA AVENIDA PRACA PARQUE CASA PREDIO APARTAMENTO
        QUARTO SALA BANHEIRO COZINHA QUINTAL JARDIM GARAGEM TELHADO MURO
        PAIS BRASIL ESTADO REGIAO CAPITAL INTERIOR CAMPO FAZENDA SITIO
        ROCA PLANTACAO COLHEITA GADO VACA BOI PORCO GALINHA PATO GANSO
        CAVALO BURRO CABRA OVELHA CACHORRO GATO PASSARO PEIXE RATO COBRA
        VERDE AZUL VERMELHO AMARELO BRANCO PRETO CINZA ROSA ROXO LARANJA
        MARROM CLARO ESCURO GRANDE PEQUENO ALTO BAIXO GORDO MAGRO BONITO
        FEIO BOM RUIM CERTO ERRADO FELIZ TRISTE ALEGRE BRAVO CALMO
        NUMERO ZERO DOIS TRES QUATRO CINCO SEIS SETE OITO NOVE
        CONTA SOMA MENOS VEZES DIVISAO IGUAL MAIOR MENOR MUITO POUCO
        NADA TUDO ALGO ALGUM NENHUM CADA OUTRO MESMO PROPRIO JUNTO
        LONGE PERTO DENTRO FORA CIMA BAIXO FRENTE TRAS LADO MEIO CENTRO
        ACHAR FAZER DIZER FALAR OUVIR COMER BEBER DORMIR ACORDAR ANDAR
        CORRER PULAR SENTAR LEVANTAR DEITAR OLHAR PENSAR SABER QUERER
        PODER DEVER GOSTAR AMAR ODIAR TEMER ESPERAR COMECAR ACABAR ABRIR
        FECHAR USAR PEGAR SOLTAR JOGAR GANHAR PERDER SUBIR DESCER ENTRAR
        SAIR CHEGAR PARTIR VOLTAR FICAR MUDAR MOVER PARAR SEGUIR BUSCAR
        BANCO BANHO CABANA CHANA BANHA CANGA CHAGA GANA BANCA GANCHO
        """.split()
        return set(basic_words)

    def _load_dictionary(self):
        """Loads and builds the Trie from the local dictionary (or downloads one).

        Finally, we have:
          - self.words (set)
          - self.trie (Trie structure for efficient searching)
        """
        words = set()

        if self.dictionary_path.exists():
            try:
                text = self.dictionary_path.read_text(encoding='utf-8')
                words = set(l.strip().upper() for l in text.splitlines() if l.strip())
                logger.info(f"Loaded local dictionary: {self.dictionary_path} ({len(words)} words)")
            except Exception as e:
                logger.warning(f"Error reading local dictionary: {e}")

        if not words:
            words = self._download_dictionary()

        if not words:
            logger.info("Using embedded basic dictionary")
            words = self._basic_dictionary()

        # normalize
        words = set(p for p in words if re.match(r'^[A-ZÀ-Ÿ]+$', p))
        self.words = words
        
        # Builds the Trie
        self.trie = TrieNode()
        for word in words:
            self._insert_into_trie(word)

    def normalize_input(self, input_str: str) -> List[str]:
        # now accepts '?' as a hidden letter token
        input_str = input_str.upper().strip()
        letters = re.findall(r'[A-ZÁÀÂÃÉÊÍÓÔÕÚÇÜ\?]', input_str)
        return letters

    def _insert_into_trie(self, word: str):
        """Inserts a word into the Trie."""
        node = self.trie
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.is_word = True
        node.full_word = word

    def _equivalent_letters(self, letter: str) -> List[str]:
        """Returns the list of equivalent letters for 'letter'.

        Ex: 'A' -> ['A','Á','À',...], 'Á' -> ['A','Á','À',...], 'B' -> ['B']
        """
        letter = letter.upper()
        # If it's a mapped variant, get the base and then all variants
        base = VARIANT_TO_BASE.get(letter, letter)
        return ACCENT_EQUIV.get(base, [letter])

    def can_form_word(self, word_counter: Counter, available_letters: Counter) -> bool:
        """Checks if word_counter can be formed by available_letters (kept for compatibility)."""
        for letter, qty in word_counter.items():
            if available_letters[letter] < qty:
                return False
        return True

    def find_anagrams(self, letters: List[str], min_length: int = 4) -> List[str]:
        """
        Finds anagrams using Trie + Recursive Pruning.
        
        The recursive search explores the Trie only through paths that can be formed
        with the available letters, pruning impossible branches early.
        """
        available_letters = Counter(letters)
        results: List[str] = []
        
        # Recursive search with pruning
        self._recursive_search(
            node=self.trie,
            remaining_letters=available_letters,
            min_length=min_length,
            results=results
        )
        
        # Sort by decreasing length and lexicographically
        results.sort(key=lambda p: (-len(p), p))
        return results

    def _recursive_search(
        self,
        node: TrieNode,
        remaining_letters: Counter,
        min_length: int,
        results: List[str]
    ):
        """
        Recursive search with pruning in the Trie.
        
        Strategy:
        - If we reach a valid word and length >= min_length, add to results
        - For each child letter, if we have that letter available, descend recursively
        - Automatically prunes branches where we don't have the necessary letters
        """
        
        # Base: if this node marks end of word and has minimum length
        if node.is_word and len(node.full_word) >= min_length:
            results.append(node.full_word)
        
        # Recursion: explore children only if we have the letters.
        # To support equivalence between accented and unaccented vowels,
        # we iterate through available letters and look for equivalent children.
        for available_letter, qty in list(remaining_letters.items()):
            if qty <= 0:
                continue
            # get equivalent letters that may exist in the Trie
            candidates = self._equivalent_letters(available_letter)
            for cand in candidates:
                child_node = node.children.get(cand)
                if not child_node:
                    continue
                # Use the available letter (count refers to user input)
                remaining_letters[available_letter] -= 1
                # Recurse down the corresponding node
                self._recursive_search(
                    node=child_node,
                    remaining_letters=remaining_letters,
                    min_length=min_length,
                    results=results
                )
                # Restore (backtrack)
                remaining_letters[available_letter] += 1

    # ================= NOVA FUNÇÃO =================
    def find_anagrams_with_wildcard(self, letters: List[str], min_length: int = 4):
        """
        When '?' exists in letters, tests all A-Z letters replacing the wildcard,
        generates a map of possible words per letter and calculates relative probabilities.

        Returns:
            confirmed: List[str] -> words found without using '?'
            letter_word_map: Dict[str, List[str]] -> for each tested letter, list of words (excluding confirmed ones)
            probs: Dict[str, float] -> relative probability (len(list)/total)
        """
        if '?' not in letters:
            # no wildcard, normal behavior
            return self.find_anagrams(letters, min_length), {}, {}

        # separate fixed letters (without '?')
        fixed_letters = [c for c in letters if c != '?']

        # words that already exist without needing the wildcard
        confirmed = self.find_anagrams(fixed_letters, min_length)

        # map letter -> list of words generated with this substitution (excluding confirmed ones)
        letter_word_map: Dict[str, List[str]] = {}
        for code in range(ord('A'), ord('Z') + 1):
            test_letter = chr(code)
            combination = fixed_letters + [test_letter]
            words = self.find_anagrams(combination, min_length)
            # keep only those not in confirmed
            new_words = [p for p in words if p not in confirmed]
            letter_word_map[test_letter] = new_words

        # calculate probabilities (proportional to the number of new words per letter)
        total = sum(len(v) for v in letter_word_map.values())
        if total == 0:
            probs = {k: 0.0 for k in letter_word_map.keys()}
        else:
            probs = {k: len(v) / total for k, v in letter_word_map.items()}

        return confirmed, letter_word_map, probs
    
    
    def detect_false_letter(self, letters: List[str], min_length: int = 4):
        """Tests each letter as false by removing it and calculating possible words."""
        results = {}
        unique_letters = list(letters)

        for letter in unique_letters:
            letters_without = [l for l in unique_letters if l != letter]
            if '?' in letters_without:
                confirmed, wildcard_map, _ = self.find_anagrams_with_wildcard(letters_without, min_length)
                words = set(confirmed)
                for v in wildcard_map.values():
                    words.update(v)
                results[letter] = list(words)
            else:
                results[letter] = self.find_anagrams(letters_without, min_length)

        # Calculate inverse probability: the MORE words → the more false
        counts = {letter: len(lst) for letter, lst in results.items()}
        total = sum(counts.values()) or 1  # avoid division by zero
        prob = {letter: counts[letter] / total for letter in results.keys()}

        return results, prob



class ImageCropDialog(tk.Toplevel):
    """Dialog to select crop area in the image (same as original, with minor improvements)"""

    def __init__(self, parent, image: 'Image.Image'):
        super().__init__(parent)
        self.title("Select Area with Letters")
        self.original_image = image
        self.cropped_image: Optional['Image.Image'] = None
        self.transient(parent)
        self.grab_set()

        # Configurations
        self.rect = None
        self.start_x = None
        self.start_y = None

        # Resize image to fit screen
        max_width, max_height = 900, 700
        self.scale = min(max_width / image.width, max_height / image.height, 1.0)

        new_size = (int(image.width * self.scale), int(image.height * self.scale))
        self.display_image = image.resize(new_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)

        # Canvas for drawing
        self.canvas = tk.Canvas(self, width=new_size[0], height=new_size[1], cursor="cross")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Instructions
        frame_info = ttk.Frame(self)
        frame_info.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame_info, text="🖱️ Drag to select the area with letters", font=('Arial', 10)).pack(side=tk.LEFT, padx=5)

        # Buttons
        frame_btns = ttk.Frame(self)
        frame_btns.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(frame_btns, text="✓ Confirm", command=self.confirm).pack(side=tk.RIGHT, padx=5)
        ttk.Button(frame_btns, text="✗ Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(frame_btns, text="🔄 Use Entire Image", command=self.use_entire_image).pack(side=tk.RIGHT, padx=5)

        # Mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Center
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        y = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='#00ff00', width=3, dash=(5, 5))

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        pass

    def use_entire_image(self):
        self.cropped_image = self.original_image
        self.destroy()

    def confirm(self):
        if not self.rect:
            messagebox.showwarning("Warning", "Select an area first or use 'Use Entire Image'")
            return

        coords = self.canvas.coords(self.rect)
        x1, y1, x2, y2 = [int(c / self.scale) for c in coords]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            messagebox.showwarning("Warning", "Area too small! Select a larger area.")
            return

        self.cropped_image = self.original_image.crop((x1, y1, x2, y2))
        self.destroy()

    def cancel(self):
        self.cropped_image = None
        self.destroy()


class WordsStreamGUI:
    """Main graphical interface with usability improvements."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Words on Stream - Anagram Generator")
        self.root.geometry("980x720")

        self.solver = WordsStreamSolver()
        self.cnn_detector = LetterDetectorCNN()
        self.current_image = None
        self._processing = False

        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        self._create_interface()

        # Bindings
        self.root.bind('<Control-v>', self.paste_image)
        self.root.bind('<Control-V>', self.paste_image)

    def _create_interface(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="🎮 WORDS ON STREAM", font=('Arial', 20, 'bold')).pack()
        ttk.Label(title_frame, text="Anagram Generator with CNN", font=('Arial', 10)).pack()

        input_frame = ttk.LabelFrame(main_frame, text="📥 Input", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="📋 Paste Image (Ctrl+V)", command=self.paste_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📂 Open File", command=self.open_file).pack(side=tk.LEFT, padx=5)

        # CNN detector status
        cnn_status = "✓ CNN Ready" if self.cnn_detector.is_ready() else "⚠️ CNN not available"
        cnn_color = 'green' if self.cnn_detector.is_ready() else 'orange'
        ttk.Label(btn_frame, text=cnn_status, foreground=cnn_color).pack(side=tk.LEFT, padx=10)

        text_frame = ttk.Frame(input_frame)
        text_frame.pack(fill=tk.X)

        ttk.Label(text_frame, text="Or type the letters:").pack(side=tk.LEFT, padx=5)
        self.entry_letters = ttk.Entry(text_frame, font=('Arial', 12))
        self.entry_letters.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entry_letters.bind('<Return>', lambda e: self.solve_async())

        self.var_false_letter = tk.BooleanVar()
        ttk.Checkbutton(text_frame, text="Has false letter", variable=self.var_false_letter).pack(side=tk.LEFT, padx=10)


        ttk.Label(text_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        self.spin_min = ttk.Spinbox(text_frame, from_=2, to=12, width=5)
        self.spin_min.set(4)
        self.spin_min.pack(side=tk.LEFT, padx=5)

        ttk.Button(text_frame, text="🔍 Solve", command=self.solve_async).pack(side=tk.LEFT, padx=5)

        self.label_letters = ttk.Label(input_frame, text="", font=('Arial', 11, 'bold'))
        self.label_letters.pack(pady=5)

        result_frame = ttk.LabelFrame(main_frame, text="📊 Results", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.text_results = scrolledtext.ScrolledText(result_frame, font=('Courier', 10), wrap=tk.WORD, state='disabled')
        self.text_results.pack(fill=tk.BOTH, expand=True)

        # tags
        self.text_results.tag_config('title', font=('Arial', 12, 'bold'))
        self.text_results.tag_config('subtitle', font=('Arial', 10, 'bold'))
        self.text_results.tag_config('highlight', foreground='#0066cc', font=('Arial', 11, 'bold'))
        # click on word copies to clipboard
        self.text_results.bind('<Button-1>', self._on_result_click)
        # temporary tag to highlight clicked word
        self.text_results.tag_config('clicked', background='#fff2b2')

        # quick actions
        actions = ttk.Frame(main_frame)
        actions.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(actions, text="📋 Copy Results", command=self.copy_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="💾 Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="🧹 Clear", command=self.clear).pack(side=tk.LEFT, padx=5)

        self.status_bar = ttk.Label(main_frame, text="Ready! Paste an image (Ctrl+V) or type the letters.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))

    # ---------- UI utilities ----------
    def _set_processing(self, value: bool, msg: Optional[str] = None):
        self._processing = value
        state = 'disabled' if value else 'normal'
        # Disable main controls
        try:
            for widget in (self.entry_letters,):
                widget.config(state=state)
        except Exception:
            pass
        if msg:
            self.status_bar.config(text=msg)

    # ---------- image input ----------
    def paste_image(self, event=None):
        # If focus is on the text field, the widget's default paste action has already occurred.
        # We need to avoid double execution.
        if self.root.focus_get() == self.entry_letters:
            try:
                # We only check if there is text. If there is, the default paste has already happened.
                self.root.clipboard_get()
                # If the code reached here, there is text in the clipboard. Default paste has already been done.
                # So, we simply stop execution to avoid double pasting or image processing.
                return "break"
            except tk.TclError:
                # No text in clipboard. Default paste failed.
                # We should proceed to try pasting as an image.
                pass

        # --- Logic to paste image (executes if focus is not on the entry, or if clipboard has no text) ---

        if not self.cnn_detector.is_ready():
            messagebox.showerror("Error", "CNN not available!\nCheck if 'letter_detector_final.onnx' model and 'label_map.json' are present.")

        if not PILLOW_AVAILABLE:
            messagebox.showerror("Error", "Pillow not available!\nInstall: pip install pillow")
            return "break"

        try:
            image = ImageGrab.grabclipboard()
            if image is None:
                messagebox.showinfo("Warning", "No image in clipboard! Copy an image and try again.")
                return "break"
            
            if not isinstance(image, Image.Image):
                try:
                    if isinstance(image, (bytes, bytearray)):
                        image = Image.open(io.BytesIO(image))
                    else:
                        messagebox.showinfo("Warning", "Clipboard content is not an image.")
                        return "break"
                except Exception:
                    messagebox.showerror("Error", "Clipboard content is not a valid image.")
                    return "break"

            self._process_image_async(image)
        except Exception as e:
            messagebox.showerror("Error", f"Error pasting image:\n{e}")
        
        return "break"

    def open_file(self):
        if not self.cnn_detector.is_ready():
            messagebox.showerror("Error", "CNN not available!\nCheck if 'letter_detector_final.h5' model and 'label_map.json' are present.")
            return

        if not PILLOW_AVAILABLE:
            messagebox.showerror("Error", "Pillow not available!\nInstall: pip install pillow")
            return

        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")])
        if not file_path:
            return
        try:
            image = Image.open(file_path)
            self._process_image_async(image)
        except Exception as e:
            messagebox.showerror("Error", f"Error opening file:\n{e}")

    # ---------- threading to avoid UI freezing ----------
    def _process_image_async(self, image):
        if self._processing:
            return
        self._set_processing(True, "Processing image...")
        # Create dialog in main thread
        dialog = ImageCropDialog(self.root, image)
        self.root.wait_window(dialog)
        if dialog.cropped_image is None:
            self.status_bar.config(text="Operation cancelled.")
            self._set_processing(False)
            return
        # Process image in separate thread
        Thread(target=self._process_image_thread, args=(dialog.cropped_image,), daemon=True).start()

    def _process_image_thread(self, cropped_image):
        try:
            letters = self._process_image_cnn(cropped_image)
            logger.info(f"DEBUG: Detected letters: {letters}")
            if letters:
                # insert letters and solve (in main thread)
                letters_str = ''.join(letters)
                len_letters = len(letters)
                logger.info(f"DEBUG: letters_str = '{letters_str}', len = {len_letters}")
                
                def insert_and_solve():
                    try:
                        logger.info(f"DEBUG: Inside insert_and_solve, attempting to insert '{letters_str}'")
                        # Ensure the field is enabled
                        self.entry_letters.config(state='normal')
                        self.entry_letters.delete(0, tk.END)
                        logger.info(f"DEBUG: Delete executed")
                        self.entry_letters.insert(0, letters_str)
                        logger.info(f"DEBUG: Insert executed successfully")
                        self.status_bar.config(text=f"✓ {len_letters} letters detected by CNN!")
                        logger.info(f"DEBUG: Status bar updated")
                        self.solve_async()
                        logger.info(f"DEBUG: solve_async called")
                    except Exception as e:
                        logger.error(f"DEBUG: Error in insert_and_solve: {e}", exc_info=True)
                
                logger.info(f"DEBUG: Scheduling insert_and_solve with root.after(0, ...)")
                self.root.after(0, insert_and_solve)
            else:
                logger.warning("DEBUG: No letters detected")
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No letters detected! Check the image and try again."))
                self.root.after(0, lambda: self.status_bar.config(text="No letters detected in the image."))
        except Exception as e:
            logger.exception("Error in image processing")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing image:\n{e}"))
        finally:
            self.root.after(0, lambda: self._set_processing(False))

    def _process_image_cnn(self, pil_image) -> Optional[List[str]]:
        """Processes image with segmentation and CNN to detect letters.
        
        Flow:
        1. Convert PIL image to temporary file
        2. Segment using segmentar.py
        3. Extract tiles
        4. Predict each tile with CNN
        5. Return list of detected letters (including '?' when CNN has low confidence)
        """
        if not self.cnn_detector.is_ready():
            return None
        
        if not SEGMENTATION_AVAILABLE:
            logger.error("Segmentation module not available")
            return None
        
        try:
            # Save PIL image to temporary file
            temp_path = "temp_image_processing.png"
            pil_image.save(temp_path)
            
            logger.info(f"Segmenting image: {temp_path}")
            
            # Segment image
            segmented_letters, original_img = segment_letters(temp_path, debug=False)
            
            if not segmented_letters:
                logger.warning("No letters segmented")
                return None
            
            logger.info(f"Found {len(segmented_letters)} letter regions")
            
            # Extract tiles in memory
            tile_images = get_tile_color_images(segmented_letters)
            
            if not tile_images:
                logger.warning("No tile images extracted")
                return None
            
            logger.info(f"Extracted {len(tile_images)} tiles")
            
            # Predict each tile
            detected_letters = []
            for tile_name, tile_image in tile_images.items():
                try:
                    result = self.cnn_detector.predict_letter(tile_image, confidence_threshold=0.3)
                    if result:
                        letter, confidence = result
                        detected_letters.append(letter)
                        logger.info(f"Tile {tile_name}: {letter} ({confidence:.2%})")
                    else:
                        # When CNN does not return (low confidence), treat as wildcard '?'
                        detected_letters.append('?')
                        logger.info(f"Tile {tile_name}: marked as '?' (hidden letter detected by CNN or low confidence)")
                except Exception as e:
                    logger.error(f"Error processing tile {tile_name}: {e}")
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except Exception:
                pass
            
            return detected_letters if detected_letters else None
        except Exception as e:
            logger.exception("Error in CNN processing")
            # Try to clean up temporary file in case of error
            try:
                if os.path.exists("temp_image_processing.png"):
                    os.remove("temp_image_processing.png")
            except Exception:
                pass
            return None

    # ---------- solve (threaded) ----------
    def solve_async(self):
        if self._processing:
            return
        input_str = self.entry_letters.get().strip()
        if not input_str:
            messagebox.showwarning("Warning", "Enter letters or paste an image!")
            return
        try:
            min_length = int(self.spin_min.get())
        except Exception:
            min_length = 4
        letters = self.solver.normalize_input(input_str)
        if not letters:
            messagebox.showerror("Error", "No valid letters found!")
            return

        self._set_processing(True, "Searching for words...")
        Thread(target=self._solve_thread, args=(letters, min_length), daemon=True).start()

    def _solve_thread(self, letters: List[str], min_length: int):
        try:
            # If there is a false letter
            if self.var_false_letter.get():
                false_results, false_probs = self.solver.detect_false_letter(letters, min_length)
                self.root.after(0, lambda:
                    self.display_false_letter_results(letters, false_results, false_probs)
                )
                return

            # If there is a wildcard '?'
            if '?' in letters:
                confirmed, letter_word_map, probs = self.solver.find_anagrams_with_wildcard(letters, min_length)
                by_length_conf = defaultdict(list)
                for p in confirmed:
                    by_length_conf[len(p)].append(p)
                self.root.after(0, lambda:
                    self.display_wildcard_results(letters, confirmed, letter_word_map, probs, by_length_conf)
                )
                return

            # Normal
            words = self.solver.find_anagrams(letters, min_length)
            by_length = defaultdict(list)
            for p in words:
                by_length[len(p)].append(p)
            self.root.after(0, lambda: self.display_results(letters, words, by_length))

        except Exception as e:
            logger.exception("Error solving anagrams")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error searching for words:\n{e}"))
        finally:
            self.root.after(0, lambda: self._set_processing(False))

    # ---------- display and actions ----------
    def display_results(self, letters, words, by_length):
        self.text_results.config(state='normal')
        self.text_results.delete(1.0, tk.END)

        if not words:
            self.text_results.insert(tk.END, "❌ No words found with these letters.\n\n")
            self.text_results.insert(tk.END, f"Available letters: {' '.join(letters)}\n")
            self.text_results.insert(tk.END, f"Minimum length: {self.spin_min.get()}\n")
            self.text_results.config(state='disabled')
            return

        self.text_results.insert(tk.END, "="*80 + "\n", 'title')
        self.text_results.insert(tk.END, f"  TOTAL: {len(words)} WORDS FOUND\n", 'title')
        self.text_results.insert(tk.END, "="*80 + "\n\n", 'title')

        for length in sorted(by_length.keys(), reverse=True):
            words_by_length = by_length[length]
            self.text_results.insert(tk.END, f"📝 {length} LETTERS ({len(words_by_length)} words)\n", 'subtitle')
            self.text_results.insert(tk.END, "-" * 80 + "\n")
            columns = 5
            for i in range(0, len(words_by_length), columns):
                line = words_by_length[i:i+columns]
                line_text = "   " + "  ".join(f"{p.lower():<15}" for p in line) + "\n"
                self.text_results.insert(tk.END, line_text)
            self.text_results.insert(tk.END, "\n")

        self.text_results.config(state='disabled')

    def display_wildcard_results(self, letters, confirmed, letter_word_map, probs, by_length_conf):
        """
        Displays:
         - confirmed words (without '?')
         - possible words grouped by tested letter, with relative probability
        """
        self.text_results.config(state='normal')
        self.text_results.delete(1.0, tk.END)

        # Header
        total_possible = sum(len(v) for v in letter_word_map.values())
        self.text_results.insert(tk.END, "="*80 + "\n", 'title')
        self.text_results.insert(tk.END, f"  TOTAL: {len(confirmed)} CONFIRMED WORDS, {total_possible} HYPOTHESES (testing A-Z)\n", 'title')
        self.text_results.insert(tk.END, "="*80 + "\n\n", 'title')

        # Confirmed
        self.text_results.insert(tk.END, "✅ CONFIRMED WORDS (do not depend on '?'):\n", 'subtitle')
        if not confirmed:
            self.text_results.insert(tk.END, " (none)\n\n")
        else:
            for length in sorted(by_length_conf.keys(), reverse=True):
                words_by_length = by_length_conf[length]
                self.text_results.insert(tk.END, f"📝 {length} LETTERS ({len(words_by_length)} words)\n")
                self.text_results.insert(tk.END, "-" * 80 + "\n")
                columns = 5
                for i in range(0, len(words_by_length), columns):
                    line = words_by_length[i:i+columns]
                    line_text = "   " + "  ".join(f"{p.lower():<15}" for p in line) + "\n"
                    self.text_results.insert(tk.END, line_text)
                self.text_results.insert(tk.END, "\n")

        # Hypotheses by letter (sort by decreasing probability)
        self.text_results.insert(tk.END, "\n🔮 POSSIBLE WORDS (depend on the hidden letter '?'):\n", 'subtitle')
        # Sort letters by decreasing probability
        ordered_letters = sorted(probs.items(), key=lambda x: (-x[1], x[0]))
        any_shown = False
        for letter, prob in ordered_letters:
            word_list = letter_word_map.get(letter, [])
            if not word_list:
                continue
            any_shown = True
            pct = prob * 100
            self.text_results.insert(tk.END, f"\n🔸 Letter {letter} — {pct:.1f}% ({len(word_list)} words)\n", 'highlight')
            self.text_results.insert(tk.END, "-" * 60 + "\n")
            # Show up to N first words per letter (avoid excess)
            max_show = 80
            to_show = word_list[:max_show]
            columns = 5
            for i in range(0, len(to_show), columns):
                line = to_show[i:i+columns]
                line_text = "   " + "  ".join(f"{p.lower():<15}" for p in line) + "\n"
                self.text_results.insert(tk.END, line_text)
            if len(word_list) > max_show:
                self.text_results.insert(tk.END, f"   ... and {len(word_list)-max_show} more words\n")
        if not any_shown:
            self.text_results.insert(tk.END, " (no hypothesis generated words)\n")

        self.text_results.config(state='disabled')

    def display_false_letter_results(self, letters, false_results, probs):
        self.text_results.config(state='normal')
        self.text_results.delete(1.0, tk.END)

        self.text_results.insert(tk.END, "⚠️ FALSE LETTER ANALYSIS\n", 'title')

        # Sort letters by probability (descending)
        ordered = sorted(probs.items(), key=lambda x: -x[1])

        for letter, prob in ordered:
            word_list = false_results[letter]
            pct = prob * 100
            self.text_results.insert(tk.END, f"\n🔸 Letter '{letter}' — {pct:.1f}% chance of being false\n", 'highlight')
            self.text_results.insert(tk.END, f"Possible words removing '{letter}': {len(word_list)}\n")
            if len(word_list) > 0:
                preview = ", ".join(word_list[:20])
                self.text_results.insert(tk.END, f"Ex: {preview}\n")

        self.text_results.config(state='disabled')


    def copy_results(self):
        try:
            text = self.text_results.get(1.0, tk.END).strip()
            if not text:
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_bar.config(text="Copied to clipboard.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not copy:\n{e}")

    def _on_result_click(self, event):
        """Handler when the user clicks on a word in the results.

        Copies the clicked word to the clipboard and briefly highlights it.
        """
        try:
            # get text index under cursor
            idx = self.text_results.index(f"@{event.x},{event.y}")
            # get word (wordstart/wordend considers punctuation)
            word = self.text_results.get(f"{idx} wordstart", f"{idx} wordend").strip()
            if not word:
                return
            # extract only letters (including accents) — protects against headers and symbols
            m = re.findall(r"[A-ZÁÀÂÃÉÊÍÓÔÕÚÇÜ]+", word.upper())
            if not m:
                return
            clean_word = m[0].lower()

            # copy to clipboard
            self.root.clipboard_clear()
            self.root.clipboard_append(clean_word)
            self.status_bar.config(text=f"Copied: {clean_word}")

            # highlight word briefly
            # we temporarily need to allow editing to add tag
            prev_state = self.text_results.cget('state')
            try:
                self.text_results.config(state='normal')
                start = self.text_results.index(f"{idx} wordstart")
                end = self.text_results.index(f"{idx} wordend")
                self.text_results.tag_add('clicked', start, end)
            finally:
                # restore disabled state (will be re-enabled after tag removal)
                self.text_results.config(state=prev_state)

            # remove highlight after 800ms
            def _clear_tag():
                try:
                    self.text_results.config(state='normal')
                    self.text_results.tag_remove('clicked', start, end)
                    self.text_results.config(state=prev_state)
                except Exception:
                    pass

            self.text_results.after(800, _clear_tag)

        except Exception:
            # do not fail UI for an invalid click
            return

    def save_results(self):
        try:
            text = self.text_results.get(1.0, tk.END).strip()
            if not text:
                messagebox.showinfo("Info", "No results to save.")
                return
            file_path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text', '*.txt')])
            if not file_path:
                return
            Path(file_path).write_text(text, encoding='utf-8')
            self.status_bar.config(text=f"Saved to: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving:\n{e}")

    def clear(self):
        self.entry_letters.delete(0, tk.END)
        self.text_results.config(state='normal')
        self.text_results.delete(1.0, tk.END)
        self.text_results.config(state='disabled')
        self.label_letters.config(text='')
        self.status_bar.config(text="Cleared. Paste an image (Ctrl+V) or type the letters.")


def main():
    root = tk.Tk()
    app = WordsStreamGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
