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
- letter_detector_model.onnx (ONNX model)
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
    
    def __init__(self, model_path=Path(__file__).parent / '../model/letter_detector_model.onnx', label_map_path=Path(__file__).parent / '../model/label_map.json'):
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

    def __init__(self):
        self.dictionary_path: Optional[Path] = None
        self.words: Set[str] = set()
        self.trie = TrieNode()

    def set_dictionary_language(self, lang: str = 'pt'):
        """Sets the dictionary language and reloads the dictionary."""
        self._load_dictionary(lang)

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

    def _load_dictionary(self, lang: str = 'pt'):
        """Loads and builds the Trie from the dictionary for the specified language."""
        self.dictionary_path = Path(__file__).parent.parent / "data" / "words" / f"words_{lang}.txt"
        words = set()

        if self.dictionary_path.exists():
            try:
                text = self.dictionary_path.read_text(encoding='utf-8')
                words = set(l.strip().upper() for l in text.splitlines() if l.strip())
                logger.info(f"Loaded local dictionary: {self.dictionary_path} ({len(words)} words)")
            except Exception as e:
                logger.warning(f"Error reading local dictionary: {e}")

        if not words and lang == 'pt':
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

    def __init__(self, parent, image: 'Image.Image', get_string_func):
        super().__init__(parent)
        self.get_string = get_string_func
        self.title(self.get_string("crop_dialog_title"))
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

        ttk.Label(frame_info, text=self.get_string("crop_dialog_instruction"), font=('Arial', 10)).pack(side=tk.LEFT, padx=5)

        # Buttons
        frame_btns = ttk.Frame(self)
        frame_btns.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(frame_btns, text=self.get_string("crop_dialog_confirm"), command=self.confirm).pack(side=tk.RIGHT, padx=5)
        ttk.Button(frame_btns, text=self.get_string("crop_dialog_cancel"), command=self.cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(frame_btns, text=self.get_string("crop_dialog_use_entire"), command=self.use_entire_image).pack(side=tk.RIGHT, padx=5)

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
            messagebox.showwarning(self.get_string("warning_title"), self.get_string("select_area_prompt"))
            return

        coords = self.canvas.coords(self.rect)
        x1, y1, x2, y2 = [int(c / self.scale) for c in coords]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            messagebox.showwarning(self.get_string("warning_title"), self.get_string("area_too_small"))
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
        self.root.geometry("980x720")

        self.solver = WordsStreamSolver()
        self.cnn_detector = LetterDetectorCNN()
        self.current_image = None
        self._processing = False

        # i18n attributes
        self.translations = {}
        self.current_lang = tk.StringVar(value='en')
        self.dictionary_lang = tk.StringVar(value='pt')
        self.dictionary_lang.trace_add('write', self._on_dict_lang_change)
        self._load_translations()

        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        self._create_interface()
        self.current_lang.trace_add('write', self._on_lang_change)

        # Bindings
        self.root.bind('<Control-v>', self.paste_image)
        self.root.bind('<Control-V>', self.paste_image)

        # Load initial dictionary
        self._on_dict_lang_change()

    def _create_interface(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top frame with language selector ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        self.title_label = ttk.Label(top_frame, font=('Arial', 20, 'bold'))
        self.title_label.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Dictionary Language
        dict_lang_options = ['pt', 'en']
        self.dict_lang_menu = ttk.OptionMenu(top_frame, self.dictionary_lang, self.dictionary_lang.get(), *dict_lang_options)
        self.dict_lang_menu.pack(side=tk.RIGHT, padx=(10, 0))
        self.dict_lang_label = ttk.Label(top_frame, text="Dictionary:")
        self.dict_lang_label.pack(side=tk.RIGHT)

        # UI Language
        lang_options = list(self.translations.keys()) if self.translations else ['en', 'pt']
        self.lang_menu = ttk.OptionMenu(top_frame, self.current_lang, self.current_lang.get(), *lang_options)
        self.lang_menu.pack(side=tk.RIGHT, padx=(10, 0))
        self.lang_label = ttk.Label(top_frame, text="Language:")
        self.lang_label.pack(side=tk.RIGHT)

        self.subtitle_label = ttk.Label(main_frame, font=('Arial', 10))
        self.subtitle_label.pack(fill=tk.X)

        self.input_frame = ttk.LabelFrame(main_frame, padding=10)
        self.input_frame.pack(fill=tk.X, pady=(10, 10))

        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.paste_button = ttk.Button(btn_frame, command=self.paste_image)
        self.paste_button.pack(side=tk.LEFT, padx=5)
        self.open_button = ttk.Button(btn_frame, command=self.open_file)
        self.open_button.pack(side=tk.LEFT, padx=5)

        # CNN detector status
        cnn_color = 'green' if self.cnn_detector.is_ready() else 'orange'
        self.cnn_status_label = ttk.Label(btn_frame, foreground=cnn_color)
        self.cnn_status_label.pack(side=tk.LEFT, padx=10)

        text_frame = ttk.Frame(self.input_frame)
        text_frame.pack(fill=tk.X)

        self.type_letters_label = ttk.Label(text_frame)
        self.type_letters_label.pack(side=tk.LEFT, padx=5)
        self.entry_letters = ttk.Entry(text_frame, font=('Arial', 12))
        self.entry_letters.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entry_letters.bind('<Return>', lambda e: self.solve_async())

        self.var_false_letter = tk.BooleanVar()
        self.false_letter_check = ttk.Checkbutton(text_frame, variable=self.var_false_letter)
        self.false_letter_check.pack(side=tk.LEFT, padx=10)

        self.min_length_label = ttk.Label(text_frame)
        self.min_length_label.pack(side=tk.LEFT, padx=5)
        self.spin_min = ttk.Spinbox(text_frame, from_=2, to=12, width=5)
        self.spin_min.set(4)
        self.spin_min.pack(side=tk.LEFT, padx=5)

        self.solve_button = ttk.Button(text_frame, command=self.solve_async)
        self.solve_button.pack(side=tk.LEFT, padx=5)

        self.label_letters = ttk.Label(self.input_frame, text="", font=('Arial', 11, 'bold'))
        self.label_letters.pack(pady=5)

        self.result_frame = ttk.LabelFrame(main_frame, padding=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        self.text_results = scrolledtext.ScrolledText(self.result_frame, font=('Courier', 10), wrap=tk.WORD, state='disabled')
        self.text_results.pack(fill=tk.BOTH, expand=True)

        # tags
        self.text_results.tag_config('title', font=('Arial', 12, 'bold'))
        self.text_results.tag_config('subtitle', font=('Arial', 10, 'bold'))
        self.text_results.tag_config('highlight', foreground='#0066cc', font=('Arial', 11, 'bold'))
        self.text_results.bind('<Button-1>', self._on_result_click)
        self.text_results.tag_config('clicked', background='#fff2b2')

        # quick actions
        actions = ttk.Frame(main_frame)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.copy_button = ttk.Button(actions, command=self.copy_results)
        self.copy_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(actions, command=self.save_results)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = ttk.Button(actions, command=self.clear)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.status_bar = ttk.Label(main_frame, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))
        self._update_ui_text() # Initial UI text setup
        self.current_lang.trace_add('write', self._on_lang_change) # Added here



    # ---------- UI utilities ----------
    def _set_processing(self, value: bool, msg: Optional[str] = None):
        self._processing = value
        state = 'disabled' if value else 'normal'
        try:
            for widget in (self.entry_letters,):
                widget.config(state=state)
        except Exception:
            pass
        if msg:
            self.status_bar.config(text=msg)

    # ---------- image input ----------
    def paste_image(self, event=None):
        if self.root.focus_get() == self.entry_letters:
            try:
                self.root.clipboard_get()
                return "break"
            except tk.TclError:
                pass

        if not self.cnn_detector.is_ready():
            messagebox.showerror(self._get_string("error_title"), self._get_string("cnn_not_available_long"))
            return "break"

        if not PILLOW_AVAILABLE:
            messagebox.showerror(self._get_string("error_title"), self._get_string("pillow_not_available"))
            return "break"

        try:
            image = ImageGrab.grabclipboard()
            if image is None:
                messagebox.showinfo(self._get_string("warning_title"), self._get_string("no_image_in_clipboard"))
                return "break"
            
            if not isinstance(image, Image.Image):
                try:
                    if isinstance(image, (bytes, bytearray)):
                        image = Image.open(io.BytesIO(image))
                    else:
                        messagebox.showinfo(self._get_string("warning_title"), self._get_string("clipboard_not_image"))
                        return "break"
                except Exception:
                    messagebox.showerror(self._get_string("error_title"), self._get_string("clipboard_not_valid_image"))
                    return "break"

            self._process_image_async(image)
        except Exception as e:
            messagebox.showerror(self._get_string("error_title"), self._get_string("error_pasting_image", e=e))
        
        return "break"

    def open_file(self):
        if not self.cnn_detector.is_ready():
            messagebox.showerror(self._get_string("error_title"), self._get_string("cnn_not_available_long_h5"))
            return

        if not PILLOW_AVAILABLE:
            messagebox.showerror(self._get_string("error_title"), self._get_string("pillow_not_available"))
            return

        file_path = filedialog.askopenfilename(title=self._get_string("select_image_title"), filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")])
        if not file_path:
            return
        try:
            image = Image.open(file_path)
            self._process_image_async(image)
        except Exception as e:
            messagebox.showerror(self._get_string("error_title"), self._get_string("error_opening_file", e=e))

    # ---------- threading to avoid UI freezing ----------
    def _process_image_async(self, image):
        if self._processing:
            return
        self._set_processing(True, self._get_string("status_processing"))
        # Create dialog in main thread
        dialog = ImageCropDialog(self.root, image, self._get_string)
        self.root.wait_window(dialog)
        if dialog.cropped_image is None:
            self.status_bar.config(text=self._get_string("status_cancelled"))
            self._set_processing(False)
            return
        # Process image in separate thread
        Thread(target=self._process_image_thread, args=(dialog.cropped_image,), daemon=True).start()

    def _process_image_thread(self, cropped_image):
        try:
            letters = self._process_image_cnn(cropped_image)
            if letters:
                letters_str = ''.join(letters)
                len_letters = len(letters)
                
                def insert_and_solve():
                    try:
                        self.entry_letters.config(state='normal')
                        self.entry_letters.delete(0, tk.END)
                        self.entry_letters.insert(0, letters_str)
                        self.status_bar.config(text=self._get_string("letters_detected_by_cnn", count=len_letters))
                        self.solve_async()
                    except Exception as e:
                        logger.error(f"Error in insert_and_solve: {e}", exc_info=True)
                
                self.root.after(0, insert_and_solve)
            else:
                self.root.after(0, lambda: messagebox.showwarning(self._get_string("warning_title"), self._get_string("no_letters_detected")))
                self.root.after(0, lambda: self.status_bar.config(text=self._get_string("no_letters_detected_status")))
        except Exception as e:
            logger.exception("Error in image processing")
            self.root.after(0, lambda: messagebox.showerror(self._get_string("error_title"), self._get_string("error_processing_image", e=e)))
        finally:
            self.root.after(0, lambda: self._set_processing(False))

    def _process_image_cnn(self, pil_image) -> Optional[List[str]]:
        if not self.cnn_detector.is_ready():
            return None
        
        if not SEGMENTATION_AVAILABLE:
            logger.error("Segmentation module not available")
            return None
        
        try:
            temp_path = Path(__file__).parent.parent / "temp_image_processing.png"
            pil_image.save(temp_path)
            
            segmented_letters, _ = segment_letters(str(temp_path), debug=False)
            
            if not segmented_letters:
                return None
            
            tile_images = get_tile_color_images(segmented_letters)
            
            if not tile_images:
                return None
            
            detected_letters = []
            for tile_name, tile_image in tile_images.items():
                try:
                    result = self.cnn_detector.predict_letter(tile_image, confidence_threshold=0.3)
                    if result:
                        letter, _ = result
                        detected_letters.append(letter)
                    else:
                        detected_letters.append('?')
                except Exception as e:
                    logger.error(f"Error processing tile {tile_name}: {e}")
            
            try:
                os.remove(temp_path)
            except Exception:
                pass
            
            return detected_letters if detected_letters else None
        except Exception as e:
            logger.exception("Error in CNN processing")
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            return None

    # ---------- solve (threaded) ----------
    def solve_async(self):
        if self._processing:
            return
        input_str = self.entry_letters.get().strip()
        if not input_str:
            messagebox.showwarning(self._get_string("warning_title"), self._get_string("enter_letters_prompt"))
            return
        try:
            min_length = int(self.spin_min.get())
        except Exception:
            min_length = 4
        letters = self.solver.normalize_input(input_str)
        if not letters:
            messagebox.showerror(self._get_string("error_title"), self._get_string("no_valid_letters"))
            return

        self._set_processing(True, self._get_string("status_searching"))
        Thread(target=self._solve_thread, args=(letters, min_length), daemon=True).start()

    def _solve_thread(self, letters: List[str], min_length: int):
        try:
            if self.var_false_letter.get():
                false_results, false_probs = self.solver.detect_false_letter(letters, min_length)
                self.root.after(0, lambda: self.display_false_letter_results(letters, false_results, false_probs))
                return

            if '?' in letters:
                confirmed, letter_word_map, probs = self.solver.find_anagrams_with_wildcard(letters, min_length)
                by_length_conf = defaultdict(list)
                for p in confirmed:
                    by_length_conf[len(p)].append(p)
                self.root.after(0, lambda: self.display_wildcard_results(letters, confirmed, letter_word_map, probs, by_length_conf))
                return

            words = self.solver.find_anagrams(letters, min_length)
            by_length = defaultdict(list)
            for p in words:
                by_length[len(p)].append(p)
            self.root.after(0, lambda: self.display_results(letters, words, by_length))

        finally:
            self.root.after(0, lambda: self._set_processing(False, self._get_string("status_ready")))

    # ---------- display and actions ----------
    def display_results(self, letters, words, by_length):
        self.text_results.config(state='normal')
        self.text_results.delete(1.0, tk.END)

        if not words:
            self.text_results.insert(tk.END, self._get_string("no_words_found") + "\n\n")
            self.text_results.insert(tk.END, self._get_string("available_letters", letters=' '.join(letters)) + "\n")
            self.text_results.insert(tk.END, self._get_string("min_length_display", length=self.spin_min.get()) + "\n")
            self.text_results.config(state='disabled')
            return

        self.text_results.insert(tk.END, "="*80 + "\n", 'title')
        self.text_results.insert(tk.END, self._get_string("total_words_found", count=len(words)) + "\n", 'title')
        self.text_results.insert(tk.END, "="*80 + "\n\n", 'title')

        for length in sorted(by_length.keys(), reverse=True):
            words_by_length = by_length[length]
            self.text_results.insert(tk.END, self._get_string("letters_of_length", length=length, count=len(words_by_length)) + "\n", 'subtitle')
            self.text_results.insert(tk.END, "-" * 80 + "\n")
            columns = 5
            for i in range(0, len(words_by_length), columns):
                line = words_by_length[i:i+columns]
                line_text = "   " + "  ".join(f"{p.lower():<15}" for p in line) + "\n"
                self.text_results.insert(tk.END, line_text)
            self.text_results.insert(tk.END, "\n")

        self.text_results.config(state='disabled')

    def display_wildcard_results(self, letters, confirmed, letter_word_map, probs, by_length_conf):
        self.text_results.config(state='normal')
        self.text_results.delete(1.0, tk.END)

        total_possible = sum(len(v) for v in letter_word_map.values())
        self.text_results.insert(tk.END, "="*80 + "\n", 'title')
        self.text_results.insert(tk.END, self._get_string("total_wildcard_found", confirmed=len(confirmed), hypotheses=total_possible) + "\n", 'title')
        self.text_results.insert(tk.END, "="*80 + "\n\n", 'title')

        self.text_results.insert(tk.END, self._get_string("confirmed_words") + "\n", 'subtitle')
        if not confirmed:
            self.text_results.insert(tk.END, self._get_string("none_found") + "\n\n")
        else:
            for length in sorted(by_length_conf.keys(), reverse=True):
                words_by_length = by_length_conf[length]
                self.text_results.insert(tk.END, self._get_string("letters_of_length", length=length, count=len(words_by_length)) + "\n")
                self.text_results.insert(tk.END, "-" * 80 + "\n")
                columns = 5
                for i in range(0, len(words_by_length), columns):
                    line = words_by_length[i:i+columns]
                    line_text = "   " + "  ".join(f"{p.lower():<15}" for p in line) + "\n"
                    self.text_results.insert(tk.END, line_text)
                self.text_results.insert(tk.END, "\n")

        self.text_results.insert(tk.END, self._get_string("possible_words") + "\n", 'subtitle')
        ordered_letters = sorted(probs.items(), key=lambda x: (-x[1], x[0]))
        any_shown = False
        for letter, prob in ordered_letters:
            word_list = letter_word_map.get(letter, [])
            if not word_list:
                continue
            any_shown = True
            pct = prob * 100
            self.text_results.insert(tk.END, self._get_string("letter_hypothesis", letter=letter, pct=pct, count=len(word_list)) + "\n", 'highlight')
            self.text_results.insert(tk.END, "-" * 60 + "\n")
            max_show = 80
            to_show = word_list[:max_show]
            columns = 5
            for i in range(0, len(to_show), columns):
                line = to_show[i:i+columns]
                line_text = "   " + "  ".join(f"{p.lower():<15}" for p in line) + "\n"
                self.text_results.insert(tk.END, line_text)
            if len(word_list) > max_show:
                self.text_results.insert(tk.END, self._get_string("and_more_words", count=len(word_list)-max_show) + "\n")
        if not any_shown:
            self.text_results.insert(tk.END, self._get_string("no_hypothesis_words") + "\n")

        self.text_results.config(state='disabled')

    def display_false_letter_results(self, letters, false_results, probs):
        self.text_results.config(state='normal')
        self.text_results.delete(1.0, tk.END)

        self.text_results.insert(tk.END, self._get_string("false_letter_analysis") + "\n", 'title')

        ordered = sorted(probs.items(), key=lambda x: -x[1])

        for letter, prob in ordered:
            word_list = false_results[letter]
            pct = prob * 100
            self.text_results.insert(tk.END, self._get_string("chance_of_being_false", letter=letter, pct=pct) + "\n", 'highlight')
            self.text_results.insert(tk.END, self._get_string("possible_words_removing", letter=letter, count=len(word_list)) + "\n")
            if len(word_list) > 0:
                preview = ", ".join(word_list[:20])
                self.text_results.insert(tk.END, self._get_string("example_words", preview=preview) + "\n")

        self.text_results.config(state='disabled')

    def copy_results(self):
        try:
            text = self.text_results.get(1.0, tk.END).strip()
            if not text:
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_bar.config(text=self._get_string("status_copied"))
        except Exception as e:
            messagebox.showerror(self._get_string("error_title"), self._get_string("error_copying", e=e))

    def _on_result_click(self, event):
        try:
            idx = self.text_results.index(f"@{event.x},{event.y}")
            word = self.text_results.get(f"{idx} wordstart", f"{idx} wordend").strip()
            if not word:
                return
            m = re.findall(r"[A-ZÁÀÂÃÉÊÍÓÔÕÚÇÜ]+", word.upper())
            if not m:
                return
            clean_word = m[0].lower()

            self.root.clipboard_clear()
            self.root.clipboard_append(clean_word)
            self.status_bar.config(text=self._get_string("status_copied_word", word=clean_word))

            prev_state = self.text_results.cget('state')
            try:
                self.text_results.config(state='normal')
                start = self.text_results.index(f"{idx} wordstart")
                end = self.text_results.index(f"{idx} wordend")
                self.text_results.tag_add('clicked', start, end)
            finally:
                self.text_results.config(state=prev_state)

            def _clear_tag():
                try:
                    self.text_results.config(state='normal')
                    self.text_results.tag_remove('clicked', start, end)
                    self.text_results.config(state=prev_state)
                except Exception:
                    pass

            self.text_results.after(800, _clear_tag)

        except Exception:
            return

    def save_results(self):
        try:
            text = self.text_results.get(1.0, tk.END).strip()
            if not text:
                messagebox.showinfo(self._get_string("info_title"), self._get_string("no_results_to_save"))
                return
            file_path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text', '*.txt')])
            if not file_path:
                return
            Path(file_path).write_text(text, encoding='utf-8')
            self.status_bar.config(text=self._get_string("status_saved", file_path=file_path))
        except Exception as e:
            messagebox.showerror(self._get_string("error_title"), self._get_string("error_saving", e=e))

    def clear(self):
        self.entry_letters.delete(0, tk.END)
        self.text_results.config(state='normal')
        self.text_results.delete(1.0, tk.END)
        self.text_results.config(state='disabled')
        self.label_letters.config(text='')
        self.status_bar.config(text=self._get_string("status_cleared"))

    # ---------- i18n methods ----------
    def _load_translations(self):
        """Loads UI strings from the JSON file."""
        try:
            lang_file_path = Path(__file__).parent.parent / "data" / "lang" / "ui_strings.json"
            with open(lang_file_path, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load language file: {e}")
            self.translations = {} # Fallback to empty

    def _get_string(self, key, **kwargs):
        """Gets a string for the current language."""
        lang = self.current_lang.get()
        if not self.translations:
            return key # Fallback to key if translations are not loaded
        
        val = self.translations.get(lang, {}).get(key, key)
        if kwargs:
            try:
                return val.format(**kwargs)
            except (KeyError, IndexError):
                return val # Return unformatted if a key is missing
        return val

    def _on_lang_change(self, *args):
        """Callback when the language is changed."""
        self._update_ui_text()

    def _on_dict_lang_change(self, *args):
        """Callback when the dictionary language is changed."""
        lang = self.dictionary_lang.get()
        self.solver.set_dictionary_language(lang)

    def _update_ui_text(self):
        """Updates all UI text elements with the current language."""
        self.root.title(self._get_string("window_title"))
        self.title_label.config(text=self._get_string("main_title"))
        self.subtitle_label.config(text=self._get_string("subtitle"))
        self.lang_label.config(text=self._get_string("language_label"))
        self.dict_lang_label.config(text=self._get_string("dictionary_label"))
        self.input_frame.config(text=self._get_string("input_frame"))
        self.paste_button.config(text=self._get_string("paste_button"))
        self.open_button.config(text=self._get_string("open_button"))
        
        cnn_status_text = self._get_string("cnn_ready") if self.cnn_detector.is_ready() else self._get_string("cnn_not_available")
        self.cnn_status_label.config(text=cnn_status_text)

        self.type_letters_label.config(text=self._get_string("type_letters_label"))
        self.false_letter_check.config(text=self._get_string("false_letter_check"))
        self.min_length_label.config(text=self._get_string("min_length_label"))
        self.solve_button.config(text=self._get_string("solve_button"))
        self.result_frame.config(text=self._get_string("results_frame"))
        self.copy_button.config(text=self._get_string("copy_button"))
        self.save_button.config(text=self._get_string("save_button"))
        self.clear_button.config(text=self._get_string("clear_button"))
        
        # Update status bar if it's in a default state
        if self.status_bar.cget("text") in [
            "Ready! Paste an image (Ctrl+V) or type the letters.",
            "Pronto! Cole uma imagem (Ctrl+V) ou digite as letras."
        ]:
            self.status_bar.config(text=self._get_string("status_ready"))


def main():
    root = tk.Tk()
    app = WordsStreamGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
