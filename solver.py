#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Words on Stream - Gerador de Anagramas (versão final com CNN)
Melhorias aplicadas:
- Uso de pathlib, logging e melhores tratamentos de exceções
- Pré-processamento do dicionário para agilizar busca (counters por palavra, index por tamanho)
- Detecção de letras com CNN (redes neurais convolucionais)
- Segmentação de tiles usando visão computacional (segmentar.py)
- Processamento de imagens em thread para não travar a UI
- Botões para salvar resultados, copiar para área de transferência e limpar
- Mensagens de status mais claras e proteção contra múltiplos cliques

Requerimentos:
- Python 3.8+
- pip install pillow numpy tensorflow opencv-python

Modelos pré-treinados necessários:
- letter_detector_final.h5 (modelo CNN)
- label_map.json (mapa de classes)

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

# Imports para processamento de imagem
try:
    from PIL import Image, ImageTk, ImageGrab
    PILLOW_DISPONIVEL = True
except Exception:
    PILLOW_DISPONIVEL = False

# Imports para CNN
try:
    from tensorflow import keras
    CNN_DISPONIVEL = True
except Exception:
    CNN_DISPONIVEL = False

# Imports para segmentação
try:
    from segmentar import segment_letters, get_tile_color_images
    SEGMENTACAO_DISPONIVEL = True
except Exception:
    SEGMENTACAO_DISPONIVEL = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TrieNode:
    """Nó da estrutura Trie para busca eficiente de prefixos."""
    
    def __init__(self):
        self.filhos: Dict[str, 'TrieNode'] = {}
        self.eh_palavra = False
        self.palavra_completa: Optional[str] = None

# Mapas de equivalência entre vogais/consoantes base e variantes acentuadas.
# Usados para permitir que letras sem acento possam formar palavras que contenham
# vogais acentuadas (e o contrário).
ACCENT_EQUIV: Dict[str, List[str]] = {
    'A': ['A', 'Á', 'À', 'Â', 'Ã', 'Ä'],
    'E': ['E', 'É', 'È', 'Ê', 'Ë'],
    'I': ['I', 'Í', 'Ì', 'Î', 'Ï'],
    'O': ['O', 'Ó', 'Ò', 'Ô', 'Õ', 'Ö'],
    'U': ['U', 'Ú', 'Ù', 'Û', 'Ü'],
    'C': ['C', 'Ç'],
}

# Mapa inverso: variante -> base
VARIANT_TO_BASE: Dict[str, str] = {}
for base, variants in ACCENT_EQUIV.items():
    for v in variants:
        VARIANT_TO_BASE[v] = base


class LetterDetectorCNN:
    """Wrapper para carregar e usar o modelo CNN de detecção de letras."""
    
    def __init__(self, model_path='letter_detector_final.h5', label_map_path='label_map.json'):
        self.model = None
        self.label_map = {}
        self.img_size = 64
        
        try:
            # Carregar o modelo
            if Path(model_path).exists():
                self.model = keras.models.load_model(model_path)
                logger.info(f"Modelo CNN carregado: {model_path}")
            else:
                logger.warning(f"Modelo não encontrado em: {model_path}")
                
            # Carregar mapa de labels
            if Path(label_map_path).exists():
                with open(label_map_path, 'r') as f:
                    self.label_map = json.load(f)
                logger.info(f"Label map carregado: {len(self.label_map)} classes")
            else:
                logger.warning(f"Label map não encontrado em: {label_map_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo CNN: {e}")
    
    def is_ready(self) -> bool:
        """Verifica se o modelo foi carregado corretamente."""
        return self.model is not None and len(self.label_map) > 0
    
    def predict_letter(self, tile_image: np.ndarray, confidence_threshold: float = 0.3) -> Optional[Tuple[str, float]]:
        """Prediz a letra de um tile individual.
        
        Args:
            tile_image: Imagem do tile (BGR ou escala de cinza)
            confidence_threshold: Limiar mínimo de confiança
            
        Returns:
            Tupla (letra, confiança) ou None se abaixo do limiar
        """
        if not self.is_ready():
            return None
        
        try:
            # Converter para escala de cinza se necessário
            if len(tile_image.shape) == 3:
                gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = tile_image
            
            # Redimensionar para tamanho esperado pelo modelo
            resized = cv2.resize(gray, (self.img_size, self.img_size))
            
            # Normalizar (0-1)
            normalized = resized.astype('float32') / 255.0
            
            # Adicionar dimensão de batch e canal
            input_data = np.expand_dims(normalized, axis=(0, -1))  # (1, 64, 64, 1)
            
            # Predição
            prediction = self.model.predict(input_data, verbose=0)[0]
            confidence = float(np.max(prediction))
            class_idx = int(np.argmax(prediction))
            
            # Verificar limiar de confiança
            if confidence < confidence_threshold:
                logger.debug(f"Predição abaixo do limiar: {confidence:.2%}")
                return None
            
            # Obter letra correspondente
            reverse_map = {v: k for k, v in self.label_map.items()}
            if class_idx in reverse_map:
                letter = reverse_map[class_idx]
                logger.debug(f"Letra predita: {letter} ({confidence:.2%})")
                return (letter, confidence)
            else:
                logger.warning(f"Classe {class_idx} não encontrada no mapa de labels")
                return None
        except Exception as e:
            logger.error(f"Erro na predição CNN: {e}")
            return None


class WordsStreamSolver:
    """Solucionador de anagramas para Words on Stream (melhorado com Trie + Poda Recursiva)

    Estratégia:
      - Carrega dicionário em maiúsculas
      - Constrói uma Trie para busca eficiente de prefixos
      - Usa busca recursiva com poda: elimina ramos impossíveis baseado nas letras disponíveis
      - Muito mais rápido que força bruta, especialmente com dicionários grandes
    """

    def __init__(self, dicionario_path: str | Path = "palavras_pt.txt"):
        self.dicionario_path = Path(dicionario_path)
        self.palavras: Set[str] = set()
        self.trie = TrieNode()  # Estrutura Trie para busca rápida
        self._carregar_dicionario()

    def _baixar_dicionario(self) -> Set[str]:
        """Tenta baixar dicionários públicos em PT-BR. Retorna conjunto de palavras."""
        urls = [
            "https://www.ime.usp.br/~pf/dicios/br-utf8.txt",
            #"https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt",
        ]

        for url in urls:
            try:
                logger.info(f"Tentando baixar dicionário: {url}")
                response = urllib.request.urlopen(url, timeout=10)
                conteudo = response.read().decode('utf-8', errors='ignore')
                palavras = set(linha.strip().upper() for linha in conteudo.splitlines() if linha.strip())

                # salvar localmente
                try:
                    self.dicionario_path.write_text('\n'.join(sorted(palavras)), encoding='utf-8')
                except Exception as e:
                    logger.warning(f"Não foi possível salvar dicionário localmente: {e}")

                return palavras
            except Exception as e:
                logger.warning(f"Falha ao baixar {url}: {e}")
                continue

        return set()

    def _dicionario_basico(self) -> Set[str]:
        """Dicionário mínimo embutido, em maiúsculas."""
        palavras_basicas = """
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
        return set(palavras_basicas)

    def _carregar_dicionario(self):
        """Carrega e constrói a Trie a partir do dicionário local (ou baixa um).

        Ao final, temos:
          - self.palavras (set)
          - self.trie (estrutura Trie para busca eficiente)
        """
        palavras = set()

        if self.dicionario_path.exists():
            try:
                texto = self.dicionario_path.read_text(encoding='utf-8')
                palavras = set(l.strip().upper() for l in texto.splitlines() if l.strip())
                logger.info(f"Carregado dicionário local: {self.dicionario_path} ({len(palavras)} palavras)")
            except Exception as e:
                logger.warning(f"Erro ao ler dicionário local: {e}")

        if not palavras:
            palavras = self._baixar_dicionario()

        if not palavras:
            logger.info("Usando dicionário básico embutido")
            palavras = self._dicionario_basico()

        # normaliza
        palavras = set(p for p in palavras if re.match(r'^[A-ZÀ-Ÿ]+$', p))
        self.palavras = palavras
        
        # Constrói a Trie
        self.trie = TrieNode()
        for palavra in palavras:
            self._inserir_na_trie(palavra)

    def normalizar_entrada(self, entrada: str) -> List[str]:
        entrada = entrada.upper().strip()
        letras = re.findall(r'[A-ZÁÀÂÃÉÊÍÓÔÕÚÇÜ]', entrada)
        return letras

    def _inserir_na_trie(self, palavra: str):
        """Insere uma palavra na Trie."""
        node = self.trie
        for letra in palavra:
            if letra not in node.filhos:
                node.filhos[letra] = TrieNode()
            node = node.filhos[letra]
        node.eh_palavra = True
        node.palavra_completa = palavra

    def _equivalent_letters(self, letra: str) -> List[str]:
        """Retorna a lista de letras equivalentes para 'letra'.

        Ex: 'A' -> ['A','Á','À',...], 'Á' -> ['A','Á','À',...], 'B' -> ['B']
        """
        letra = letra.upper()
        # Se for uma variante mapeada, obter a base e então todas as variantes
        base = VARIANT_TO_BASE.get(letra, letra)
        return ACCENT_EQUIV.get(base, [letra])

    def pode_formar_palavra(self, contador_palavra: Counter, letras_disponiveis: Counter) -> bool:
        """Verifica se contador_palavra pode ser formado por letras_disponiveis (mantido para compatibilidade)."""
        for letra, qtd in contador_palavra.items():
            if letras_disponiveis[letra] < qtd:
                return False
        return True

    def encontrar_anagramas(self, letras: List[str], min_tamanho: int = 4) -> List[str]:
        """
        Encontra anagramas usando Trie + Poda Recursiva.
        
        A busca recursiva explora a Trie só pelos caminhos que podem ser formados
        com as letras disponíveis, podando ramos impossíveis cedo.
        """
        letras_disponiveis = Counter(letras)
        resultados: List[str] = []
        
        # Busca recursiva com poda
        self._buscar_recursivo(
            node=self.trie,
            letras_restantes=letras_disponiveis,
            min_tamanho=min_tamanho,
            resultados=resultados
        )
        
        # Ordena por comprimento decrescente e lexicograficamente
        resultados.sort(key=lambda p: (-len(p), p))
        return resultados

    def _buscar_recursivo(
        self,
        node: TrieNode,
        letras_restantes: Counter,
        min_tamanho: int,
        resultados: List[str]
    ):
        """
        Busca recursiva com poda na Trie.
        
        Estratégia:
        - Se chegamos a uma palavra válida e tamanho >= min_tamanho, adicionamos aos resultados
        - Para cada letra filha, se temos essa letra disponível, descemos recursivamente
        - Automaticamente poda ramos onde não temos as letras necessárias
        """
        
        # Base: se este nó marca final de palavra e tem tamanho mínimo
        if node.eh_palavra and len(node.palavra_completa) >= min_tamanho:
            resultados.append(node.palavra_completa)
        
        # Recursão: explorar filhos apenas se temos as letras.
        # Para suportar equivalência entre vogais acentuadas e não-acentuadas,
        # iteramos pelas letras disponíveis e procuramos filhos equivalentes.
        for letra_disponivel, qtd in list(letras_restantes.items()):
            if qtd <= 0:
                continue
            # obter letras equivalentes que podem existir na Trie
            candidatos = self._equivalent_letters(letra_disponivel)
            for cand in candidatos:
                filho_node = node.filhos.get(cand)
                if not filho_node:
                    continue
                # Usar a letra disponível (a contagem refere-se à entrada do usuário)
                letras_restantes[letra_disponivel] -= 1
                # Recursar descendo no nó correspondente
                self._buscar_recursivo(
                    node=filho_node,
                    letras_restantes=letras_restantes,
                    min_tamanho=min_tamanho,
                    resultados=resultados
                )
                # Restaurar (backtrack)
                letras_restantes[letra_disponivel] += 1


class ImageCropDialog(tk.Toplevel):
    """Diálogo para selecionar área de recorte na imagem (igual ao original, com pequenas melhorias)"""

    def __init__(self, parent, imagem: 'Image.Image'):
        super().__init__(parent)
        self.title("Selecionar Área com Letras")
        self.imagem_original = imagem
        self.imagem_recortada: Optional['Image.Image'] = None
        self.transient(parent)
        self.grab_set()

        # Configurações
        self.rect = None
        self.start_x = None
        self.start_y = None

        # Redimensiona imagem para caber na tela
        max_width, max_height = 900, 700
        self.scale = min(max_width / imagem.width, max_height / imagem.height, 1.0)

        new_size = (int(imagem.width * self.scale), int(imagem.height * self.scale))
        self.imagem_display = imagem.resize(new_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.imagem_display)

        # Canvas para desenhar
        self.canvas = tk.Canvas(self, width=new_size[0], height=new_size[1], cursor="cross")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Instruções
        frame_info = ttk.Frame(self)
        frame_info.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame_info, text="🖱️ Arraste para selecionar a área com as letras", font=('Arial', 10)).pack(side=tk.LEFT, padx=5)

        # Botões
        frame_btns = ttk.Frame(self)
        frame_btns.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(frame_btns, text="✓ Confirmar", command=self.confirmar).pack(side=tk.RIGHT, padx=5)
        ttk.Button(frame_btns, text="✗ Cancelar", command=self.cancelar).pack(side=tk.RIGHT, padx=5)
        ttk.Button(frame_btns, text="🔄 Usar Imagem Inteira", command=self.usar_toda).pack(side=tk.RIGHT, padx=5)

        # Eventos do mouse
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Centraliza
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

    def usar_toda(self):
        self.imagem_recortada = self.imagem_original
        self.destroy()

    def confirmar(self):
        if not self.rect:
            messagebox.showwarning("Aviso", "Selecione uma área primeiro ou use 'Usar Imagem Inteira'")
            return

        coords = self.canvas.coords(self.rect)
        x1, y1, x2, y2 = [int(c / self.scale) for c in coords]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            messagebox.showwarning("Aviso", "Área muito pequena! Selecione uma área maior.")
            return

        self.imagem_recortada = self.imagem_original.crop((x1, y1, x2, y2))
        self.destroy()

    def cancelar(self):
        self.imagem_recortada = None
        self.destroy()


class WordsStreamGUI:
    """Interface gráfica principal com melhorias de usabilidade."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Words on Stream - Gerador de Anagramas")
        self.root.geometry("980x720")

        self.solver = WordsStreamSolver()
        self.cnn_detector = LetterDetectorCNN()
        self.imagem_atual = None
        self._processando = False

        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        self._criar_interface()

        # Bindings
        self.root.bind('<Control-v>', lambda e: self.colar_imagem())
        self.root.bind('<Control-V>', lambda e: self.colar_imagem())

    def _criar_interface(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="🎮 WORDS ON STREAM", font=('Arial', 20, 'bold')).pack()
        ttk.Label(title_frame, text="Gerador de Anagramas com CNN", font=('Arial', 10)).pack()

        input_frame = ttk.LabelFrame(main_frame, text="📥 Entrada", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="📋 Colar Imagem (Ctrl+V)", command=self.colar_imagem).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📂 Abrir Arquivo", command=self.abrir_arquivo).pack(side=tk.LEFT, padx=5)

        # Status do detector CNN
        cnn_status = "✓ CNN Pronta" if self.cnn_detector.is_ready() else "⚠️ CNN não disponível"
        cnn_color = 'green' if self.cnn_detector.is_ready() else 'orange'
        ttk.Label(btn_frame, text=cnn_status, foreground=cnn_color).pack(side=tk.LEFT, padx=10)

        text_frame = ttk.Frame(input_frame)
        text_frame.pack(fill=tk.X)

        ttk.Label(text_frame, text="Ou digite as letras:").pack(side=tk.LEFT, padx=5)
        self.entry_letras = ttk.Entry(text_frame, font=('Arial', 12))
        self.entry_letras.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entry_letras.bind('<Return>', lambda e: self.resolver_async())

        ttk.Label(text_frame, text="Mín:").pack(side=tk.LEFT, padx=5)
        self.spin_min = ttk.Spinbox(text_frame, from_=2, to=12, width=5)
        self.spin_min.set(4)
        self.spin_min.pack(side=tk.LEFT, padx=5)

        ttk.Button(text_frame, text="🔍 Resolver", command=self.resolver_async).pack(side=tk.LEFT, padx=5)

        self.label_letras = ttk.Label(input_frame, text="", font=('Arial', 11, 'bold'))
        self.label_letras.pack(pady=5)

        result_frame = ttk.LabelFrame(main_frame, text="📊 Resultados", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.text_resultados = scrolledtext.ScrolledText(result_frame, font=('Courier', 10), wrap=tk.WORD, state='disabled')
        self.text_resultados.pack(fill=tk.BOTH, expand=True)

        # tags
        self.text_resultados.tag_config('titulo', font=('Arial', 12, 'bold'))
        self.text_resultados.tag_config('subtitulo', font=('Arial', 10, 'bold'))
        self.text_resultados.tag_config('destaque', foreground='#0066cc', font=('Arial', 11, 'bold'))
        # clique em palavra copia para área de transferência
        self.text_resultados.bind('<Button-1>', self._on_result_click)
        # tag temporária para destacar palavra clicada
        self.text_resultados.tag_config('clicked', background='#fff2b2')

        # ações rápidas
        actions = ttk.Frame(main_frame)
        actions.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(actions, text="📋 Copiar Resultados", command=self.copiar_resultados).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="💾 Salvar Resultados", command=self.salvar_resultados).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="🧹 Limpar", command=self.limpar).pack(side=tk.LEFT, padx=5)

        self.status_bar = ttk.Label(main_frame, text="Pronto! Cole uma imagem (Ctrl+V) ou digite as letras.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))

    # ---------- utilitários de UI ----------
    def _set_processando(self, valor: bool, msg: Optional[str] = None):
        self._processando = valor
        state = 'disabled' if valor else 'normal'
        # Desabilitar controles principais
        try:
            for widget in (self.entry_letras,):
                widget.config(state=state)
        except Exception:
            pass
        if msg:
            self.status_bar.config(text=msg)

    # ---------- entrada de imagem ----------
    def colar_imagem(self):
        if not self.cnn_detector.is_ready():
            messagebox.showerror("Erro", "CNN não disponível!\nVerifique se o modelo 'letter_detector_final.h5' e 'label_map.json' estão presentes.")
            return

        if not PILLOW_DISPONIVEL:
            messagebox.showerror("Erro", "Pillow não disponível!\nInstale: pip install pillow")
            return

        try:
            imagem = ImageGrab.grabclipboard()
            if imagem is None:
                messagebox.showinfo("Aviso", "Nenhuma imagem na área de transferência! Copie uma imagem e tente novamente.")
                return
            if not isinstance(imagem, Image.Image):
                # às vezes o clipboard devolve bytes; tentar abrir
                try:
                    if isinstance(imagem, (bytes, bytearray)):
                        imagem = Image.open(io.BytesIO(imagem))
                except Exception:
                    messagebox.showerror("Erro", "O conteúdo da área de transferência não é uma imagem válida.")
                    return

            self._processar_imagem_async(imagem)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao colar imagem:\n{e}")

    def abrir_arquivo(self):
        if not self.cnn_detector.is_ready():
            messagebox.showerror("Erro", "CNN não disponível!\nVerifique se o modelo 'letter_detector_final.h5' e 'label_map.json' estão presentes.")
            return

        if not PILLOW_DISPONIVEL:
            messagebox.showerror("Erro", "Pillow não disponível!\nInstale: pip install pillow")
            return

        arquivo = filedialog.askopenfilename(title="Selecionar Imagem", filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Todos", "*.*")])
        if not arquivo:
            return
        try:
            imagem = Image.open(arquivo)
            self._processar_imagem_async(imagem)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao abrir arquivo:\n{e}")

    # ---------- threading para não travar UI ----------
    def _processar_imagem_async(self, imagem):
        if self._processando:
            return
        self._set_processando(True, "Processando imagem...")
        # Criar dialog na thread principal
        dialog = ImageCropDialog(self.root, imagem)
        self.root.wait_window(dialog)
        if dialog.imagem_recortada is None:
            self.status_bar.config(text="Operação cancelada.")
            self._set_processando(False)
            return
        # Processar imagem em thread separada
        Thread(target=self._processar_imagem_thread, args=(dialog.imagem_recortada,), daemon=True).start()

    def _processar_imagem_thread(self, imagem_recortada):
        try:
            letras = self._processar_imagem_cnn(imagem_recortada)
            logger.info(f"DEBUG: Letras detectadas: {letras}")
            if letras:
                # inserir letras e resolver (na thread principal)
                letras_str = ''.join(letras)
                len_letras = len(letras)
                logger.info(f"DEBUG: letras_str = '{letras_str}', len = {len_letras}")
                
                def inserir_e_resolver():
                    try:
                        logger.info(f"DEBUG: Dentro de inserir_e_resolver, tentando inserir '{letras_str}'")
                        # Garantir que o campo está habilitado
                        self.entry_letras.config(state='normal')
                        self.entry_letras.delete(0, tk.END)
                        logger.info(f"DEBUG: Delete executado")
                        self.entry_letras.insert(0, letras_str)
                        logger.info(f"DEBUG: Insert executado com sucesso")
                        self.status_bar.config(text=f"✓ {len_letras} letras detectadas pela CNN!")
                        logger.info(f"DEBUG: Status bar atualizado")
                        self.resolver_async()
                        logger.info(f"DEBUG: resolver_async chamado")
                    except Exception as e:
                        logger.error(f"DEBUG: Erro em inserir_e_resolver: {e}", exc_info=True)
                
                logger.info(f"DEBUG: Agendando inserir_e_resolver com root.after(0, ...)")
                self.root.after(0, inserir_e_resolver)
            else:
                logger.warning("DEBUG: Nenhuma letra detectada")
                self.root.after(0, lambda: messagebox.showwarning("Aviso", "Nenhuma letra detectada! Verifique a imagem e tente novamente."))
                self.root.after(0, lambda: self.status_bar.config(text="Nenhuma letra detectada na imagem."))
        except Exception as e:
            logger.exception("Erro no processamento de imagem")
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao processar imagem:\n{e}"))
        finally:
            self.root.after(0, lambda: self._set_processando(False))

    def _processar_imagem_cnn(self, imagem_pil) -> Optional[List[str]]:
        """Processa imagem com segmentação e CNN para detectar letras.
        
        Fluxo:
        1. Converter imagem PIL para arquivo temporário
        2. Segmentar usando segmentar.py
        3. Extrair tiles
        4. Predizer cada tile com CNN
        5. Retornar lista de letras detectadas
        """
        if not self.cnn_detector.is_ready():
            return None
        
        if not SEGMENTACAO_DISPONIVEL:
            logger.error("Módulo de segmentação não disponível")
            return None
        
        try:
            # Salvar imagem PIL em arquivo temporário
            temp_path = "temp_image_processing.png"
            imagem_pil.save(temp_path)
            
            logger.info(f"Segmentando imagem: {temp_path}")
            
            # Segmentar imagem
            segmented_letters, original_img = segment_letters(temp_path, debug=False)
            
            if not segmented_letters:
                logger.warning("Nenhuma letra segmentada")
                return None
            
            logger.info(f"Encontradas {len(segmented_letters)} regiões de letras")
            
            # Extrair tiles em memória
            tile_images = get_tile_color_images(segmented_letters)
            
            if not tile_images:
                logger.warning("Nenhuma imagem de tile extraída")
                return None
            
            logger.info(f"Extraídos {len(tile_images)} tiles")
            
            # Predizer cada tile
            letras_detectadas = []
            for tile_name, tile_image in tile_images.items():
                try:
                    resultado = self.cnn_detector.predict_letter(tile_image, confidence_threshold=0.3)
                    if resultado:
                        letra, confianca = resultado
                        letras_detectadas.append(letra)
                        logger.info(f"Tile {tile_name}: {letra} ({confianca:.2%})")
                    else:
                        logger.debug(f"Tile {tile_name}: Predição abaixo do limiar ou erro")
                except Exception as e:
                    logger.error(f"Erro ao processar tile {tile_name}: {e}")
            
            # Limpar arquivo temporário
            try:
                os.remove(temp_path)
            except Exception:
                pass
            
            return letras_detectadas if letras_detectadas else None
        except Exception as e:
            logger.exception("Erro no processamento CNN")
            # Tentar limpar arquivo temporário em caso de erro
            try:
                if os.path.exists("temp_image_processing.png"):
                    os.remove("temp_image_processing.png")
            except Exception:
                pass
            return None

    # ---------- resolver (threaded) ----------
    def resolver_async(self):
        if self._processando:
            return
        entrada = self.entry_letras.get().strip()
        if not entrada:
            messagebox.showwarning("Aviso", "Digite as letras ou cole uma imagem!")
            return
        try:
            min_tamanho = int(self.spin_min.get())
        except Exception:
            min_tamanho = 4
        letras = self.solver.normalizar_entrada(entrada)
        if not letras:
            messagebox.showerror("Erro", "Nenhuma letra válida encontrada!")
            return

        self._set_processando(True, "Buscando palavras...")
        Thread(target=self._resolver_thread, args=(letras, min_tamanho), daemon=True).start()

    def _resolver_thread(self, letras: List[str], min_tamanho: int):
        try:
            palavras = self.solver.encontrar_anagramas(letras, min_tamanho)
            por_tamanho = defaultdict(list)
            for p in palavras:
                por_tamanho[len(p)].append(p)
            self.root.after(0, lambda: self.exibir_resultados(letras, palavras, por_tamanho))
            self.root.after(0, lambda: self.status_bar.config(text=f"✓ {len(palavras)} palavras encontradas!"))
        except Exception as e:
            logger.exception("Erro ao resolver anagramas")
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao buscar palavras:\n{e}"))
        finally:
            self.root.after(0, lambda: self._set_processando(False))

    # ---------- exibição e ações ----------
    def exibir_resultados(self, letras, palavras, por_tamanho):
        self.text_resultados.config(state='normal')
        self.text_resultados.delete(1.0, tk.END)

        if not palavras:
            self.text_resultados.insert(tk.END, "❌ Nenhuma palavra encontrada com essas letras.\n\n")
            self.text_resultados.insert(tk.END, f"Letras disponíveis: {' '.join(letras)}\n")
            self.text_resultados.insert(tk.END, f"Tamanho mínimo: {self.spin_min.get()}\n")
            self.text_resultados.config(state='disabled')
            return

        self.text_resultados.insert(tk.END, "="*80 + "\n", 'titulo')
        self.text_resultados.insert(tk.END, f"  TOTAL: {len(palavras)} PALAVRAS ENCONTRADAS\n", 'titulo')
        self.text_resultados.insert(tk.END, "="*80 + "\n\n", 'titulo')

        for tamanho in sorted(por_tamanho.keys(), reverse=True):
            palavras_tamanho = por_tamanho[tamanho]
            self.text_resultados.insert(tk.END, f"📝 {tamanho} LETRAS ({len(palavras_tamanho)} palavras)\n", 'subtitulo')
            self.text_resultados.insert(tk.END, "-" * 80 + "\n")
            colunas = 5
            for i in range(0, len(palavras_tamanho), colunas):
                linha = palavras_tamanho[i:i+colunas]
                texto_linha = "   " + "  ".join(f"{p.lower():<15}" for p in linha) + "\n"
                self.text_resultados.insert(tk.END, texto_linha)
            self.text_resultados.insert(tk.END, "\n")

        """ palavra_mais_longa = palavras[0]
        self.text_resultados.insert(tk.END, "\n" + "="*80 + "\n")
        self.text_resultados.insert(tk.END, f"🏆 PALAVRA MAIS LONGA: {palavra_mais_longa.lower()} ({len(palavra_mais_longa)} letras)\n", 'destaque')
        self.text_resultados.insert(tk.END, "="*80 + "\n") """
        self.text_resultados.config(state='disabled')

    def copiar_resultados(self):
        try:
            texto = self.text_resultados.get(1.0, tk.END).strip()
            if not texto:
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(texto)
            self.status_bar.config(text="Copiado para a área de transferência.")
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível copiar:\n{e}")

    def _on_result_click(self, event):
        """Handler quando o usuário clica em uma palavra nos resultados.

        Copia a palavra clicada para a área de transferência e destaca-a brevemente.
        """
        try:
            # obter índice textual sob o cursor
            idx = self.text_resultados.index(f"@{event.x},{event.y}")
            # obter palavra (wordstart/wordend considera pontuação)
            palavra = self.text_resultados.get(f"{idx} wordstart", f"{idx} wordend").strip()
            if not palavra:
                return
            # extrair apenas letras (incluindo acentos) — protege contra cabeçalhos e símbolos
            m = re.findall(r"[A-ZÁÀÂÃÉÊÍÓÔÕÚÇÜ]+", palavra.upper())
            if not m:
                return
            palavra_clean = m[0].lower()

            # copiar para área de transferência
            self.root.clipboard_clear()
            self.root.clipboard_append(palavra_clean)
            self.status_bar.config(text=f"Copiado: {palavra_clean}")

            # destacar palavra brevemente
            # precisamos temporariamente permitir edição para adicionar tag
            prev_state = self.text_resultados.cget('state')
            try:
                self.text_resultados.config(state='normal')
                start = self.text_resultados.index(f"{idx} wordstart")
                end = self.text_resultados.index(f"{idx} wordend")
                self.text_resultados.tag_add('clicked', start, end)
            finally:
                # restaurar estado desabilitado (será re-habilitado após remoção da tag)
                self.text_resultados.config(state=prev_state)

            # remover destaque após 800ms
            def _clear_tag():
                try:
                    self.text_resultados.config(state='normal')
                    self.text_resultados.tag_remove('clicked', start, end)
                    self.text_resultados.config(state=prev_state)
                except Exception:
                    pass

            self.text_resultados.after(800, _clear_tag)

        except Exception:
            # não falhar na UI por um clique inválido
            return

    def salvar_resultados(self):
        try:
            texto = self.text_resultados.get(1.0, tk.END).strip()
            if not texto:
                messagebox.showinfo("Info", "Nenhum resultado para salvar.")
                return
            arquivo = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text', '*.txt')])
            if not arquivo:
                return
            Path(arquivo).write_text(texto, encoding='utf-8')
            self.status_bar.config(text=f"Salvo em: {arquivo}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar:\n{e}")

    def limpar(self):
        self.entry_letras.delete(0, tk.END)
        self.text_resultados.config(state='normal')
        self.text_resultados.delete(1.0, tk.END)
        self.text_resultados.config(state='disabled')
        self.label_letras.config(text='')
        self.status_bar.config(text="Limpo. Cole uma imagem (Ctrl+V) ou digite as letras.")


def main():
    root = tk.Tk()
    app = WordsStreamGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
