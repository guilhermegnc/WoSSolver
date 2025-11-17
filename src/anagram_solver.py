# -*- coding: utf-8 -*-
import logging
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Set, Optional, Dict

from constants import ACCENT_EQUIV, VARIANT_TO_BASE
from trie import TrieNode

logger = logging.getLogger(__name__)


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
