# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional

class TrieNode:
    """Node for the Trie structure for efficient prefix searching."""
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_word = False
        self.full_word: Optional[str] = None
