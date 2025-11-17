# -*- coding: utf-8 -*-
from typing import Dict, List

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
