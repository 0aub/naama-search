"""
Production Processing - Winner Configuration ONLY
full_proc: morphology=True + lexical_filtering=True
"""

import re
import yaml
import pyarabic.araby as araby
from typing import Dict, List, Set


class ProductionNormalizer:
    """Hardcoded text normalizer for production"""

    _re_ar_diac = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]')
    _re_ws = re.compile(r'\s+')

    def __init__(self):
        # Protected terms that should not be heavily normalized
        self.protected_terms = {
            'بسة', 'قطة', 'قط', 'قطط', 'القطط', 'هرة', 'هر', 'كلب', 'كلاب',
            'حصان', 'خيل', 'فرس', 'نحل', 'عسل', 'مواشي', 'أبقار', 'غنم'
        }

    def normalize(self, text: str) -> str:
        """Normalize Arabic text"""
        if not text:
            return ""

        # Remove diacritics
        text = self._re_ar_diac.sub('', text)

        # Normalize whitespace
        text = self._re_ws.sub(' ', text)

        # Basic Arabic normalization (but protect specific terms)
        words = text.split()
        normalized_words = []

        for word in words:
            if word in self.protected_terms:
                normalized_words.append(word)
            else:
                # Basic normalization
                word = araby.normalize_alef(word)
                word = araby.normalize_teh(word)
                normalized_words.append(word)

        return ' '.join(normalized_words).strip()


class ProductionMorphReducer:
    """Hardcoded morphological reducer for production"""

    _re_tok = re.compile(r"\S+")

    def __init__(self):
        # Load morphology config
        with open('config/morphology.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        arabic_config = config["arabic_stemming"]
        self.prefixes = tuple(arabic_config["prefixes"])
        self.suffixes = tuple(arabic_config["suffixes"])

    def reduce(self, text: str) -> str:
        """Apply morphological reduction"""
        if not text:
            return ""

        tokens = self._re_tok.findall(text)
        reduced_tokens = []

        for token in tokens:
            if araby.is_arabicword(token):
                reduced_token = self._reduce_word(token)
                reduced_tokens.append(reduced_token)
            else:
                reduced_tokens.append(token)

        return ' '.join(reduced_tokens)

    def _reduce_word(self, word: str) -> str:
        """Reduce a single Arabic word"""
        # Remove prefixes
        for prefix in self.prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break

        # Remove suffixes
        for suffix in self.suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break

        return word


class ProductionLexicalFilter:
    """Hardcoded lexical filter for production"""

    def __init__(self, normalizer):
        self.normalizer = normalizer

        # Load lexical config
        with open('config/lexical.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.arabic_species = config.get('arabic_species', {})
        self.arabic_actions = config.get('arabic_actions', [])
        self.domains = config.get('domains', {})

    def passes_filter(self, query: str, document_text: str) -> bool:
        """Check if document passes lexical filtering"""
        # Normalize both query and document
        norm_query = self.normalizer.normalize(query.lower())
        norm_doc = self.normalizer.normalize(document_text.lower())

        # Extract key terms from query
        query_terms = set(norm_query.split())

        # Check for species matches
        for term in query_terms:
            if term in self.arabic_species:
                related_terms = self.arabic_species[term]
                if any(related_term in norm_doc for related_term in related_terms):
                    return True

        # Check for action matches
        for action in self.arabic_actions:
            if action in norm_query and action in norm_doc:
                return True

        # Default: pass if no specific filtering rules apply
        return True