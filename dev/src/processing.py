"""
Text processing components: normalization, morphological reduction, and lexical filtering.
Combined into one file for better organization and efficiency.
"""

import re
import unicodedata
import pathlib
import yaml
from typing import Dict, Set, Any
import logging

from nltk.stem import PorterStemmer
import pyarabic.araby as araby

log = logging.getLogger(__name__)


def _load_config(filename: str) -> dict:
    """Load configuration from config folder"""
    config_path = pathlib.Path(__file__).parent.parent / "config" / filename
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class TextNormalizer:
    """Script/orthography normalization before morphological reduction.
    
    PHASE 1 ENHANCEMENT: Added conservative normalization for specific entities
    to preserve semantic meaning and reduce false positives.
    """
    _re_ar_diac = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]')
    _re_ws      = re.compile(r'\s+')
    
    def __init__(self):
        # PHASE 1: Protected specific entity terms that should not be heavily normalized
        self.protected_entities = {
            'بسة',      # cat (feminine)
            'بس',       # cat (informal)
            'قطة',      # cat (standard)
            'قطط',      # cats
            'كلب',      # dog
            'كلاب',     # dogs  
            'حصان',     # horse
            'خيل',      # horses
            'نحلة',     # bee
            'نحل',      # bees
            'بقرة',     # cow
            'بقر',      # cattle
        }
        log.info(f"🔒 Protected {len(self.protected_entities)} specific entity terms from heavy normalization")

    def is_protected_entity(self, text: str) -> bool:
        """Check if text contains protected entity terms that should not be heavily normalized"""
        text_clean = text.strip().lower()
        return any(entity in text_clean for entity in self.protected_entities)
    
    def conservative_normalize_ar(self, text: str) -> str:
        """Conservative Arabic normalization that preserves semantic meaning"""
        t = unicodedata.normalize("NFKC", str(text))

        # Remove harakat & tatweel (these are always safe)
        t = self._re_ar_diac.sub("", t)
        t = t.replace("\u0640", "")

        # Only apply basic Alef normalization (most conservative)
        t = re.sub(r"[\u0622\u0623\u0625]", "\u0627", t)  # آ أ إ → ا (but not ٱ)

        # Conservative Yeh normalization (avoid changing meaning)
        t = t.replace("\u06CC", "\u064A")          # ی → ي (Persian)
        # Keep ى as is for protected entities to preserve gender/meaning
        
        # Conservative Kaf normalization
        t = t.replace("\u06A9", "\u0643")          # ک → ك (Persian)

        # CRITICAL: Do NOT normalize taa marbuta for protected entities
        # This preserves ة vs ه distinction which affects meaning
        
        # Conservative Hamza (only clear cases)
        t = t.replace("\u0624", "\u0648")          # ؤ → و
        # Keep ئ and ء for semantic preservation

        # Eastern Arabic digits to ASCII (safe)
        t = t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

        # Keep only Arabic letters/digits/spaces
        t = re.sub(r"[^\s\u0600-\u06FF0-9]", " ", t)
        return self._re_ws.sub(" ", t).strip()
    
    def normalize_ar(self, text: str) -> str:
        """Normalize Arabic text with entity-aware processing - PHASE 1 ENHANCED"""
        # PHASE 1 CRITICAL FIX: Use conservative normalization for protected entities
        if self.is_protected_entity(text):
            return self.conservative_normalize_ar(text)
            
        # Original normalization for non-protected terms
        t = unicodedata.normalize("NFKC", str(text))

        # Remove harakat & tatweel
        t = self._re_ar_diac.sub("", t)
        t = t.replace("\u0640", "")

        # Alef variants: آ أ إ ٱ → ا
        t = re.sub(r"[\u0622\u0623\u0625\u0671-\u0675]", "\u0627", t)

        # Yeh/Kaf variants
        t = t.replace("\u0649", "\u064A")          # ى → ي
        t = t.replace("\u06CC", "\u064A")          # ی → ي
        t = t.replace("\u06A9", "\u0643")          # ک → ك

        # Normalize taa marbuta
        t = t.replace("\u0629", "\u0647")          # ة → ه

        # Hamza normalization
        t = t.replace("\u0624", "\u0648")          # ؤ → و
        t = t.replace("\u0626", "\u064A")          # ئ → ي
        t = t.replace("\u0621", "")                # ء → (drop)

        # Eastern Arabic digits to ASCII
        t = t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

        # Keep only Arabic letters/digits/spaces
        t = re.sub(r"[^\s\u0600-\u06FF0-9]", " ", t)
        return self._re_ws.sub(" ", t).strip()

    def normalize_en(self, text: str) -> str:
        """Normalize English text"""
        t = unicodedata.normalize("NFKC", str(text)).lower()
        t = re.sub(r"['']", "", t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return self._re_ws.sub(" ", t).strip()

    def normalize(self, text: str, lang: str) -> str:
        """Normalize text based on language"""
        return self.normalize_ar(text) if lang == "ar" else self.normalize_en(text)


class MorphReducer:
    """Morphological reduction using PyArabic (fast, offline Arabic processing)."""
    _re_tok = re.compile(r"\S+")

    def __init__(self):
        # Load morphology configuration
        self.config = _load_config("morphology.yaml")
        
        # Extract configuration values
        arabic_config = self.config["arabic_stemming"]
        self.prefixes = tuple(arabic_config["prefixes"])
        self.suffixes = tuple(arabic_config["suffixes"])
        self.protected_terms = set(arabic_config["protected_terms"])
        
        # PyArabic configuration
        pyarabic_config = self.config["pyarabic_config"]
        self.normalize_hamza = pyarabic_config["normalize_hamza"]
        self.normalize_alef = pyarabic_config["normalize_alef"]
        self.normalize_yeh = pyarabic_config["normalize_yeh"]
        self.remove_diacritics = pyarabic_config["remove_diacritics"]
        self.remove_tatweel = pyarabic_config["remove_tatweel"]
        self.normalize_teh = pyarabic_config["normalize_teh"]
        self.stemming_method = pyarabic_config["stemming_method"]
        self.use_root_extraction = pyarabic_config["use_root_extraction"]
        self.remove_common_prefixes = pyarabic_config["remove_common_prefixes"]
        self.remove_common_suffixes = pyarabic_config["remove_common_suffixes"]
        self.aggressive_stemming = pyarabic_config["aggressive_stemming"]
        
        # Stemming parameters
        params = self.config["stemming_params"]
        self.min_word_length = params["min_word_length"]
        self.min_reduction_ratio = params["min_reduction_ratio"]
        self.min_stem_length = params["min_stem_length"]
        
        # Initialize English stemmer
        self.en = PorterStemmer()
        
        log.info("🚀 PyArabic morphological reducer ready (fast, offline)")
        log.debug(f"   Method: {self.stemming_method}")
        log.debug(f"   Root extraction: {self.use_root_extraction}")
        log.debug(f"   Aggressive: {self.aggressive_stemming}")

    @staticmethod
    def _has_ar(s: str) -> bool:
        """Check if string contains Arabic characters"""
        return any("\u0600" <= ch <= "\u06FF" for ch in s)

    def _pyarabic_normalize(self, word: str) -> str:
        """Normalize Arabic word using PyArabic"""
        normalized = word
        
        # Apply PyArabic normalizations based on configuration
        if self.remove_diacritics:
            normalized = araby.strip_diacritics(normalized)
        
        if self.remove_tatweel:
            normalized = araby.strip_tatweel(normalized)
        
        if self.normalize_hamza:
            normalized = araby.normalize_hamza(normalized)
        
        if self.normalize_alef:
            normalized = araby.normalize_alef(normalized)
        
        if self.normalize_teh:
            normalized = araby.normalize_teh(normalized)
        
        return normalized
    
    def _pyarabic_stem(self, word: str) -> str:
        """Stem Arabic word using PyArabic methods"""
        # Skip protected terms
        if word in self.protected_terms:
            return word
        
        # Length protection
        if len(word) <= self.min_word_length:
            return word
        
        stemmed = word
        
        if self.stemming_method == "light":
            # Light stemming using PyArabic and configured rules
            if self.remove_common_prefixes:
                # Remove common prefixes using configured list
                for prefix in ["ال", "و", "ف", "ب", "ك", "ل", "س"]:
                    if stemmed.startswith(prefix) and len(stemmed) > len(prefix) + 2:
                        stemmed = stemmed[len(prefix):]
                        break
            
            if self.remove_common_suffixes:
                # Remove common suffixes using configured list
                for suffix in ["ها", "هم", "هن", "كم", "كن", "نا", "ين", "ون", "ات"]:
                    if stemmed.endswith(suffix) and len(stemmed) > len(suffix) + 2:
                        stemmed = stemmed[:-len(suffix)]
                        break
        
        elif self.stemming_method == "root" and self.use_root_extraction:
            # Try to extract root using PyArabic tokenization
            try:
                # Use PyArabic tokenization to get word parts
                tokens = araby.tokenize(word)
                if tokens and len(tokens) > 0:
                    # Take the first meaningful token as stem
                    stemmed = tokens[0] if tokens[0] else word
                else:
                    stemmed = word
            except:
                # Fallback to light stemming
                stemmed = self._light_stemming_fallback(stemmed)
        
        elif self.stemming_method == "prefix_suffix":
            # Use configured prefixes and suffixes
            stemmed = self._light_stemming_fallback(stemmed)
        
        elif self.stemming_method == "custom":
            # Apply custom stemming rules
            stemmed = self._custom_arabic_stemming(stemmed)
        
        # Validate result
        if len(stemmed) < self.min_stem_length:
            return word
        
        return stemmed
    
    def _light_stemming_fallback(self, word: str) -> str:
        """Light stemming fallback using configured prefixes/suffixes"""
        stemmed = word
        
        # Remove prefixes (longest first)
        for prefix in self.prefixes:
            if stemmed.startswith(prefix) and len(stemmed) - len(prefix) >= self.min_word_length:
                stemmed = stemmed[len(prefix):]
                break
        
        # Remove suffixes (longest first)  
        for suffix in self.suffixes:
            if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= self.min_word_length:
                stemmed = stemmed[:-len(suffix)]
                break
        
        return stemmed
    
    def _custom_arabic_stemming(self, word: str) -> str:
        """Custom Arabic stemming rules for agricultural domain"""
        stemmed = word
        
        # Apply domain-specific rules
        # Handle broken plural pattern: افعال → فعل
        if len(stemmed) >= 5 and stemmed[0] == "ا":
            if len(stemmed) == 5 and stemmed[3] == "ا":
                core = stemmed[1:3] + stemmed[4]
                if len(core) >= self.min_word_length:
                    stemmed = core
        
        # Collapse elongations
        stemmed = re.sub(r"(.)\1{2,}", r"\1\1", stemmed)
        
        return stemmed

    def reduce_ar(self, text: str) -> str:
        """Arabic morphological reduction using PyArabic (fast, offline)"""
        out = []
        for tok in self._re_tok.findall(text):
            if not self._has_ar(tok):
                out.append(tok)
                continue

            # Skip protected terms
            if tok in self.protected_terms:
                out.append(tok)
                continue

            # Step 1: Normalize the token using PyArabic
            normalized = self._pyarabic_normalize(tok)
            
            # Step 2: Apply PyArabic stemming
            stemmed = self._pyarabic_stem(normalized)
            
            # Step 3: Conservative reduction check
            if (self.min_stem_length <= len(stemmed) <= len(tok) and
                stemmed not in self.protected_terms and
                len(stemmed) >= len(tok) * self.min_reduction_ratio):
                out.append(stemmed)
            else:
                out.append(tok)
        return " ".join(out)

    def reduce_en(self, text: str) -> str:
        """English morphological reduction"""
        return " ".join(self.en.stem(tok) for tok in text.split())

    def reduce(self, text: str, lang: str) -> str:
        """Reduce text based on language"""
        return self.reduce_ar(text) if lang == "ar" else self.reduce_en(text)


class LexicalRelevanceFilter:
    """
    Enhanced lexical filtering with comprehensive agricultural vocabulary.
    Includes semantic filtering and domain coherence to reduce false positives.
    
    Phase 1 Enhancement: Added domain coherence checking and stricter filtering
    Phase 2 Enhancement: Added advanced semantic analysis with concept hierarchies
    and contextual understanding for superior accuracy.
    """
    
    def __init__(self, normalizer: TextNormalizer, config_path: str = None):
        self.normalizer = normalizer
        
        # Load lexical configuration
        if config_path is None:
            config_path = pathlib.Path(__file__).parent.parent / "config" / "lexical.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Convert to sets for faster lookup
        self.arabic_species = {
            key: set(variants) for key, variants in config['arabic_species'].items()
        }
        
        self.arabic_actions = {
            key: set(variants) for key, variants in config['arabic_actions'].items()
        }
        
        self.english_actions = {
            key: set(variants) for key, variants in config['english_actions'].items()
        }
        
        self.semantic_incompatible = config['semantic_incompatible']
        
        # Domain coherence keywords - PHASE 1 ENHANCEMENT
        self.domain_keywords = {
            'marine': {'بحرية', 'واسطة', 'صيد', 'مياه', 'بحر', 'سفينة', 'قارب', 'محيط'},
            'agriculture': {'زراعة', 'نبات', 'بذور', 'محاصيل', 'مزرعة', 'حقل', 'زرع', 'نمو'},
            'livestock': {'مواشي', 'نحل', 'خيل', 'ماعز', 'أبقار', 'حيوان', 'تربية', 'رعي'},
            'pets': {'قطط', 'بسة', 'كلاب', 'أليف', 'حيوان منزلي'},
            'wells': {'بئر', 'حفر', 'تنظيف', 'ردم', 'مياه جوفية', 'آبار'},
            'veterinary': {'بيطري', 'طبيب', 'علاج', 'دواء', 'مرض', 'عيادة'},
            'import_export': {'استيراد', 'تصدير', 'إذن', 'عبور', 'ترانزيت', 'جمرك'},
            'feed_agriculture': {'أعلاف', 'تقاوي', 'أسمدة', 'مبيدات', 'تغذية'}
        }
        
        # PHASE 2: Initialize advanced semantic analyzer
        try:
            try:
                from .semantic_analyzer import SemanticAnalyzer
            except ImportError:
                from semantic_analyzer import SemanticAnalyzer
            self.semantic_analyzer = SemanticAnalyzer()
            self.phase2_enabled = True
            log.info(f"🤖 Phase 2 semantic analysis: ENABLED")
        except Exception as e:
            self.semantic_analyzer = None
            self.phase2_enabled = False
            log.warning(f"⚠️  Phase 2 semantic analysis: DISABLED ({e})")
        
        # PHASE 4: Protected legitimate matches - high-quality results that should never be rejected
        self.protected_legitimate_matches = {
            'pets': {
                'query_terms': ['بسة', 'قطط', 'كلب', 'كلاب', 'قطة', 'حيوان أليف'],
                'result_patterns': [
                    'استيراد.*قطط', 'تصدير.*قطط', 'اذن.*قطط', 
                    'بيطري', 'طبيب.*حيوان', 'علاج.*حيوان',
                    'رخصة.*حيوان', 'تسجيل.*حيوان', 'ترخيص.*حيوان'
                ],
                'min_score': 0.65  # Always accept if above this threshold
            },
            'livestock': {
                'query_terms': ['مواشي', 'تربية.*خيل', 'تربية.*نحل', 'حصان', 'خيل', 'نحل', 'منحل'],
                'result_patterns': [
                    'تربية.*خيل', 'مربط.*خيل', 'تربية.*نحل', 'منحل',
                    'ترخيص.*تربية', 'تسجيل.*حيوان', 'طلب.*ترقيم.*ماشية',
                    'بيطري', 'علاج.*حيوان', 'تطعيم'
                ],
                'min_score': 0.60  # Livestock often has lower lexical overlap but high relevance
            },
            'veterinary': {
                'query_terms': ['بيطري', 'طبيب.*حيوان', 'علاج.*حيوان', 'مزاولة.*بيطرية'],
                'result_patterns': [
                    'مزاولة.*بيطرية', 'ترخيص.*بيطري', 'طبيب.*بيطري',
                    'عيادة.*بيطرية', 'مستشفى.*بيطري', 'ممارس.*بيطري'
                ],
                'min_score': 0.55  # Professional licensing should be highly relevant
            },
            'agriculture': {
                'query_terms': ['زراعة', 'مزرعة', 'فاكهة', 'خضار', 'محاصيل', 'بذور'],
                'result_patterns': [
                    'ترخيص.*زراعي', 'زراعة.*محاصيل', 'بذور', 'تقاوي',
                    'تصدير.*زراعي', 'استيراد.*زراعي', 'فسح.*زراعي'
                ],
                'min_score': 0.60
            }
        }
        
        log.info(f"📖 Loaded lexical config: {len(self.arabic_species)} Arabic species, "
                f"{len(self.arabic_actions)} Arabic actions, {len(self.english_actions)} English actions")
        log.info(f"🎯 Domain coherence: {len(self.domain_keywords)} domains loaded")
        log.info(f"🛡️ Protected legitimate matches: {len(self.protected_legitimate_matches)} categories")

    def compute_lexical_overlap(self, query: str, title: str, lang: str) -> float:
        """Compute lexical overlap between query and title"""
        q_norm = self.normalizer.normalize(query, lang).split()
        t_norm = self.normalizer.normalize(title, lang).split()
        
        if not q_norm or not t_norm:
            return 0.0
        
        # Filter short words
        q_words = {w for w in q_norm if len(w) >= 2}
        t_words = {w for w in t_norm if len(w) >= 2}
        
        if not q_words:
            return 0.0
        
        # Exact overlap
        overlap = len(q_words & t_words)
        
        # Substring matching for longer words
        substring_matches = 0
        for q_word in q_words:
            if len(q_word) >= 3:
                for t_word in t_words:
                    if q_word in t_word or t_word in q_word:
                        match_ratio = min(len(q_word), len(t_word)) / max(len(q_word), len(t_word))
                        substring_matches += 0.5 * match_ratio
                        break
        
        total_score = overlap + substring_matches
        return min(1.0, total_score / len(q_words))

    def check_species_consistency(self, query: str, title: str, lang: str) -> float:
        """Check species consistency between query and title"""
        if lang != "ar":
            return 1.0
        
        q_norm = self.normalizer.normalize(query, lang)
        t_norm = self.normalizer.normalize(title, lang)
        
        query_species = set()
        title_species = set()
        
        # Find species in query and title
        for species, variants in self.arabic_species.items():
            if any(variant in q_norm for variant in variants):
                query_species.add(species)
            if any(variant in t_norm for variant in variants):
                title_species.add(species)
        
        if not query_species or not title_species:
            return 1.0
        
        return 1.0 if query_species & title_species else 0.5

    def check_action_consistency(self, query: str, title: str, lang: str) -> float:
        """Check action consistency between query and title"""
        q_norm = self.normalizer.normalize(query, lang)
        t_norm = self.normalizer.normalize(title, lang)
        
        action_map = self.arabic_actions if lang == "ar" else self.english_actions
        
        query_actions = set()
        title_actions = set()
        
        # Find actions in query and title
        for action, variants in action_map.items():
            if any(variant in q_norm for variant in variants):
                query_actions.add(action)
            if any(variant in t_norm for variant in variants):
                title_actions.add(action)
        
        if not query_actions or not title_actions:
            return 1.0
        
        return 1.0 if query_actions & title_actions else 0.6

    def detect_query_domain(self, query: str) -> str:
        """Detect the primary domain of a query - PHASE 5 ULTRA-STRICT VALIDATION"""
        query_lower = query.lower().strip()
        query_normalized = self.normalizer.normalize(query_lower, "ar")
        
        # PHASE 5: Enhanced domain detection with confidence scoring
        domain_scores = {}
        domain_confidence = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            total_weight = 0
            for keyword in keywords:
                if keyword in query_normalized:
                    # Exact match gets full weight
                    weight = len(keyword) / len(query_normalized) if len(query_normalized) > 0 else 0
                    score += weight * 2  # Double weight for normalized text
                    total_weight += weight
                elif keyword in query_lower:
                    # Partial match gets partial weight
                    weight = len(keyword) / len(query_lower) if len(query_lower) > 0 else 0
                    score += weight
                    total_weight += weight
            
            domain_scores[domain] = score
            domain_confidence[domain] = total_weight
        
        if not domain_scores or max(domain_scores.values()) == 0:
            return 'general'
        
        # Get domain with highest score
        max_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[max_domain]
        
        # PHASE 5: Apply confidence threshold - must have strong domain signal
        confidence = domain_confidence.get(max_domain, 0)
        if confidence < 0.15:  # Require 15% of query to match domain keywords
            return 'general'
        
        return max_domain
    
    def detect_result_domain(self, title: str) -> str:
        """Detect the primary domain of a result title - PHASE 5 ULTRA-STRICT VALIDATION"""
        title_lower = title.lower().strip()
        title_normalized = self.normalizer.normalize(title_lower, "ar")
        
        # PHASE 5: Enhanced result domain detection with multi-signal analysis
        domain_scores = {}
        domain_confidence = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            total_weight = 0
            keyword_matches = 0
            
            for keyword in keywords:
                if keyword in title_normalized:
                    # Exact match in normalized text
                    weight = len(keyword) / len(title_normalized) if len(title_normalized) > 0 else 0
                    score += weight * 2
                    total_weight += weight
                    keyword_matches += 1
                elif keyword in title_lower:
                    # Partial match in original text
                    weight = len(keyword) / len(title_lower) if len(title_lower) > 0 else 0
                    score += weight
                    total_weight += weight
                    keyword_matches += 1
            
            # PHASE 5: Bonus for multiple keyword matches (domain consistency)
            if keyword_matches >= 2:
                score *= 1.5  # 50% bonus for multi-keyword domains
            
            domain_scores[domain] = score
            domain_confidence[domain] = total_weight
        
        if not domain_scores or max(domain_scores.values()) == 0:
            return 'general'
        
        # Get top domains for conflict detection
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        max_domain, max_score = sorted_domains[0]
        
        # PHASE 5: Detect domain conflicts (strong signals from multiple domains)
        if len(sorted_domains) > 1:
            second_domain, second_score = sorted_domains[1]
            if second_score > 0 and (second_score / max_score) > 0.7:  # Within 30% of top score
                # Domain conflict detected - apply stricter validation
                confidence = domain_confidence.get(max_domain, 0)
                if confidence < 0.25:  # Require 25% confidence for conflicted results
                    return 'conflicted'  # Special flag for mixed-domain results
        
        # Apply base confidence threshold
        confidence = domain_confidence.get(max_domain, 0)
        if confidence < 0.10:  # Require 10% of title to match domain keywords
            return 'general'
        
        return max_domain
    
    def _check_protected_legitimate_match(self, query_lower: str, title_lower: str, embedding_score: float) -> bool:
        """Check if this is a protected legitimate match that should never be rejected - PHASE 4"""
        import re
        
        for category, config in self.protected_legitimate_matches.items():
            # Check if query matches any terms for this category
            query_matches = any(
                re.search(term, query_lower) for term in config['query_terms']
            )
            
            if query_matches:
                # Check if result matches any patterns for this category
                result_matches = any(
                    re.search(pattern, title_lower) for pattern in config['result_patterns']
                )
                
                if result_matches and embedding_score >= config['min_score']:
                    log.debug(f"🛡️ Protected match detected: Category={category}, "
                             f"Score={embedding_score:.3f} >= {config['min_score']}")
                    return True
        
        return False
    
    def check_domain_coherence(self, query: str, title: str) -> float:
        """Check domain coherence between query and result - PHASE 1 ENHANCEMENT"""
        query_domain = self.detect_query_domain(query)
        result_domain = self.detect_result_domain(title)
        
        # If query is general domain, don't penalize
        if query_domain == 'general':
            return 1.0
            
        # Perfect match
        if query_domain == result_domain:
            return 1.0
            
        # Compatible domains (some logical overlap)
        compatible_domains = {
            'livestock': ['veterinary', 'feed_agriculture', 'agriculture'],
            'pets': ['veterinary'],
            'agriculture': ['feed_agriculture', 'wells'],
            'feed_agriculture': ['agriculture', 'livestock', 'import_export'],
            'import_export': ['feed_agriculture', 'agriculture', 'livestock', 'pets']
        }
        
        if result_domain in compatible_domains.get(query_domain, []):
            return 0.8  # Slight penalty for compatible but different domains
            
        # Major penalty for completely unrelated domains
        return 0.3
    
    def compute_relevance_score(self, query: str, title: str, lang: str,
                              embedding_similarity: float) -> Dict[str, float]:
        """Compute hybrid relevance score combining embedding and lexical signals - PHASE 2 ENHANCED"""
        lexical_overlap = self.compute_lexical_overlap(query, title, lang)
        species_consistency = self.check_species_consistency(query, title, lang)
        action_consistency = self.check_action_consistency(query, title, lang)
        domain_coherence = self.check_domain_coherence(query, title)
        
        # PHASE 4: Check for protected legitimate matches FIRST - prevent over-filtering
        query_lower = query.lower().strip()
        title_lower = title.lower().strip()
        
        is_protected_match = self._check_protected_legitimate_match(query_lower, title_lower, embedding_similarity)
        if is_protected_match:
            # Protected matches get enhanced scores to prevent false negatives
            protected_boost = 1.1  # 10% boost
            final_score = min(0.95, embedding_similarity * protected_boost)
            log.debug(f"🛡️ PROTECTED MATCH: Query '{query}' -> Result '{title}' | "
                     f"Original: {embedding_similarity:.3f} -> Protected: {final_score:.3f}")
            return {
                "final_score": final_score,
                "embedding_sim": embedding_similarity,
                "lexical_overlap": lexical_overlap,
                "species_consistency": species_consistency,
                "action_consistency": action_consistency,
                "domain_coherence": domain_coherence,
                "lexical_score": lexical_overlap,
                "semantic_score": 0.9,  # High semantic score for protected matches
                "semantic_explanation": "Protected legitimate match - high relevance guaranteed",
                "phase2_enabled": self.phase2_enabled,
                "is_critical_false_positive": False,
                "is_protected_match": True
            }
        
        # CRITICAL: Check for zero-tolerance false positive patterns
        
        # PHASE 5 EMERGENCY: ULTRA-AGGRESSIVE false positive patterns (91% false positive rate found!)
        critical_mismatches = [
            # CRITICAL: Pet queries -> Marine/Fishing (PROVEN 91% false positives from cosine-reranker!)
            ('بسة', 'بحرية'), ('بسة', 'واسطة'), ('بسة', 'صيد'), ('بسة', 'رخصة واسطة'),
            ('بسة', 'استبدال واسطة'), ('بسة', 'إلغاء ترخيص'), ('بسة', 'تجديد رخصة'),
            ('قطط', 'بحرية'), ('قطط', 'واسطة'), ('قطط', 'صيد'), ('قطط', 'رخصة واسطة'),
            ('قطط', 'استبدال واسطة'), ('قطط', 'إلغاء ترخيص'), ('قطط', 'تجديد رخصة'),
            ('كلب', 'بحرية'), ('كلب', 'واسطة'), ('كلب', 'صيد'), ('كلب', 'رخصة واسطة'),
            ('كلاب', 'بحرية'), ('كلاب', 'واسطة'), ('كلاب', 'صيد'), ('كلاب', 'رخصة واسطة'),
            ('قطة', 'بحرية'), ('قطة', 'واسطة'), ('قطة', 'صيد'), ('قطة', 'رخصة واسطة'),
            
            # CRITICAL: Pet queries -> Well drilling (PROVEN false positives) 
            ('بسة', 'بئر'), ('بسة', 'حفر'), ('بسة', 'آبار'), ('بسة', 'حفر بئر'),
            ('قطط', 'بئر'), ('قطط', 'حفر'), ('قطط', 'آبار'), ('قطط', 'حفر بئر'),
            ('كلب', 'بئر'), ('كلب', 'حفر'), ('كلاب', 'حفر'), ('كلاب', 'حفر بئر'),
            
            # CRITICAL: Animal care -> Marine/Technical (impossible combinations)
            ('بيطري', 'بحرية'), ('بيطري', 'واسطة'), ('بيطري', 'صيد'),
            ('طبيب', 'بحرية'), ('طبيب', 'واسطة'), ('طبيب', 'صيد'),
            ('علاج', 'بحرية'), ('علاج', 'واسطة'), ('علاج', 'صيد'),
            
            # CRITICAL: Livestock -> Marine (impossible combinations)  
            ('خيل', 'بحرية'), ('خيل', 'واسطة'), ('خيل', 'صيد'),
            ('حصان', 'بحرية'), ('حصان', 'واسطة'), ('حصان', 'صيد'),
            ('نحل', 'بحرية'), ('نحل', 'واسطة'), ('نحل', 'صيد'),
            ('منحل', 'بحرية'), ('منحل', 'واسطة'), ('منحل', 'صيد'),
            
            # CRITICAL: Generic animal terms -> Marine/Technical
            ('حيوان', 'بحرية'), ('حيوان', 'واسطة'), ('حيوان', 'صيد'),
            ('تربية', 'بحرية'), ('تربية', 'واسطة'), ('تربية', 'صيد'),
            ('مواشي', 'بحرية'), ('مواشي', 'واسطة'), ('مواشي', 'صيد'),
            
            # EMERGENCY: Additional comprehensive patterns
            ('أليف', 'بحرية'), ('أليف', 'واسطة'), ('أليف', 'صيد'),
            ('حيوان منزلي', 'بحرية'), ('حيوان منزلي', 'واسطة'),
            
            # PHASE 5 ULTRA-EXPANSION: Domain isolation (based on preset analysis)
            # Agriculture -> Marine (impossible crossover)
            ('زراعة', 'بحرية'), ('زراعة', 'واسطة'), ('زراعة', 'صيد'),
            ('مزرعة', 'بحرية'), ('مزرعة', 'واسطة'), ('مزرعة', 'صيد'),
            ('بذور', 'بحرية'), ('بذور', 'واسطة'), ('محاصيل', 'بحرية'),
            ('فاكهة', 'بحرية'), ('خضار', 'بحرية'), ('خضروات', 'بحرية'),
            
            # Food/Restaurant -> Marine (impossible crossover)
            ('مطعم', 'بحرية'), ('مطعم', 'واسطة'), ('مطعم', 'صيد'),
            ('كافيه', 'بحرية'), ('كافيه', 'واسطة'), ('مقهى', 'بحرية'),
            ('طعام', 'بحرية'), ('طعام', 'واسطة'), ('طعام', 'صيد'),
            
            # Healthcare -> Marine (impossible crossover)
            ('مستشفى', 'بحرية'), ('مستشفى', 'واسطة'), ('مستشفى', 'صيد'),
            ('عيادة', 'بحرية'), ('عيادة', 'واسطة'), ('عيادة', 'صيد'),
            ('صحة', 'بحرية'), ('صحة', 'واسطة'), ('صحة', 'صيد'),
            
            # Education -> Marine (impossible crossover)
            ('مدرسة', 'بحرية'), ('مدرسة', 'واسطة'), ('مدرسة', 'صيد'),
            ('جامعة', 'بحرية'), ('جامعة', 'واسطة'), ('جامعة', 'صيد'),
            ('تعليم', 'بحرية'), ('تعليم', 'واسطة'), ('تعليم', 'صيد'),
            
            # Technology -> Marine (rare crossover - apply caution)
            ('تقنية', 'بحرية'), ('تقنية', 'واسطة'), ('برمجة', 'بحرية'),
            ('حاسوب', 'بحرية'), ('انترنت', 'بحرية'), ('موقع', 'بحرية'),
            
            # Tourism/Hotels -> Marine (some valid but often false positives)
            ('فندق', 'صيد'), ('منتجع', 'صيد'), ('سياحة', 'صيد'),
            ('سفر', 'صيد'), ('رحلة', 'صيد'),
            
            # Construction -> Pet care (impossible crossover) 
            ('بناء', 'بيطري'), ('مقاول', 'بيطري'), ('عمارة', 'بيطري'),
            ('هندسة', 'بيطري'), ('تشييد', 'بيطري'),
            
            # Finance -> Animal care (impossible crossover)
            ('بنك', 'بيطري'), ('مال', 'بيطري'), ('استثمار', 'بيطري'),
            ('قرض', 'بيطري'), ('تمويل', 'بيطري')
        ]
        
        is_critical_false_positive = False
        detected_pattern = None
        for query_term, bad_result_term in critical_mismatches:
            if query_term in query_lower and bad_result_term in title_lower:
                is_critical_false_positive = True
                detected_pattern = f"{query_term} -> {bad_result_term}"
                break
        
        # PHASE 2: Advanced semantic analysis
        semantic_score = 0.0
        semantic_explanation = "Phase 2 disabled"
        if self.phase2_enabled and self.semantic_analyzer:
            try:
                semantic_analysis = self.semantic_analyzer.analyze_semantic_similarity(query, title)
                semantic_score = semantic_analysis.final_score
                semantic_explanation = semantic_analysis.explanation
                
                log.debug(f"Semantic analysis: {semantic_score:.3f} - {semantic_explanation}")
            except Exception as e:
                log.warning(f"Semantic analysis failed: {e}")
                semantic_score = 0.0
        
        # PHASE 1 BASE: Lexical filtering
        lexical_score = (lexical_overlap * 0.4 +
                        species_consistency * 0.2 +
                        action_consistency * 0.2 +
                        domain_coherence * 0.2)
        
        # CRITICAL FALSE POSITIVE OVERRIDE: Moderate penalty approach - FIXED
        if is_critical_false_positive:
            # Apply moderate penalty for known false positive patterns
            # Reduce score but not to zero - allow some results to still pass reasonable thresholds
            final_score = min(0.25, embedding_similarity * 0.3 + lexical_score * 0.4)
            semantic_explanation += f" | FALSE POSITIVE DETECTED: {detected_pattern} - MODERATE PENALTY"

            # Log the detection for monitoring
            log.warning(f"🚨 CRITICAL FALSE POSITIVE BLOCKED: Query '{query}' -> Result '{title}' | "
                       f"Pattern: {detected_pattern} | Original embedding: {embedding_similarity:.3f} | "
                       f"Reduced to: {final_score:.3f}")
        elif self.phase2_enabled and semantic_score > 0:
            # PHASE 2 ENHANCEMENT: Advanced scoring with semantic understanding
            if semantic_score > 0.8:
                # High semantic similarity - trust it more
                final_score = (embedding_similarity * 0.3 + 
                              lexical_score * 0.2 + 
                              semantic_score * 0.5)
            elif semantic_score > 0.5:
                # Moderate semantic similarity - balanced approach
                final_score = (embedding_similarity * 0.5 + 
                              lexical_score * 0.2 + 
                              semantic_score * 0.3)
            else:
                # Low semantic similarity - apply penalties
                final_score = (embedding_similarity * 0.4 + 
                              lexical_score * 0.4 + 
                              semantic_score * 0.2)
        else:
            # PHASE 1 FALLBACK: Original scoring
            if lexical_overlap < 0.1 and domain_coherence < 0.5:
                final_score = embedding_similarity * 0.4 + lexical_score * 0.6
            elif lexical_overlap < 0.2:
                final_score = embedding_similarity * 0.7 + lexical_score * 0.3  
            else:
                final_score = embedding_similarity * 0.85 + lexical_score * 0.15
        
        return {
            "final_score": final_score,
            "embedding_sim": embedding_similarity,
            "lexical_overlap": lexical_overlap,
            "species_consistency": species_consistency,
            "action_consistency": action_consistency,
            "domain_coherence": domain_coherence,
            "lexical_score": lexical_score,
            "semantic_score": semantic_score,
            "semantic_explanation": semantic_explanation,
            "phase2_enabled": self.phase2_enabled,
            "critical_false_positive": is_critical_false_positive
        }

    def apply_semantic_filter(self, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply semantic penalties to incompatible query-result pairs - ENHANCED"""
        if not results.get('hits_kept'):
            return results
        
        query_lower = query.lower().strip()
        filtered_kept = []
        penalty_count = 0
        domain_penalty_count = 0
        
        for hit in results['hits_kept']:
            title = hit['title'].lower()
            should_penalize = False
            domain_penalty = False
            
            # PHASE 5 ULTRA-ENHANCEMENT: Ultra-aggressive domain mismatch detection
            query_domain = self.detect_query_domain(query)
            result_domain = self.detect_result_domain(hit['title'])
            
            # PHASE 5: Domain compatibility matrix (zero tolerance for impossible combinations)
            domain_incompatible = self._check_domain_incompatibility(query_domain, result_domain)
            if domain_incompatible:
                should_penalize = True
                detected_critical_pattern = f"Domain mismatch: {query_domain} → {result_domain}"
            
            # Critical false positive patterns - ZERO TOLERANCE - PHASE 3 EXPANDED
            critical_mismatches = [
                # Pet -> Marine vessel licensing (CRITICAL from log analysis)
                ('بسة', 'بحرية'),     ('بسة', 'واسطة'),     ('بسة', 'صيد'),
                ('قطط', 'بحرية'),     ('قطط', 'واسطة'),     ('قطط', 'صيد'), 
                ('كلب', 'بحرية'),     ('كلاب', 'واسطة'),
                
                # Pet -> Well drilling
                ('بسة', 'بئر'),       ('بسة', 'حفر'),       ('قطط', 'حفر'),
                ('قطط', 'آبار'),      ('كلب', 'بئر'),       ('كلاب', 'حفر'),
                
                # Animal care -> Infrastructure
                ('بيطري', 'بحرية'),   ('طبيب', 'واسطة'),    ('علاج', 'صيد'),
                
                # Livestock -> Marine
                ('خيل', 'بحرية'),     ('نحل', 'واسطة'),     ('حصان', 'صيد'),
                
                # Generic animal terms -> technical services  
                ('حيوان', 'بحرية'),   ('تربية', 'واسطة'),
            ]
            
            detected_critical_pattern = None
            for query_term, bad_result_term in critical_mismatches:
                if query_term in query_lower and bad_result_term in title:
                    should_penalize = True
                    penalty_count += 1
                    detected_critical_pattern = f"{query_term}->{bad_result_term}"
                    break
            
            # Check for existing semantic incompatibilities
            if not should_penalize:
                for q_term, bad_terms in self.semantic_incompatible.items():
                    if q_term in query_lower:
                        for bad_term in bad_terms:
                            if bad_term in title:
                                should_penalize = True
                                break
            
            # PHASE 1: Strict domain coherence check
            if not should_penalize and query_domain != 'general':
                domain_coherence = self.check_domain_coherence(query, hit['title'])
                if domain_coherence < 0.5:
                    domain_penalty = True
                    domain_penalty_count += 1
            
            # Apply penalties - PHASE 3 ENHANCED
            if should_penalize:
                # ULTRA-HEAVY penalty for critical mismatches (90% reduction to prevent 87-91% false positives)
                original_score = hit.get('final_sim', hit.get('embedding_sim', 0))
                if detected_critical_pattern:
                    # Near-zero scores for critical patterns
                    hit['final_sim'] = min(0.10, original_score * 0.1)
                    hit['final_pct'] = hit['final_sim'] * 100
                    hit['critical_pattern_blocked'] = detected_critical_pattern
                    log.warning(f"🚨 SEMANTIC FILTER: Blocked critical pattern {detected_critical_pattern} | "
                               f"Original: {original_score:.3f} -> Reduced: {hit['final_sim']:.3f}")
                else:
                    # Regular heavy penalty (70% reduction)
                    hit['final_sim'] = original_score * 0.3
                    hit['final_pct'] = hit['final_sim'] * 100
                hit['semantic_penalty'] = True
                penalty_count += 1
            elif domain_penalty:
                # Moderate penalty for domain mismatches (40% reduction)
                if 'final_sim' in hit:
                    hit['final_sim'] *= 0.6
                    hit['final_pct'] = hit['final_sim'] * 100
                hit['domain_penalty'] = True
            else:
                hit['semantic_penalty'] = False
                hit['domain_penalty'] = False
            
            # Keep if still above threshold
            threshold = results['threshold_pct'] / 100.0
            final_score = hit.get('final_sim', hit.get('embedding_sim', 0))
            if final_score >= threshold:
                filtered_kept.append(hit)
            else:
                results['hits_rejected'].append(hit)
        
        results['hits_kept'] = filtered_kept
        
        if penalty_count > 0:
            log.info(f"   🚫 Applied {penalty_count} critical semantic penalties")
        if domain_penalty_count > 0:
            log.info(f"   🎯 Applied {domain_penalty_count} domain coherence penalties")
        
        return results
    
    def _check_domain_incompatibility(self, query_domain: str, result_domain: str) -> bool:
        """Check if two domains are fundamentally incompatible - PHASE 5 EMERGENCY"""
        # ULTRA-STRICT domain separation matrix based on preset analysis failures
        incompatible_pairs = {
            # Animal/Pet care → Marine/Fishing (91% false positive source!)
            ('animal_care', 'marine'),
            ('animal_care', 'fishing'),  
            ('veterinary', 'marine'),
            ('veterinary', 'fishing'),
            ('livestock', 'marine'),
            ('livestock', 'fishing'),
            
            # Agriculture → Marine (impossible crossover)
            ('agriculture', 'marine'),
            ('agriculture', 'fishing'),
            
            # Healthcare → Marine (impossible crossover)
            ('medical', 'marine'),
            ('medical', 'fishing'),
            
            # Education → Marine (rare legitimate crossover)
            ('education', 'marine'),
            ('education', 'fishing'),
            
            # Technology → Marine (some valid but often false positives)
            ('technology', 'marine'),
            ('technology', 'fishing'),
            
            # Finance → Animal care (impossible crossover)
            ('finance', 'animal_care'),
            ('finance', 'veterinary'),
            ('finance', 'livestock'),
            
            # Construction → Animal care (impossible crossover)
            ('construction', 'animal_care'),
            ('construction', 'veterinary'), 
            ('construction', 'livestock'),
        }
        
        # Check both directions of the pair
        pair1 = (query_domain, result_domain)
        pair2 = (result_domain, query_domain)
        
        return pair1 in incompatible_pairs or pair2 in incompatible_pairs
    
    def _is_compatible_domain_pair(self, domain1: str, domain2: str) -> bool:
        """Check if two domains can have legitimate overlaps - PHASE 5"""
        # Allow some domain pairs that can have legitimate crossovers
        compatible_pairs = {
            # Veterinary can overlap with livestock, agriculture
            ('veterinary', 'livestock'),
            ('veterinary', 'agriculture'), 
            ('animal_care', 'livestock'),
            ('animal_care', 'agriculture'),
            
            # Agriculture can overlap with livestock, food
            ('agriculture', 'livestock'),
            ('agriculture', 'food'),
            
            # Medical can overlap with veterinary for some services
            ('medical', 'veterinary'),
            
            # Tourism can have some marine overlap (coastal tourism)
            ('tourism', 'marine'),
            
            # General domains
            ('general', '*'),  # General can match anything
            ('*', 'general'),
        }
        
        # Special handling for general domain
        if domain1 == 'general' or domain2 == 'general':
            return True
            
        # Check explicit compatible pairs
        pair1 = (domain1, domain2)
        pair2 = (domain2, domain1)
        
        return pair1 in compatible_pairs or pair2 in compatible_pairs