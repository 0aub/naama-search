#!/usr/bin/env python3
"""
PHASE 2: Advanced Semantic Analyzer
Sophisticated semantic similarity and concept hierarchy processing
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)

@dataclass
class SemanticScore:
    """Comprehensive semantic similarity score with breakdown"""
    conceptual_similarity: float
    hierarchical_distance: float
    intent_alignment: float
    contextual_relevance: float
    final_score: float
    explanation: str

class SemanticAnalyzer:
    """
    Advanced semantic analysis for Arabic queries and results
    
    Phase 2 Enhancement: Adds sophisticated semantic understanding
    beyond basic domain matching, with concept hierarchies and
    contextual relevance scoring.
    """
    
    def __init__(self):
        self.semantic_hierarchy = self._build_semantic_hierarchy()
        self.intent_patterns = self._build_intent_patterns()
        self.concept_embeddings = self._build_concept_embeddings()
        self.compatibility_matrix = self._build_compatibility_matrix()
        
        # SOLUTION 5: Add intent compatibility mapping to prevent legitimate service mismatches
        self.intent_compatibility = {
            ('breeding_production', 'licensing'): 0.9,  # Breeding permits are licensing services
            ('commercial_trade', 'licensing'): 0.9,     # Trade licenses are compatible
            ('general', 'licensing'): 0.8,              # General queries can match licenses
            ('breeding_production', 'services'): 0.85,  # Breeding services are compatible
            ('commercial_trade', 'services'): 0.85,     # Trade services are compatible
        }
        
        log.info(f"🧠 Semantic analyzer initialized:")
        log.info(f"   Concepts: {len(self.semantic_hierarchy)} categories")
        log.info(f"   Intent patterns: {len(self.intent_patterns)} types") 
        log.info(f"   Compatibility rules: {len(self.compatibility_matrix)} mappings")
        log.info(f"   Intent compatibility: {len(self.intent_compatibility)} mappings")
    
    def _build_semantic_hierarchy(self) -> Dict[str, Dict[str, List[str]]]:
        """Build comprehensive semantic concept hierarchy"""
        return {
            'animals': {
                'pets': [
                    'قطط', 'بسة', 'بس', 'قطة', 'هرة',  # cats
                    'كلاب', 'كلب', 'جرو',  # dogs
                    'أرانب', 'أرنب',  # rabbits
                    'طيور الزينة', 'عصافير', 'ببغاء',  # pet birds
                    'أسماك الزينة', 'سمك زينة'  # ornamental fish
                ],
                'livestock': [
                    'مواشي', 'ماشية', 'بهائم',  # general livestock
                    'أبقار', 'بقر', 'بقرة', 'عجل', 'ثور',  # cattle
                    'ماعز', 'معز', 'عنز', 'جدي',  # goats
                    'خراف', 'غنم', 'نعاج', 'حملان',  # sheep
                    'إبل', 'جمال', 'ناقة', 'جمل'  # camels
                ],
                'working_animals': [
                    'خيل', 'خيول', 'حصان', 'فرس', 'مهر',  # horses
                    'حمير', 'حمار',  # donkeys
                    'بغال', 'بغل'  # mules
                ],
                'poultry': [
                    'دواجن', 'فراخ', 'دجاج', 'ديك',  # chickens
                    'بط', 'وز', 'رومي',  # ducks, geese, turkey
                    'حمام', 'يمام'  # pigeons, doves
                ],
                'bees_insects': [
                    'نحل', 'نحلة', 'عسل', 'منحل', 'خلية',  # bees
                    'دود القز', 'حرير'  # silkworms
                ],
                'wildlife': [
                    'صقور', 'صقر', 'باز', 'نسر',  # birds of prey
                    'غزلان', 'غزال', 'ظبي',  # deer, gazelles
                    'أسماك', 'سمك', 'أسماك برية'  # wild fish
                ]
            },
            'agriculture': {
                'crops': [
                    'محاصيل', 'زراعة', 'زرع', 'حقول', 'مزارع',  # general crops
                    'خضار', 'خضروات', 'خضراوات',  # vegetables
                    'فواكه', 'فاكهة', 'ثمار',  # fruits
                    'حبوب', 'قمح', 'شعير', 'ذرة', 'أرز',  # grains
                    'نباتات عطرية', 'أعشاب', 'توابل'  # herbs, spices
                ],
                'supplies': [
                    'أعلاف', 'علف', 'تبن', 'شعير علف',  # animal feed
                    'بذور', 'تقاوي', 'شتلات', 'أشتال',  # seeds, seedlings
                    'أسمدة', 'سماد', 'كومبوست',  # fertilizers
                    'مبيدات', 'مبيد', 'مكافحة آفات'  # pesticides
                ],
                'equipment': [
                    'جرارات', 'جرار', 'تراكتور',  # tractors
                    'آلات زراعية', 'معدات', 'أدوات',  # farm equipment
                    'أنظمة الري', 'ري', 'رش'  # irrigation
                ],
                'structures': [
                    'بيوت محمية', 'صوب', 'بيت بلاستيكي',  # greenhouses
                    'مخازن', 'صوامع', 'مستودعات',  # storage
                    'حظائر', 'زرائب', 'إسطبلات'  # animal shelters
                ]
            },
            'services': {
                'licensing': [
                    'رخصة', 'ترخيص', 'تصريح', 'إذن',  # licenses, permits
                    'اعتماد', 'تسجيل', 'شهادة',  # certification, registration
                    'تجديد', 'إلغاء', 'نقل ملكية'  # renewal, cancellation, transfer
                ],
                'veterinary': [
                    'بيطري', 'طبيب بيطري', 'عيادة بيطرية',  # veterinary
                    'علاج', 'دواء', 'أدوية بيطرية',  # treatment, medicine
                    'تطعيم', 'لقاح', 'تحصين',  # vaccination
                    'فحص', 'تشخيص', 'كشف'  # examination, diagnosis
                ],
                'trade': [
                    'بيع', 'شراء', 'تجارة', 'تسويق',  # trading
                    'استيراد', 'تصدير', 'نقل', 'شحن',  # import, export
                    'مزاد', 'مزادات', 'مناقصات',  # auctions, tenders
                    'توريد', 'توزيع', 'تداول'  # supply, distribution
                ],
                'infrastructure': [
                    'بئر', 'آبار', 'حفر', 'تنظيف بئر',  # wells
                    'مياه', 'مياه جوفية', 'خزانات',  # water systems
                    'كهرباء', 'طاقة', 'ألواح شمسية',  # electricity, energy
                    'طرق', 'مواصلات', 'نقل'  # roads, transportation
                ]
            },
            'marine': {
                'vessels': [
                    'واسطة بحرية', 'قارب', 'سفينة', 'زورق',  # vessels
                    'مركب', 'لنش', 'يخت', 'عبارة'  # boats, yacht, ferry
                ],
                'fishing': [
                    'صيد', 'صياد', 'شباك', 'قوارب صيد',  # fishing
                    'سمك', 'أسماك بحرية', 'مأكولات بحرية',  # fish, seafood
                    'محار', 'روبيان', 'جمبري'  # shellfish, shrimp
                ],
                'licenses': [
                    'رخصة بحرية', 'تصريح إبحار', 'رخصة صيد',  # marine licenses
                    'رخصة قبطان', 'شهادة بحرية'  # captain's license
                ]
            }
        }
    
    def _build_intent_patterns(self) -> Dict[str, List[str]]:
        """Build query intent classification patterns"""
        return {
            'licensing': [
                'رخصة', 'إذن', 'تصريح', 'ترخيص', 'اعتماد',
                'تسجيل', 'شهادة', 'موافقة', 'تجديد', 'إلغاء'
            ],
            'import_export': [
                'استيراد', 'تصدير', 'جمرك', 'عبور', 'ترانزيت',
                'شحن', 'نقل', 'فسح', 'إفراج'
            ],
            'breeding_production': [
                'تربية', 'إنتاج', 'تكاثر', 'تناسل', 'زراعة',
                'غرس', 'بذر', 'زرع'
            ],
            'medical_treatment': [
                'علاج', 'تطعيم', 'فحص', 'تشخيص', 'دواء',
                'طب', 'صحة', 'مرض', 'وقاية'
            ],
            'commercial_trade': [
                'بيع', 'شراء', 'تجارة', 'تسويق', 'توريد',
                'توزيع', 'تداول', 'مزاد'
            ],
            'maintenance_service': [
                'صيانة', 'تنظيف', 'إصلاح', 'ترميم', 'تطوير',
                'تحسين', 'خدمة', 'رعاية'
            ]
        }
    
    def _build_concept_embeddings(self) -> Dict[str, List[str]]:
        """Build concept-to-keyword mappings for semantic similarity"""
        concept_map = {}
        for category, subcategories in self.semantic_hierarchy.items():
            for subcategory, keywords in subcategories.items():
                concept_key = f"{category}_{subcategory}"
                concept_map[concept_key] = keywords
        return concept_map
    
    def _build_compatibility_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build compatibility scoring matrix between different concepts"""
        return {
            # High compatibility (0.9-1.0)
            ('animals_pets', 'services_veterinary'): 0.95,
            ('animals_livestock', 'services_veterinary'): 0.90,
            ('animals_poultry', 'services_veterinary'): 0.90,
            ('agriculture_supplies', 'animals_livestock'): 0.95,
            ('agriculture_supplies', 'animals_poultry'): 0.90,
            ('agriculture_crops', 'agriculture_supplies'): 0.95,
            ('services_licensing', 'animals_pets'): 0.85,
            ('services_licensing', 'animals_livestock'): 0.85,
            ('services_licensing', 'agriculture_crops'): 0.85,
            
            # Medium compatibility (0.6-0.8)
            ('animals_livestock', 'agriculture_equipment'): 0.70,
            ('agriculture_crops', 'services_trade'): 0.75,
            ('animals_pets', 'services_trade'): 0.65,
            ('services_veterinary', 'services_licensing'): 0.70,
            ('agriculture_supplies', 'services_import_export'): 0.80,
            
            # Low compatibility (0.3-0.5)
            ('animals_pets', 'marine_vessels'): 0.20,  # Phase 1 identified issue
            ('animals_pets', 'services_infrastructure'): 0.30,
            ('agriculture_crops', 'marine_fishing'): 0.40,
            ('animals_livestock', 'marine_vessels'): 0.25,
            
            # Very low compatibility (0.1-0.2)
            ('animals_pets', 'marine_licenses'): 0.15,  # Critical false positive
            ('animals_bees_insects', 'marine_vessels'): 0.10,
            ('agriculture_supplies', 'marine_vessels'): 0.15,
        }
    
    def extract_concepts(self, text: str) -> Dict[str, float]:
        """Extract semantic concepts from text with confidence scores"""
        text_lower = text.lower()
        concepts = {}
        
        for category, subcategories in self.semantic_hierarchy.items():
            for subcategory, keywords in subcategories.items():
                concept_key = f"{category}_{subcategory}"
                score = 0.0
                matches = 0
                
                for keyword in keywords:
                    if keyword in text_lower:
                        # Exact match gets higher score
                        if text_lower == keyword:
                            score += 1.0
                        # Word boundary match
                        elif re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                            score += 0.8
                        # Substring match
                        else:
                            score += 0.3
                        matches += 1
                
                if score > 0:
                    # Normalize score based on number of keywords in concept
                    normalized_score = min(1.0, score / len(keywords) * 2)
                    concepts[concept_key] = normalized_score
        
        return concepts
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify query intent with confidence score"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if pattern in query_lower:
                    if re.search(rf'\b{re.escape(pattern)}\b', query_lower):
                        score += 1.0  # Word boundary match
                    else:
                        score += 0.5  # Substring match
            
            if score > 0:
                intent_scores[intent_type] = score / len(patterns)
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            return best_intent, confidence
        
        return 'general', 0.0
    
    def calculate_conceptual_similarity(self, query_concepts: Dict[str, float], 
                                     result_concepts: Dict[str, float]) -> float:
        """Calculate conceptual similarity between query and result concepts"""
        if not query_concepts or not result_concepts:
            return 0.0
        
        max_similarity = 0.0
        
        # Direct concept matches
        for q_concept, q_score in query_concepts.items():
            if q_concept in result_concepts:
                r_score = result_concepts[q_concept]
                similarity = min(q_score, r_score)
                max_similarity = max(max_similarity, similarity)
        
        # Compatible concept matches
        for q_concept, q_score in query_concepts.items():
            for r_concept, r_score in result_concepts.items():
                compatibility_key = (q_concept, r_concept)
                reverse_key = (r_concept, q_concept)
                
                if compatibility_key in self.compatibility_matrix:
                    compatibility = self.compatibility_matrix[compatibility_key]
                    similarity = min(q_score, r_score) * compatibility
                    max_similarity = max(max_similarity, similarity)
                elif reverse_key in self.compatibility_matrix:
                    compatibility = self.compatibility_matrix[reverse_key]
                    similarity = min(q_score, r_score) * compatibility
                    max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def calculate_hierarchical_distance(self, query_concepts: Dict[str, float],
                                      result_concepts: Dict[str, float]) -> float:
        """Calculate hierarchical distance between concepts"""
        if not query_concepts or not result_concepts:
            return 1.0  # Maximum distance
        
        min_distance = 1.0
        
        for q_concept in query_concepts:
            q_category, q_subcategory = q_concept.split('_', 1)
            
            for r_concept in result_concepts:
                r_category, r_subcategory = r_concept.split('_', 1)
                
                if q_category == r_category:
                    if q_subcategory == r_subcategory:
                        distance = 0.0  # Same subcategory
                    else:
                        distance = 0.3  # Same category, different subcategory
                else:
                    distance = 0.8  # Different category
                
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def analyze_semantic_similarity(self, query: str, result_title: str) -> SemanticScore:
        """Comprehensive semantic similarity analysis"""
        
        # Extract concepts from both texts
        query_concepts = self.extract_concepts(query)
        result_concepts = self.extract_concepts(result_title)
        
        # Classify query intent
        intent, intent_confidence = self.classify_intent(query)
        
        # Calculate various similarity metrics
        conceptual_sim = self.calculate_conceptual_similarity(query_concepts, result_concepts)
        hierarchical_dist = self.calculate_hierarchical_distance(query_concepts, result_concepts)
        hierarchical_sim = 1.0 - hierarchical_dist
        
        # Intent alignment (how well result matches query intent)
        intent_alignment = self._calculate_intent_alignment(intent, result_concepts, result_title)
        
        # Contextual relevance (considering query and result context)
        contextual_relevance = self._calculate_contextual_relevance(
            query, result_title, query_concepts, result_concepts
        )
        
        # Final weighted score
        final_score = (
            conceptual_sim * 0.35 +
            hierarchical_sim * 0.25 +
            intent_alignment * 0.25 +
            contextual_relevance * 0.15
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            conceptual_sim, hierarchical_sim, intent_alignment, 
            contextual_relevance, query_concepts, result_concepts, intent
        )
        
        return SemanticScore(
            conceptual_similarity=conceptual_sim,
            hierarchical_distance=hierarchical_dist,
            intent_alignment=intent_alignment,
            contextual_relevance=contextual_relevance,
            final_score=final_score,
            explanation=explanation
        )
    
    def _calculate_intent_alignment(self, intent: str, result_concepts: Dict[str, float], 
                                  result_title: str) -> float:
        """Calculate how well result aligns with query intent"""
        if intent == 'general':
            return 0.5  # Neutral for general queries
        
        # SOLUTION 5: First check for intent compatibility to avoid false mismatches
        result_intent = self._classify_result_intent(result_title)
        compatibility_key = (intent, result_intent)
        
        if compatibility_key in self.intent_compatibility:
            return self.intent_compatibility[compatibility_key]
        
        title_lower = result_title.lower()
        
        # Check if result contains intent-related keywords
        intent_keywords = self.intent_patterns.get(intent, [])
        intent_score = 0.0
        
        for keyword in intent_keywords:
            if keyword in title_lower:
                if re.search(rf'\b{re.escape(keyword)}\b', title_lower):
                    intent_score += 1.0
                else:
                    intent_score += 0.5
        
        # Normalize by number of intent keywords
        if intent_keywords:
            intent_score = min(1.0, intent_score / len(intent_keywords))
        
        # Boost score if result concepts match intent expectation
        concept_boost = 0.0
        if intent == 'licensing' and any('services_licensing' in concept for concept in result_concepts):
            concept_boost = 0.3
        elif intent == 'medical_treatment' and any('services_veterinary' in concept for concept in result_concepts):
            concept_boost = 0.3
        elif intent == 'commercial_trade' and any('services_trade' in concept for concept in result_concepts):
            concept_boost = 0.3
        
        return min(1.0, intent_score + concept_boost)
    
    def _calculate_contextual_relevance(self, query: str, result_title: str,
                                      query_concepts: Dict[str, float],
                                      result_concepts: Dict[str, float]) -> float:
        """Calculate contextual relevance considering broader context"""
        
        # Length-based relevance (longer, more specific titles may be more relevant)
        length_factor = min(1.0, len(result_title.split()) / 5.0)
        
        # Specificity factor (more specific concepts get higher relevance)
        specificity = 0.0
        if query_concepts and result_concepts:
            # Average concept scores indicate specificity
            query_specificity = sum(query_concepts.values()) / len(query_concepts)
            result_specificity = sum(result_concepts.values()) / len(result_concepts)
            specificity = (query_specificity + result_specificity) / 2.0
        
        # Semantic coherence (do all parts of the result relate to query)
        coherence = self._calculate_semantic_coherence(query, result_title)
        
        return (length_factor * 0.2 + specificity * 0.4 + coherence * 0.4)
    
    def _classify_result_intent(self, result_title: str) -> str:
        """SOLUTION 5: Classify result intent to enable compatibility checking"""
        title_lower = result_title.lower()
        
        # Check for licensing keywords
        if any(keyword in title_lower for keyword in ['ترخيص', 'رخصة', 'تراخيص', 'اذن', 'إذن', 'تصريح']):
            return 'licensing'
        
        # Check for service keywords  
        if any(keyword in title_lower for keyword in ['خدمة', 'خدمات', 'مزاولة', 'تشغيل']):
            return 'services'
            
        # Check for trade/commercial keywords
        if any(keyword in title_lower for keyword in ['بيع', 'تجارة', 'تسويق', 'استيراد', 'تصدير']):
            return 'commercial_trade'
            
        return 'general'
    
    def _calculate_semantic_coherence(self, query: str, result_title: str) -> float:
        """Calculate semantic coherence between query and result"""
        query_words = set(query.lower().split())
        result_words = set(result_title.lower().split())
        
        # Remove common stop words
        stop_words = {'في', 'من', 'إلى', 'على', 'عن', 'مع', 'أو', 'و', 'ال', 'ان', 'ما'}
        query_words -= stop_words
        result_words -= stop_words
        
        if not query_words or not result_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(query_words & result_words)
        union = len(query_words | result_words)
        
        return overlap / union if union > 0 else 0.0
    
    def _generate_explanation(self, conceptual_sim: float, hierarchical_sim: float,
                            intent_alignment: float, contextual_relevance: float,
                            query_concepts: Dict[str, float], result_concepts: Dict[str, float],
                            intent: str) -> str:
        """Generate human-readable explanation of semantic scoring"""
        
        explanations = []
        
        if conceptual_sim > 0.8:
            explanations.append(f"Strong conceptual match ({conceptual_sim:.2f})")
        elif conceptual_sim > 0.5:
            explanations.append(f"Moderate conceptual similarity ({conceptual_sim:.2f})")
        else:
            explanations.append(f"Low conceptual similarity ({conceptual_sim:.2f})")
        
        if hierarchical_sim > 0.7:
            explanations.append("Same domain hierarchy")
        elif hierarchical_sim > 0.4:
            explanations.append("Related domains")
        else:
            explanations.append("Different domain hierarchies")
        
        if intent_alignment > 0.7:
            explanations.append(f"Intent well-aligned ({intent})")
        elif intent_alignment > 0.4:
            explanations.append(f"Partial intent match ({intent})")
        else:
            explanations.append(f"Intent mismatch ({intent})")
        
        top_query_concepts = sorted(query_concepts.items(), key=lambda x: x[1], reverse=True)[:2]
        top_result_concepts = sorted(result_concepts.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if top_query_concepts:
            q_concepts = [c[0].replace('_', ' ') for c in top_query_concepts]
            explanations.append(f"Query concepts: {', '.join(q_concepts)}")
        
        if top_result_concepts:
            r_concepts = [c[0].replace('_', ' ') for c in top_result_concepts]
            explanations.append(f"Result concepts: {', '.join(r_concepts)}")
        
        return " | ".join(explanations)