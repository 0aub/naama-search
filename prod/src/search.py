"""
Production Search Engine - Winner Configuration ONLY
multilingual_e5_pure-cosine-reranker_k100_a0.6_full_proc
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.docstore.document import Document

class ProductionSearch:
    """
    Production search engine with ONLY the winning configuration hardcoded.
    No presets, no configuration switching, no other similarity methods.
    """

    def __init__(self, documents: List[Document], normalizer, morpher, lexical_filter):
        # HARDCODED WINNER CONFIG WITH GPU OPTIMIZATION
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {device}")

        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large', trust_remote_code=True, device=device)
        self.reranker = CrossEncoder('oddadmix/arabic-reranker', device=device)

        # Processing components (full_proc)
        self.normalizer = normalizer
        self.morpher = morpher
        self.lexical_filter = lexical_filter

        # Optimized parameters for faster performance
        self.candidates_k = 100  # Restored to 100 for better recall (from test logs)
        self.alpha = 0.6
        self.top_k = 30  # Increased to 30 for better coverage
        self.threshold = 0.30  # Lowered threshold for Arabic language variants (based on test logs)

        # Pre-compute document embeddings
        self.documents = documents
        self._build_embeddings()

    def _build_embeddings(self):
        """Build embeddings for all documents"""
        doc_texts = []
        for doc in self.documents:
            # Apply winner processing (full_proc)
            text = doc.page_content
            text = self.normalizer.normalize(text)
            text = self.morpher.reduce(text)
            doc_texts.append(f"passage: {text}")

        self.doc_embeddings = self.embedding_model.encode(doc_texts)

    def search(self, query: str, candidates_k: int = None, alpha: float = None,
               top_k: int = None, threshold: float = None) -> Dict[str, Any]:
        """Search with winner configuration and optional parameter overrides"""
        # Use provided parameters or defaults
        use_candidates_k = candidates_k if candidates_k is not None else self.candidates_k
        use_alpha = alpha if alpha is not None else self.alpha
        use_top_k = top_k if top_k is not None else self.top_k
        use_threshold = threshold if threshold is not None else self.threshold


        # Apply winner processing (full_proc)
        processed_query = self.normalizer.normalize(query)
        processed_query = self.morpher.reduce(processed_query)

        # Embed query with E5 format
        query_embedding = self.embedding_model.encode([f"query: {processed_query}"])

        # Pure cosine similarity
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        # Get top candidates_k for reranker
        top_indices = np.argsort(similarities)[::-1][:use_candidates_k]
        candidates = [self.documents[i] for i in top_indices]
        candidate_scores = similarities[top_indices]

        # Rerank with Arabic reranker
        pairs = [(query, doc.page_content) for doc in candidates]
        reranker_scores = self.reranker.predict(pairs)

        # Combine scores with alpha weighting
        final_scores = []
        for i, (cos_score, rerank_score) in enumerate(zip(candidate_scores, reranker_scores)):
            final_score = (1 - use_alpha) * cos_score + use_alpha * rerank_score
            final_scores.append((final_score, candidates[i]))

        # Sort by final score and filter by threshold
        final_scores.sort(reverse=True, key=lambda x: x[0])

        results = []
        for score, doc in final_scores:
            if score >= use_threshold and len(results) < use_top_k:
                # Apply lexical filtering (full_proc)
                if self.lexical_filter.passes_filter(query, doc.page_content):
                    # Skip services with empty/null sectors
                    sector = doc.metadata.get('sector', '')
                    if (sector is None or
                        not sector or
                        str(sector).strip() == '' or
                        str(sector).lower() in ['null', 'none', 'nan']):
                        continue  # Skip if sector is empty, None, null, or nan

                    classification = doc.metadata.get('classification', '').strip() or 'غير محدد'

                    results.append({
                        'service': doc.metadata.get('service', ''),
                        'description': doc.page_content,
                        'classification': classification,
                        'sector': sector
                    })

        return {
            'hits_kept': results,
            'total_time': 0.0,  # Can add timing if needed
            'query': query
        }