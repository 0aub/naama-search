import numpy as np
import time
from typing import List
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

log = logging.getLogger(__name__)

try:
    import hnswlib
except ImportError:
    hnswlib = None

try:
    import scann
except ImportError:
    scann = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer, CrossEncoder
    sklearn_available = True
except ImportError:
    sklearn_available = False

try:
    from langchain_core.embeddings import Embeddings
    langchain_core_available = True
except ImportError:
    try:
        from langchain.embeddings.base import Embeddings
        langchain_core_available = True
    except ImportError:
        # Define a minimal interface for compatibility
        class Embeddings:
            def embed_documents(self, texts): raise NotImplementedError
            def embed_query(self, text): raise NotImplementedError
        langchain_core_available = False


class JinaTaskWrapper(Embeddings):
    """
    Memory-efficient wrapper for HuggingFaceEmbeddings that adds task parameter support for Jina v3/v4 models.
    REUSES the underlying SentenceTransformer instead of loading it twice.
    """
    
    def __init__(self, embedder: HuggingFaceEmbeddings):
        self.embedder = embedder
        self.model_name = embedder.model_name
        
        # Extract the already-loaded SentenceTransformer from HuggingFaceEmbeddings
        # instead of loading it again (MEMORY OPTIMIZATION)
        self.sentence_transformer = self._extract_existing_model()
        
        # Set appropriate task based on model version
        if "jina-embeddings-v3" in self.model_name.lower():
            self.task = "retrieval.query"
            self.doc_task = "retrieval.passage"
        elif "jina-embeddings-v4" in self.model_name.lower():
            self.task = "retrieval"
            self.doc_task = "retrieval"
        else:
            self.task = "retrieval"
            self.doc_task = "retrieval"
        
        if self.sentence_transformer:
            log.debug(f"✅ JinaTaskWrapper: Reusing loaded {self.model_name} (memory efficient)")
            log.debug(f"   🎯 Query task: {self.task}, Doc task: {self.doc_task}")
        else:
            log.info(f"🔄 JinaTaskWrapper: Using fallback embedder for {self.model_name}")
    
    def _extract_existing_model(self):
        """Extract the already-loaded SentenceTransformer from HuggingFaceEmbeddings."""
        try:
            # HuggingFaceEmbeddings stores the SentenceTransformer in client or _client
            if hasattr(self.embedder, 'client'):
                return self.embedder.client
            elif hasattr(self.embedder, '_client'):
                return self.embedder._client
            else:
                log.warning(f"⚠️ Could not extract SentenceTransformer from {self.model_name}")
                return None
        except Exception as e:
            log.warning(f"⚠️ Failed to extract existing model: {e}")
            return None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with task parameter for Jina models."""
        if self.sentence_transformer is not None:
            try:
                # Use document-specific task for better embeddings
                embeddings = self.sentence_transformer.encode(texts, task=self.doc_task, convert_to_tensor=False)
                return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
            except Exception as e:
                log.warning(f"⚠️ JinaTaskWrapper: Task embedding failed, falling back: {e}")
                return self.embedder.embed_documents(texts)
        else:
            # Use original embedder
            return self.embedder.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with task parameter for Jina models."""
        if self.sentence_transformer is not None:
            try:
                # Use SentenceTransformer directly with task parameter
                embedding = self.sentence_transformer.encode([text], task=self.task, convert_to_tensor=False)[0]
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            except Exception as e:
                log.warning(f"⚠️ JinaTaskWrapper: Task query embedding failed, falling back: {e}")
                return self.embedder.embed_query(text)
        else:
            # Use original embedder
            return self.embedder.embed_query(text)
    
    # Add compatibility methods that langchain might expect
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Make the wrapper callable for langchain compatibility."""
        return self.embed_documents(texts)
    
    @property
    def client(self):
        """Expose client for compatibility."""
        return self.sentence_transformer or self.embedder.client


class _BaseVector:
    def similarity_search_with_score(self, query: str, k: int):
        raise NotImplementedError


class FaissSimilarity(_BaseVector):
    def __init__(self, vs: FAISS, embedder: HuggingFaceEmbeddings):
        self._vs = vs
        self._embed = embedder
        # Check if this is a Jina model that needs task parameter
        self._is_jina = hasattr(embedder, 'model_name') and ("jina-embeddings-v3" in embedder.model_name.lower() or "jina-embeddings-v4" in embedder.model_name.lower())

    @classmethod
    def build(cls, embedder: HuggingFaceEmbeddings, documents: List[Document]):
        from langchain_community.vectorstores import FAISS
        import time
        import torch
        
        start_time = time.time()
        doc_count = len(documents)
        
        # CPU optimization: Warn about large datasets
        if doc_count > 200 and not torch.cuda.is_available():
            log.warning(f"🚨 CPU Performance Warning: Building FAISS index for {doc_count} documents on CPU")
            log.warning(f"💡 This may take 10-30 minutes. Consider:")
            log.warning(f"   • Enabling GPU support for 10-100x speedup")
            log.warning(f"   • Reducing dataset size for testing")
            log.warning(f"   • Using faster similarity methods like 'minilm-reranker'")
        
        # GPU acceleration setup
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            torch.cuda.empty_cache()  # Clear GPU memory
            log.info(f"🚀 GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB")
        
        # Universal model handling - works for ALL types: BERT, Jina, custom, future models
        if hasattr(embedder, 'model_name') and ("jina-embeddings-v3" in embedder.model_name.lower() or "jina-embeddings-v4" in embedder.model_name.lower()):
            log.debug(f"🎯 FAISS: Detected {embedder.model_name} - using memory-efficient wrapper")
            embedder_wrapper = JinaTaskWrapper(embedder)
            log.debug(f"⚡ FAISS: Building index for {doc_count} documents...")
            if gpu_available:
                log.info(f"🚀 Using GPU acceleration for FAISS index building...")
            vs = FAISS.from_documents(documents, embedder_wrapper)
        else:
            model_name = getattr(embedder, 'model_name', 'embedder')
            log.debug(f"⚡ FAISS: Building index for {doc_count} documents with {model_name}")
            log.info(f"💾 Memory efficient: Standard embedder (BERT/custom models work directly)")
            # BERT, sentence-transformers, Qwen, and other standard models work efficiently
            if gpu_available:
                log.info(f"🚀 Using GPU acceleration for FAISS index building...")
            elif doc_count > 100:
                log.info(f"📊 Progress will be shown in batches (CPU processing)...")
            vs = FAISS.from_documents(documents, embedder)
        
        # GPU acceleration for FAISS index
        if gpu_available:
            try:
                import faiss
                # Check if we can move to GPU (requires faiss-gpu)
                if hasattr(faiss, 'StandardGpuResources'):
                    log.info(f"🚀 Moving FAISS index to GPU...")
                    
                    # Create GPU resources
                    gpu_res = faiss.StandardGpuResources()
                    
                    # Get the underlying FAISS index
                    cpu_index = vs.index
                    
                    # Move to GPU
                    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
                    
                    # Replace the index
                    vs.index = gpu_index
                    
                    # Check GPU memory usage after moving index
                    gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                    log.info(f"✅ FAISS index on GPU! Memory used: {gpu_memory_used:.2f}GB")
                else:
                    log.warning(f"⚠️ FAISS GPU not available (faiss-gpu not properly installed)")
                    
            except Exception as e:
                log.warning(f"⚠️ GPU acceleration failed, using CPU: {e}")
                log.info(f"💡 FAISS will run on CPU (still works, just slower)")
        
        elapsed = time.time() - start_time
        docs_per_sec = doc_count / elapsed if elapsed > 0 else 0
        device_info = "GPU" if gpu_available else "CPU"
        log.info(f"✅ FAISS: Index built in {elapsed:.1f}s on {device_info} ({docs_per_sec:.1f} docs/sec)")
        
        return cls(vs, embedder)

    def similarity_search_with_score(self, query: str, k: int):
        return self._vs.similarity_search_with_score(query, k=k)


class HNSWSimilarity(_BaseVector):
    def __init__(self, documents: List[Document], embedder: HuggingFaceEmbeddings,
                 space: str = "cosine", ef_construction: int = 100, M: int = 16) -> None:

        if hnswlib is None:
            raise ModuleNotFoundError("hnswlib is not installed. Use 'pip install hnswlib'.")

        self._documents = documents
        
        # Check if this is a Jina model that needs task parameter
        if hasattr(embedder, 'model_name') and ("jina-embeddings-v3" in embedder.model_name.lower() or "jina-embeddings-v4" in embedder.model_name.lower()):
            log.debug("🎯 HNSW: Detected Jina v3/v4 model - using JinaTaskWrapper")
            self._embed = JinaTaskWrapper(embedder)
        else:
            self._embed = embedder
        
        # Performance optimization: Process documents in batches for better memory usage
        doc_texts = [d.page_content for d in documents]
        log.debug(f"⚡ HNSW: Embedding {len(doc_texts)} documents with batch processing")
        vectors = self._embed.embed_documents(doc_texts)
        self._vecs = np.asarray(vectors, dtype=np.float32)
        dim = self._vecs.shape[1]
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(max_elements=len(documents), ef_construction=ef_construction, M=M)
        self._index.add_items(self._vecs)
        self._index.set_ef(50)

    @classmethod
    def build(cls, embedder: HuggingFaceEmbeddings, documents: List[Document]):
        """Factory so ServiceSearch can create the index uniformly."""
        return cls(documents, embedder)      # order matches __init__

    def similarity_search_with_score(self, query: str, k: int):
        q_vec = np.asarray(self._embed.embed_query(query), dtype=np.float32)
        labels, distances = self._index.knn_query(q_vec, k=k)
        labels = labels[0]
        distances = distances[0]
        # HNSW returns cosine *distance* (0 = identical). Convert to similarity.
        scores = 1.0 - distances
        return [(self._documents[idx], float(score)) for idx, score in zip(labels, scores)]


class ScannSimilarity(_BaseVector):
    def __init__(self, documents: List[Document], embedder: HuggingFaceEmbeddings, k: int = 10) -> None:

        if scann is None:
            raise ModuleNotFoundError("scann is not installed. Use 'pip install scann'.")

        self._documents = documents
        self._k_build = k

        # Check if this is a Jina model that needs task parameter
        if hasattr(embedder, 'model_name') and ("jina-embeddings-v3" in embedder.model_name.lower() or "jina-embeddings-v4" in embedder.model_name.lower()):
            log.debug("🎯 SCANN: Detected Jina v3/v4 model - using JinaTaskWrapper")
            self._embed = JinaTaskWrapper(embedder)
        else:
            self._embed = embedder

        # CRITICAL FIX: Store the exact texts that were embedded for index building
        # This ensures consistency between index building and query processing
        self._indexed_texts = [d.page_content for d in documents]

        try:
            log.debug(f"⚡ SCANN: Embedding {len(self._indexed_texts)} documents with optimized batch processing")
            vectors = self._embed.embed_documents(self._indexed_texts)
            self._vecs = np.asarray(vectors, dtype=np.float32)

            # Ensure we have valid vectors
            if len(self._vecs) == 0:
                raise ValueError("No vectors generated for SCANN index")

            # Adjust SCANN parameters based on actual data size
            num_docs = len(self._vecs)
            num_leaves = min(200, max(10, num_docs // 5))
            leaves_to_search = min(20, max(5, num_leaves // 10))
            training_sample_size = min(2500, num_docs)
            reorder_count = min(100, num_docs)

            self._searcher = (
                scann.scann_ops_pybind.builder(self._vecs, k, "dot_product")
                .tree(num_leaves=num_leaves,
                      num_leaves_to_search=leaves_to_search,
                      training_sample_size=training_sample_size)
                .score_ah(2, anisotropic_quantization_threshold=0.2)
                .reorder(reorder_count)
                .build()
            )
            log.debug(f"SCANN index built successfully with {len(self._vecs)} vectors")

        except Exception as e:
            log.error(f"SCANN index building failed: {e}")
            raise RuntimeError(f"Failed to build SCANN index: {e}") from e

    @classmethod
    def build(cls, embedder: HuggingFaceEmbeddings, documents: List[Document]):
        return cls(documents, embedder)

    def similarity_search_with_score(self, query: str, k: int):
        try:
            # CRITICAL: Use the same embedding process as was used for indexing
            q_vec = np.asarray(self._embed.embed_query(query), dtype=np.float32)

            # Ensure query vector has the right dimensionality
            if len(q_vec) != self._vecs.shape[1]:
                raise ValueError(f"Query vector dimension {len(q_vec)} != index dimension {self._vecs.shape[1]}")

            # Search with proper k bounds
            actual_k = min(k, len(self._documents))
            labels, distances = self._searcher.search_batched(
                q_vec[None, :],
                final_num_neighbors=actual_k
            )

            labels = labels[0]
            distances = distances[0]

            # Filter out invalid indices
            valid_results = []
            for idx, dist in zip(labels, distances):
                if 0 <= idx < len(self._documents):
                    # For dot_product search, ScaNN returns a "distance" ~ negative dot-product.
                    # Convert back to dot (cosine for unit-normalized vectors).
                    dot = -float(dist)
                    valid_results.append((self._documents[idx], dot))

            if not valid_results:
                log.warning(f"SCANN search returned no valid results for query: {query[:50]}")

            return valid_results

        except Exception as e:
            log.error(f"SCANN search failed: {e}")
            # Fallback: return empty results rather than crash
            return []


class PureCosineVectorSimilarity(_BaseVector):
    """
    Pure cosine similarity implementation using SentenceTransformer directly.
    
    Based on the provided code example - uses direct SentenceTransformer
    embedding with cosine similarity computation for all documents.
    No vector indexing - pure mathematical cosine similarity.
    """
    
    def __init__(self, documents: List[Document], embedder: HuggingFaceEmbeddings):
        self._documents = documents
        self._original_embedder = embedder
        
        # Extract model name for SentenceTransformer loading
        if hasattr(embedder, 'model_name'):
            model_name = embedder.model_name
        else:
            raise ValueError("Embedder must have model_name attribute")
        
        # Load SentenceTransformer directly for pure cosine similarity
        try:
            log.debug(f"🎯 PureCosine: Loading {model_name} with SentenceTransformer")
            
            # Determine device
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Handle Jina models - PROPER FIX for wrap_triton issues
            if "jina-embeddings-v3" in model_name.lower() or "jina-embeddings-v4" in model_name.lower():
                # Apply definitive fix for Jina models
                import torch.library
                if not hasattr(torch.library, 'wrap_triton'):
                    log.info("🔧 Applying wrap_triton fix for Jina models...")
                    torch.library.wrap_triton = lambda fn: fn
                    log.info("✅ wrap_triton fix applied")

                # Force GPU for Jina models (required for Flash Attention/Triton)
                if device == 'cpu' or not torch.cuda.is_available():
                    log.warning("⚠️ Jina models require GPU for Flash Attention - forcing CUDA")
                    device = 'cuda'

                log.info(f"🔧 Loading Jina model on {device} with wrap_triton fix...")
                self.sentence_transformer = SentenceTransformer(model_name, trust_remote_code=True, device=device)
                self.use_task_param = True  # Enable task parameters with fix
                log.info(f"✅ PureCosine: Loaded Jina model on {device} with proper fix")
            else:
                self.sentence_transformer = SentenceTransformer(model_name, device=device)
                self.use_task_param = False
                log.info(f"✅ PureCosine: Loaded {model_name} on {device}")
                
        except Exception as e:
            log.error(f"❌ PureCosine: Failed to load {model_name}: {e}")
            raise RuntimeError(f"Cannot load SentenceTransformer: {e}")
        
        # Pre-compute document embeddings
        log.info(f"🔍 PureCosine: Computing embeddings for {len(documents)} documents...")
        doc_texts = [d.page_content for d in documents]
        
        # Encode with task parameter if needed - with Triton error handling
        if self.use_task_param:
            # Use correct task for Jina v3/v4 models - documents need retrieval.passage
            task = 'retrieval.passage' if "jina-embeddings-v3" in model_name.lower() else 'retrieval'
            try:
                # Try with task parameter first
                self.doc_embeddings = self.sentence_transformer.encode(
                    doc_texts, convert_to_tensor=False, task=task
                )
                log.info(f"✅ Document embeddings computed with task='{task}'")
            except Exception as task_error:
                if "wrap_triton" in str(task_error):
                    # Triton error during encoding - fall back to no task parameter
                    log.warning(f"⚠️ Task encoding failed with Triton error, using no task parameter: {task_error}")
                    self.doc_embeddings = self.sentence_transformer.encode(
                        doc_texts, convert_to_tensor=False
                    )
                    self.use_task_param = False  # Disable task params for future calls
                    log.info("✅ Document embeddings computed without task parameter")
                else:
                    raise task_error
        else:
            self.doc_embeddings = self.sentence_transformer.encode(
                doc_texts, convert_to_tensor=False
            )
        
        self.doc_embeddings = np.array(self.doc_embeddings, dtype=np.float32)
        log.info(f"✅ PureCosine: Document embeddings computed: {self.doc_embeddings.shape}")
    
    @classmethod
    def build(cls, embedder: HuggingFaceEmbeddings, documents: List[Document]):
        """Factory method for consistent interface"""
        return cls(documents, embedder)
    
    def similarity_search_with_score(self, query: str, k: int):
        """
        Pure cosine similarity search using SentenceTransformer directly
        
        This follows the exact pattern from the provided code example:
        1. Encode query with SentenceTransformer
        2. Compute cosine similarity with all document embeddings
        3. Return top-k results sorted by similarity
        """
        # 1. Encode query
        if self.use_task_param:
            # Use correct task for Jina v3/v4 models - queries need retrieval.query for v3
            # Get model name from SentenceTransformer (uses model_name_or_path)
            model_name = getattr(self.sentence_transformer, 'model_name_or_path', getattr(self.sentence_transformer, '_model_name', ''))
            # Jina v3 needs specific task parameters
            if "jina-embeddings-v3" in model_name.lower():
                task = 'retrieval.query'
            elif "jina-embeddings-v4" in model_name.lower():
                task = 'retrieval'  # v4 still uses generic 'retrieval'
            else:
                task = None  # No task for non-Jina models
            try:
                if task:
                    query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=False, task=task)[0]
                else:
                    query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=False)[0]
            except Exception as task_error:
                if "wrap_triton" in str(task_error):
                    # Triton error during query encoding - fall back to no task parameter
                    log.warning(f"⚠️ Query task encoding failed with Triton error, using no task parameter")
                    query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=False)[0]
                    self.use_task_param = False  # Disable task params for future calls
                else:
                    raise task_error
        else:
            query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=False)[0]
        
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # 2. Compute cosine similarity with all documents
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # 3. Get top-k results
        k = min(k, len(self._documents))
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # 4. Return results with similarity scores
        results = []
        for idx in top_indices:
            doc = self._documents[idx]
            score = float(similarities[idx])
            results.append((doc, score))
        
        return results


class PureCosineWithRerankerSimilarity(_BaseVector):
    """
    Pure cosine similarity with optional reranker combination.
    
    This provides the best of both worlds:
    1. Pure SentenceTransformer cosine similarity (like the provided code example)
    2. Optional Arabic reranker for refinement
    3. Configurable to use pure cosine OR cosine+reranker
    """
    
    def __init__(self, documents: List[Document], embedder: HuggingFaceEmbeddings,
                 reranker_model: str = "oddadmix/arabic-reranker", 
                 use_reranker: bool = True, 
                 candidates_k: int = 100):
        """
        Args:
            documents: List of documents to search
            embedder: HuggingFace embedder (model name will be extracted)
            reranker_model: Cross-encoder model for reranking  
            use_reranker: Whether to apply reranking (False = pure cosine only)
            candidates_k: Number of candidates to pass to reranker
        """
        self._documents = documents
        self._original_embedder = embedder
        self.use_reranker = use_reranker
        self.candidates_k = candidates_k
        
        # Extract model name for SentenceTransformer loading
        if hasattr(embedder, 'model_name'):
            model_name = embedder.model_name
        else:
            raise ValueError("Embedder must have model_name attribute")
        
        # Load SentenceTransformer directly (like pure cosine approach)
        try:
            log.debug(f"🎯 PureCosine+Reranker: Loading {model_name} with SentenceTransformer")
            
            # Determine device
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Handle Jina models - PROPER FIX for wrap_triton issues
            if "jina-embeddings-v3" in model_name.lower() or "jina-embeddings-v4" in model_name.lower():
                # Apply definitive fix for Jina models
                import torch.library
                if not hasattr(torch.library, 'wrap_triton'):
                    log.info("🔧 Applying wrap_triton fix for Jina models...")
                    torch.library.wrap_triton = lambda fn: fn
                    log.info("✅ wrap_triton fix applied")

                # Force GPU for Jina models (required for Flash Attention/Triton)
                if device == 'cpu' or not torch.cuda.is_available():
                    log.warning("⚠️ Jina models require GPU for Flash Attention - forcing CUDA")
                    device = 'cuda'

                log.info(f"🔧 Loading Jina model on {device} with wrap_triton fix...")
                self.sentence_transformer = SentenceTransformer(model_name, trust_remote_code=True, device=device)
                self.use_task_param = True  # Enable task parameters with fix
                log.info(f"✅ PureCosine+Reranker: Loaded Jina model on {device} with proper fix")
            else:
                self.sentence_transformer = SentenceTransformer(model_name, device=device)
                self.use_task_param = False
                log.info(f"✅ PureCosine+Reranker: Loaded {model_name} on {device}")
                
        except Exception as e:
            log.error(f"❌ PureCosine+Reranker: Failed to load {model_name}: {e}")
            raise RuntimeError(f"Cannot load SentenceTransformer: {e}")
        
        # Pre-compute document embeddings (same as pure cosine)
        log.info(f"🔍 PureCosine+Reranker: Computing embeddings for {len(documents)} documents...")
        doc_texts = [d.page_content for d in documents]
        
        # Encode with task parameter if needed - with Triton error handling
        if self.use_task_param:
            # Use correct task for Jina v3/v4 models - documents need retrieval.passage
            task = 'retrieval.passage' if "jina-embeddings-v3" in model_name.lower() else 'retrieval'
            try:
                # Try with task parameter first
                self.doc_embeddings = self.sentence_transformer.encode(
                    doc_texts, convert_to_tensor=False, task=task
                )
                log.info(f"✅ Document embeddings computed with task='{task}'")
            except Exception as task_error:
                if "wrap_triton" in str(task_error):
                    # Triton error during encoding - fall back to no task parameter
                    log.warning(f"⚠️ Task encoding failed with Triton error, using no task parameter: {task_error}")
                    self.doc_embeddings = self.sentence_transformer.encode(
                        doc_texts, convert_to_tensor=False
                    )
                    self.use_task_param = False  # Disable task params for future calls
                    log.info("✅ Document embeddings computed without task parameter")
                else:
                    raise task_error
        else:
            self.doc_embeddings = self.sentence_transformer.encode(
                doc_texts, convert_to_tensor=False
            )
        
        self.doc_embeddings = np.array(self.doc_embeddings, dtype=np.float32)
        log.info(f"✅ PureCosine+Reranker: Document embeddings computed: {self.doc_embeddings.shape}")
        
        # Load reranker if requested
        if self.use_reranker:
            try:
                log.debug(f"🎯 PureCosine+Reranker: Loading reranker model: {reranker_model}")
                self.reranker = CrossEncoder(reranker_model)
                log.debug("✅ PureCosine+Reranker: Reranker model loaded successfully")
            except Exception as e:
                log.warning(f"⚠️ PureCosine+Reranker: Failed to load reranker {reranker_model}: {e}")
                log.info("🔄 Falling back to pure cosine similarity only")
                self.reranker = None
                self.use_reranker = False
        else:
            log.info("🔄 PureCosine+Reranker: Reranker disabled - using pure cosine similarity only")
            self.reranker = None
    
    @classmethod
    def build(cls, embedder: HuggingFaceEmbeddings, documents: List[Document], 
             reranker_model: str = "oddadmix/arabic-reranker", 
             use_reranker: bool = True, candidates_k: int = 100):
        """Factory method for consistent interface"""
        return cls(documents, embedder, reranker_model, use_reranker, candidates_k)
    
    def similarity_search_with_score(self, query: str, k: int):
        """
        Pure cosine similarity search with optional reranking
        
        Combines the provided code example approach with optional reranking:
        1. Use pure SentenceTransformer cosine similarity (exactly like provided example)
        2. Optionally apply reranker to top candidates for refinement
        3. Return final ranked results
        """
        # 1. Encode query (same as pure cosine approach)
        if self.use_task_param:
            # Use correct task for Jina v3/v4 models - queries need retrieval.query for v3
            # Get model name from SentenceTransformer (uses model_name_or_path)
            model_name = getattr(self.sentence_transformer, 'model_name_or_path', getattr(self.sentence_transformer, '_model_name', ''))
            # Jina v3 needs specific task parameters
            if "jina-embeddings-v3" in model_name.lower():
                task = 'retrieval.query'
            elif "jina-embeddings-v4" in model_name.lower():
                task = 'retrieval'  # v4 still uses generic 'retrieval'
            else:
                task = None  # No task for non-Jina models
            try:
                if task:
                    query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=False, task=task)[0]
                else:
                    query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=False)[0]
            except Exception as task_error:
                if "wrap_triton" in str(task_error):
                    # Triton error during query encoding - fall back to no task parameter
                    log.warning(f"⚠️ Query task encoding failed with Triton error, using no task parameter")
                    query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=False)[0]
                    self.use_task_param = False  # Disable task params for future calls
                else:
                    raise task_error
        else:
            query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=False)[0]
        
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # 2. Compute cosine similarity with all documents (pure approach)
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # 3. Get candidates for potential reranking
        if self.use_reranker and self.reranker is not None:
            # Get more candidates for reranking
            candidates_k = min(self.candidates_k, len(self._documents))
            top_indices = np.argsort(similarities)[::-1][:candidates_k]
            
            candidate_docs = [self._documents[idx] for idx in top_indices]
            candidate_scores = similarities[top_indices]
            
            log.debug(f"PureCosine+Reranker: Selected {len(candidate_docs)} candidates for reranking")
            
            # 4. Apply reranker
            try:
                # Prepare query-document pairs for reranker
                pairs = [[query, doc.page_content] for doc in candidate_docs]
                
                # Get reranker scores
                reranker_scores = self.reranker.predict(pairs)
                
                # Combine cosine similarity with reranker scores
                alpha = 0.7  # Weight for reranker vs cosine scores
                combined_scores = (alpha * reranker_scores + 
                                 (1 - alpha) * candidate_scores)
                
                # Re-sort by combined scores
                reranked_indices = np.argsort(combined_scores)[::-1]
                
                final_docs = [candidate_docs[idx] for idx in reranked_indices]
                final_scores = combined_scores[reranked_indices]
                
                log.debug(f"PureCosine+Reranker: Reranking completed, score range: {final_scores.min():.3f} - {final_scores.max():.3f}")
                
            except Exception as e:
                log.warning(f"PureCosine+Reranker: Reranking failed: {e}, using pure cosine similarity")
                # Fall back to pure cosine
                k_actual = min(k, len(self._documents))
                top_indices = np.argsort(similarities)[::-1][:k_actual]
                final_docs = [self._documents[idx] for idx in top_indices]
                final_scores = similarities[top_indices]
        else:
            # Pure cosine similarity only (exactly like provided code example)
            k_actual = min(k, len(self._documents))
            top_indices = np.argsort(similarities)[::-1][:k_actual]
            final_docs = [self._documents[idx] for idx in top_indices]
            final_scores = similarities[top_indices]
        
        # 5. Return top-k results
        k = min(k, len(final_docs))
        results = [(final_docs[i], float(final_scores[i])) for i in range(k)]
        
        return results


# Aliases for backward compatibility
if sklearn_available:
    # Use existing PureCosineWithRerankerSimilarity as the main reranker implementation
    CosineRerankerSimilarity = PureCosineWithRerankerSimilarity
    MiniLMRerankerSimilarity = PureCosineWithRerankerSimilarity  # Same implementation, different name
    HybridMiniLMRerankerSimilarity = PureCosineWithRerankerSimilarity  # Simplified to same implementation
    log.debug("✅ Reranker similarity classes available")
else:
    log.warning("⚠️ scikit-learn or sentence-transformers not available - reranker disabled")