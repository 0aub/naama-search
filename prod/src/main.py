"""
Production FastAPI Application - Winner Configuration ONLY
multilingual_e5_pure-cosine-reranker_k100_a0.6_full_proc
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import time

from .search import ProductionSearch
from .processing import ProductionNormalizer, ProductionMorphReducer, ProductionLexicalFilter
from .utils import ProductionDataLoader
from .logger import get_logger

# Initialize logger
logger = get_logger("naama-production")

# Initialize FastAPI app
app = FastAPI(
    title="Naama Search API - Production",
    description="Winner configuration: multilingual_e5_pure-cosine-reranker_k100_a0.6_full_proc",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SearchParameters(BaseModel):
    candidates_k: int = 100
    alpha: float = 0.6
    top_k: int = 20
    similarity_threshold: float = 0.55

class SearchRequest(BaseModel):
    query: str
    parameters: SearchParameters = None

class SearchResponse(BaseModel):
    hits_kept: List[Dict[str, Any]]
    total_time: float
    query: str
    parameters_used: SearchParameters

# Global search engine instance (initialized on startup)
search_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup"""
    global search_engine

    logger.info("🚀 Starting Naama Search Production API")
    logger.info("Configuration: multilingual_e5_pure-cosine-reranker_k100_a0.6_full_proc")

    try:
        # Initialize processing components (full_proc)
        normalizer = ProductionNormalizer()
        morpher = ProductionMorphReducer()
        lexical_filter = ProductionLexicalFilter(normalizer)

        # Load data
        data_loader = ProductionDataLoader()
        documents = data_loader.documents

        # Initialize search engine with winner config
        search_engine = ProductionSearch(documents, normalizer, morpher, lexical_filter)

        logger.info(f"✅ Search engine initialized with {len(documents)} documents")
        logger.info("🏆 Winner configuration loaded and ready!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize search engine: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Naama Search API - Production",
        "configuration": "multilingual_e5_pure-cosine-reranker_k100_a0.6_full_proc",
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "configuration": "winner"}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search endpoint with winner configuration and optional parameter overrides"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        start_time = time.time()

        # Get parameters (use provided or defaults)
        params = request.parameters or SearchParameters()

        # Search with winner configuration and optional overrides
        results = search_engine.search(
            request.query,
            candidates_k=params.candidates_k,
            alpha=params.alpha,
            top_k=params.top_k,
            threshold=params.similarity_threshold
        )

        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['parameters_used'] = params

        if request.parameters:
            logger.info(f"Developer search: '{request.query}' with custom params -> {len(results['hits_kept'])} results in {total_time:.3f}s")
        else:
            logger.info(f"Search completed: '{request.query}' -> {len(results['hits_kept'])} results in {total_time:.3f}s")

        return SearchResponse(**results)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/parameters")
async def get_default_parameters():
    """Get default search parameters"""
    return SearchParameters()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)