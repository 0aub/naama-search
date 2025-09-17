"""
Production configuration presets for Naama Search System
Clean, minimal presets for production use only.
"""

import yaml
import pathlib

def load_config_file(file_path: str) -> dict:
    """Load a YAML configuration file"""
    config_path = pathlib.Path(file_path)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

# Load configuration files
_models_config = load_config_file('config/models.yaml')
_search_config = load_config_file('config/search.yaml')

# Base data configuration shared by all presets
BASE_DATA_CONFIG = {
    "data": {
        "ar": {
            "path": "./data/NaamaServiceIn full Details.xlsx",
            "rename_map": {
                "الاسم عربي": "service",
                "التصنيف عربي": "classification",
                "القطاع عربي": "sector",
                "الوصف المختصر عربي": "description_short",
                "الوصف عربي": "description",
                "المستفيدين من الخدمة": "beneficiaries",
            },
            "combine_cols": (
                "service", "classification", "sector",
                "description_short", "description", "beneficiaries",
            ),
        }
    },
    "processing": {
        "use_morphology": True,
        "use_normalization": True,
        "use_lexical_filtering": False,
    }
}

# Production configuration presets
CONFIG_PRESETS = {
    # Original production configuration
    "original": {
        **BASE_DATA_CONFIG,
        "embedding_model": {
            "ar": "asafaya/bert-base-arabic",
            "en": "asafaya/bert-base-arabic",
            "default": "asafaya/bert-base-arabic",
        },
        "similarity": {
            "ar": "hnsw",
            "en": "hnsw",
            "default": "hnsw",
        },
        "search": {
            "top_k": 20,
            "similarity_threshold_pct": 60,
            "similarity_threshold": 0.60,
            "use_lexical_filtering": True,
        },
        "description": "Original production configuration with Arabic BERT + HNSW"
    },

    # High-quality Jina-based configuration
    "jina_reranker": {
        **BASE_DATA_CONFIG,
        "embedding_model": {
            "ar": "jinaai/jina-embeddings-v3",
            "en": "jinaai/jina-embeddings-v3",
            "default": "jinaai/jina-embeddings-v3",
        },
        "similarity": {
            "ar": "pure-cosine-reranker",
            "en": "pure-cosine-reranker",
            "default": "pure-cosine-reranker",
        },
        "search": {
            "top_k": 20,
            "similarity_threshold_pct": 50,
            "similarity_threshold": 0.50,
            "use_lexical_filtering": True,
        },
        "description": "High-quality Jina v3 embeddings with Arabic reranker"
    },

    # Fast lightweight configuration
    "lightweight_fast": {
        **BASE_DATA_CONFIG,
        "embedding_model": {
            "ar": "all-MiniLM-L6-v2",
            "en": "all-MiniLM-L6-v2",
            "default": "all-MiniLM-L6-v2",
        },
        "similarity": {
            "ar": "faiss",
            "en": "faiss",
            "default": "faiss",
        },
        "search": {
            "top_k": 20,
            "similarity_threshold_pct": 60,
            "similarity_threshold": 0.60,
            "use_lexical_filtering": True,
        },
        "description": "Fast lightweight configuration with MiniLM + FAISS"
    },

    # Arabic-optimized configuration
    "arabic_optimized": {
        **BASE_DATA_CONFIG,
        "embedding_model": {
            "ar": "asafaya/bert-base-arabic",
            "en": "jinaai/jina-embeddings-v3",
            "default": "jinaai/jina-embeddings-v3",
        },
        "similarity": {
            "ar": "hnsw",
            "en": "faiss",
            "default": "hnsw",
        },
        "search": {
            "top_k": 20,
            "similarity_threshold_pct": 60,
            "similarity_threshold": 0.60,
            "use_lexical_filtering": True,
        },
        "description": "Arabic-optimized with GATE-AraBert for Arabic, Jina v3 for English"
    }
}


def get_config(preset_name: str) -> dict:
    """Get a configuration preset by name"""
    if preset_name not in CONFIG_PRESETS:
        available = list(CONFIG_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    return CONFIG_PRESETS[preset_name].copy()


def list_presets() -> dict:
    """List all available configuration presets with descriptions"""
    return {name: config["description"] for name, config in CONFIG_PRESETS.items()}


def customize_config(preset_name: str, **overrides) -> dict:
    """
    Get a preset configuration with custom overrides

    Example:
        config = customize_config("jina_reranker",
                                 search={"top_k": 30, "similarity_threshold_pct": 70})
    """
    config = get_config(preset_name)

    # Apply overrides recursively
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    deep_update(config, overrides)
    return config