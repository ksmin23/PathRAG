"""
PathRAG Manager - Centralized management of PathRAG instances
"""

import os
import logging
from PathRAG import PathRAG
from PathRAG.llm import litellm_complete, litellm_embedding

logger = logging.getLogger("PathRAG")

# Setup a working directory for PathRAG.
WORKING_DIR = os.path.join(os.getcwd(), 'data')
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# LLM / Embedding model names from environment (defaults to OpenAI models)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# Global PathRAG instance
_rag_instance = None

def get_rag_instance():
    """
    Get the current PathRAG instance, initializing it if necessary.
    Uses LiteLLM so any provider can be configured via LLM_MODEL_NAME env var.
    """
    global _rag_instance

    if _rag_instance is None:
        logger.info(f"Initializing PathRAG instance (model={LLM_MODEL_NAME})...")
        _rag_instance = PathRAG(
            working_dir=WORKING_DIR,
            llm_model_func=litellm_complete,
            llm_model_name=LLM_MODEL_NAME,
            embedding_func=litellm_embedding,
        )
        logger.info("PathRAG instance initialized successfully")

    return _rag_instance

def reload_rag_instance():
    """
    Reload the PathRAG instance to recognize new data files.
    """
    global _rag_instance

    logger.info("Reloading PathRAG instance...")

    _rag_instance = PathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=litellm_complete,
        llm_model_name=LLM_MODEL_NAME,
        embedding_func=litellm_embedding,
    )

    logger.info("PathRAG instance reloaded successfully")
    return _rag_instance

# Initialize the instance on module import
rag = get_rag_instance()
