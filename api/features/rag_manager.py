"""
PathRAG Manager - Centralized management of PathRAG instances
"""

import os
import logging
from PathRAG import PathRAG
from PathRAG.llm import litellm_complete

logger = logging.getLogger("PathRAG")

# Setup a working directory for PathRAG.
WORKING_DIR = os.path.join(os.getcwd(), 'data')
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# LLM / Embedding model names from environment (defaults to OpenAI models).
# Any LiteLLM-supported provider can be used by setting the appropriate model name.
#
# Embedding model examples (set EMBEDDING_MODEL_NAME and EMBEDDING_DIM together):
#   OpenAI  : EMBEDDING_MODEL_NAME="text-embedding-3-small"               EMBEDDING_DIM=1536
#   OpenAI  : EMBEDDING_MODEL_NAME="text-embedding-3-large"               EMBEDDING_DIM=3072
#   Gemini  : EMBEDDING_MODEL_NAME="gemini/gemini-embedding-001"           EMBEDDING_DIM=3072
#   Bedrock : EMBEDDING_MODEL_NAME="bedrock/amazon.titan-embed-text-v2:0" EMBEDDING_DIM=1024
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

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
            embedding_model_name=EMBEDDING_MODEL_NAME,
            embedding_dim=EMBEDDING_DIM,
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
        embedding_model_name=EMBEDDING_MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
    )

    logger.info("PathRAG instance reloaded successfully")
    return _rag_instance

# Initialize the instance on module import
rag = get_rag_instance()
