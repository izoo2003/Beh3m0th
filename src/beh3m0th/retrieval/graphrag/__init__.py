"""Project 3: NetworkX knowledge graph and Leiden communities."""

from beh3m0th.retrieval.graphrag.community import detect_communities, hashing_vector
from beh3m0th.retrieval.graphrag.errors import GraphExtractionError, GraphQueryError
from beh3m0th.retrieval.graphrag.extractor import extract_prose, extract_python, merge_results
from beh3m0th.retrieval.graphrag.graph_store import GraphStoreProtocol, NetworkXGraphStore
from beh3m0th.retrieval.graphrag.llm_extract import LLMProseExtractor, TextCompleter
from beh3m0th.retrieval.graphrag.neo4j_store import Neo4jGraphStore
from beh3m0th.retrieval.graphrag.pipeline import GraphRAGPipeline
from beh3m0th.retrieval.graphrag.resolve import cosine_similarity, merge_by_cosine
from beh3m0th.retrieval.graphrag.schemas import (
    CommunitySummary,
    Entity,
    GraphExtractionResult,
    GraphTraversal,
    Relationship,
    SubgraphResult,
)

__all__ = [
    "CommunitySummary",
    "Entity",
    "GraphExtractionError",
    "GraphExtractionResult",
    "GraphQueryError",
    "GraphRAGPipeline",
    "GraphStoreProtocol",
    "GraphTraversal",
    "LLMProseExtractor",
    "Neo4jGraphStore",
    "NetworkXGraphStore",
    "Relationship",
    "SubgraphResult",
    "TextCompleter",
    "cosine_similarity",
    "detect_communities",
    "extract_prose",
    "extract_python",
    "hashing_vector",
    "merge_by_cosine",
    "merge_results",
]
