"""Paper2Code: Multi-agent system for automated code generation from research papers.

This package implements a three-stage pipeline with memory-augmented agents:
1. Planning Agent: Analyzes paper and creates implementation roadmap
2. Analyzing Agent: Performs detailed logic analysis for each component
3. Coding Agent: Generates actual code following the plan

The system includes a hierarchical memory architecture for learning from
previous papers and code implementations.
"""

__version__ = "1.0.0"
__author__ = "Paper2Code Team"

from codes.utils import (
    extract_planning,
    content_to_json,
    extract_code_from_content,
    format_json_data,
    cal_cost,
    print_response,
    print_log_cost,
)

from codes.memory import (
    MemoryEntry,
    NodeMemory,
    ShortTermMemory,
    LongTermMemory,
    MemoryManager,
)

from codes.embedding_utils import (
    EmbeddingGenerator,
    embed_paper,
    embed_code,
    compute_similarity,
)

__all__ = [
    # Utils
    "extract_planning",
    "content_to_json",
    "extract_code_from_content",
    "format_json_data",
    "cal_cost",
    "print_response",
    "print_log_cost",
    # Memory
    "MemoryEntry",
    "NodeMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryManager",
    # Embeddings
    "EmbeddingGenerator",
    "embed_paper",
    "embed_code",
    "compute_similarity",
]
