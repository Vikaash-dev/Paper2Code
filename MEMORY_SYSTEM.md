# Memory System Architecture

## Overview

Paper2Code now incorporates a sophisticated hierarchical memory system inspired by recent advances in code generation research. This system enables the multi-agent framework to learn from previous papers and code implementations, improving generation quality over time.

## Research Foundations

The memory system integrates insights from three key research papers:

### 1. CodeGen: An Open Large Language Model for Code
**Key Insights Integrated:**
- Multi-turn conversation management for maintaining context
- Progressive code generation through iterative refinement
- Turn-level context tracking for agent decision-making

**Implementation:**
- `NodeMemory` class maintains per-agent conversation context
- Recent entries tracking with configurable window size
- Context state management for each agent (planning, analyzing, coding)

### 2. Competition-Level Code Generation with AlphaCode
**Key Insights Integrated:**
- Clustering and retrieval of similar problems and solutions
- Test case and solution pattern matching
- Large-scale pre-training data organization

**Implementation:**
- Vector embedding-based similarity search
- Long-term memory storage of code patterns
- Relevance scoring based on access patterns and recency

### 3. PaperCoder: Automated Code Generation from Papers
**Key Insights Integrated:**
- Cross-paper learning and pattern extraction
- Methodology-aware code generation
- Semantic matching between papers

**Implementation:**
- Paper embedding generation focusing on methodology sections
- Similar paper retrieval during planning phase
- Code pattern extraction from successful implementations

## Memory Hierarchy

### 1. Node Memory (Per-Agent)
**Purpose:** Local context for individual agents during processing

**Features:**
- Recent entries tracking (default: 50 most recent)
- Current context state management
- Entry type filtering
- Agent-specific memory isolation

**Use Cases:**
- Planning agent tracks decisions and constraints
- Analyzing agent maintains file-level context
- Coding agent references previous code generations

**Example:**
```python
from memory import NodeMemory

# Create node memory for planning agent
planning_memory = NodeMemory("planning", max_size=50)

# Add entries during processing
planning_memory.add_entry(
    content="Decided to use PyTorch for implementation",
    entry_type="decision",
    metadata={"reasoning": "Paper uses PyTorch examples"}
)

# Update current context
planning_memory.update_context("framework", "pytorch")

# Retrieve recent decisions
recent_decisions = planning_memory.get_recent_entries(
    n=5,
    entry_type="decision"
)
```

### 2. Short-Term Memory (Session-Level)
**Purpose:** Maintain context for current paper being processed

**Features:**
- Paper content storage
- Planning decisions tracking
- Generated code accumulation
- Intermediate artifacts storage
- Integration with node memories

**Lifecycle:**
1. Created at session start
2. Populated during processing
3. Saved at session end
4. Optionally migrated to long-term memory

**Example:**
```python
from memory import ShortTermMemory

# Initialize for current session
stm = ShortTermMemory("transformer_paper_2025")

# Store paper being processed
stm.set_paper(paper_content)

# Track planning stages
stm.add_planning("overview", overview_plan)
stm.add_planning("architecture", architecture_design)

# Store generated code
stm.add_code("model.py", model_code)

# Save session
stm.save("./outputs/transformer")
```

### 3. Long-Term Memory (Cross-Session)
**Purpose:** Persistent learning across all papers and sessions

**Features:**
- Paper repository with vector embeddings
- Code pattern library
- Planning strategy database
- Relevance-based retrieval
- Access count tracking

**Storage:**
- Pickled Python objects for fast serialization
- Separate stores for papers, patterns, strategies
- Automatic persistence to disk

**Example:**
```python
from memory import LongTermMemory
from embedding_utils import EmbeddingGenerator, embed_paper

# Initialize long-term memory
ltm = LongTermMemory("./memory_store")

# Generate embedding for new paper
embedding_gen = EmbeddingGenerator(backend="openai")
paper_emb = embed_paper(paper_content, embedding_gen)

# Add paper to long-term memory
ltm.add_paper(
    paper_name="attention_is_all_you_need",
    paper_content=paper_content,
    metadata={
        "domain": "NLP",
        "keywords": ["transformer", "attention", "seq2seq"]
    },
    embedding=paper_emb
)

# Retrieve similar papers
similar = ltm.retrieve_similar_papers(
    query_embedding=new_paper_emb,
    top_k=5
)

# Add code pattern
ltm.add_code_pattern(
    pattern_name="transformer_block",
    code=transformer_code,
    description="Multi-head attention with feedforward network",
    metadata={"framework": "pytorch"}
)
```

## Memory Manager

The `MemoryManager` class coordinates all memory systems:

```python
from memory import MemoryManager

# Initialize memory manager
manager = MemoryManager(storage_dir="./memory_store")

# Start new session
stm = manager.start_session("new_paper_2025")

# Retrieve relevant context
context = manager.retrieve_similar_context(
    query_embedding=paper_embedding,
    keywords=["transformer", "attention"],
    top_k=3
)

# Use retrieved papers and patterns in prompts
# ... generate code ...

# End session and optionally save to long-term memory
manager.end_session(
    output_dir="./outputs/new_paper",
    save_to_long_term=True  # Extract successful patterns
)

# Get memory statistics
stats = manager.get_memory_stats()
print(f"Total papers in memory: {stats['long_term']['total_papers']}")
```

## Vector Embeddings

### Embedding Generation

The system supports multiple embedding backends:

**OpenAI (Recommended for Production):**
```python
from embedding_utils import EmbeddingGenerator

# Use OpenAI's text-embedding-3-small
embedder = EmbeddingGenerator(
    backend="openai",
    model="text-embedding-3-small",
    cache_dir="./embedding_cache"
)

embedding = embedder.embed_text("Your text here")
```

**Local Models (For Offline Use):**
```python
# Use sentence-transformers
embedder = EmbeddingGenerator(
    backend="local",
    model="all-MiniLM-L6-v2",
    cache_dir="./embedding_cache"
)
```

### Embedding Strategy

**For Papers:**
1. Extract title, abstract, and methodology sections
2. Combine into representative text
3. Generate embedding (cached for reuse)
4. Store with paper entry

**For Code:**
1. Extract docstrings and comments
2. Combine with actual code
3. Generate embedding
4. Store with code pattern entry

### Similarity Search

Cosine similarity is used for retrieval:

```python
from embedding_utils import compute_similarity, find_most_similar

# Compute similarity between two embeddings
similarity = compute_similarity(emb1, emb2)

# Find most similar from candidates
similar_indices = find_most_similar(
    query_embedding=query_emb,
    candidate_embeddings=candidate_embs,
    top_k=5
)
```

## Integration with Code Generation Pipeline

### Planning Stage
```python
from codes.memory import MemoryManager
from codes.embedding_utils import EmbeddingGenerator, embed_paper

# Initialize memory
memory_manager = MemoryManager()
stm = memory_manager.start_session(paper_name)

# Generate paper embedding
embedder = EmbeddingGenerator(backend="openai")
paper_emb = embed_paper(paper_content, embedder)

# Retrieve similar papers
retrieved = memory_manager.retrieve_similar_context(
    query_embedding=paper_emb,
    top_k=3
)

# Enhance prompts with retrieved context
enhanced_prompt = add_memory_context(
    base_prompt,
    similar_papers=retrieved['similar_papers'],
    code_patterns=retrieved['code_patterns']
)
```

### Analyzing Stage
```python
# Get analyzing agent's node memory
analyzing_agent = stm.get_node_memory("analyzing")

# Track current file being analyzed
analyzing_agent.update_context("current_file", "model.py")

# Add analysis to short-term memory
stm.add_analysis("model.py", analysis_text)

# Store in node memory
analyzing_agent.add_entry(
    content={"file": "model.py", "complexity": "high"},
    entry_type="analysis_complete"
)
```

### Coding Stage
```python
# Get coding agent's node memory
coding_agent = stm.get_node_memory("coding")

# Retrieve relevant code patterns
patterns = memory_manager.long_term_memory.retrieve_code_patterns(
    keywords=["pytorch", "transformer", "attention"],
    top_k=3
)

# Generate code with pattern context
generated_code = generate_with_patterns(patterns)

# Store in short-term memory
stm.add_code("model.py", generated_code)

# Track in node memory
coding_agent.add_entry(
    content={"file": "model.py", "lines": len(generated_code.split('\n'))},
    entry_type="code_generated"
)
```

## Usage Examples

### Basic Usage
```bash
# Run planning with memory integration
python codes/1_planning_memory.py \
    --paper_name "attention_paper" \
    --gpt_version "o3-mini" \
    --pdf_json_path "papers/attention.json" \
    --output_dir "./outputs/attention" \
    --memory_dir "./memory_store"
```

### Advanced Usage with Custom Memory

```python
from memory import MemoryManager, LongTermMemory
from embedding_utils import EmbeddingGenerator

# Initialize with custom storage
memory = MemoryManager(storage_dir="./custom_memory")

# Pre-populate with known papers
ltm = memory.long_term_memory
embedder = EmbeddingGenerator(backend="openai")

# Add reference papers
for paper_file in reference_papers:
    with open(paper_file) as f:
        content = json.load(f)
    
    emb = embed_paper(content, embedder)
    ltm.add_paper(
        paper_name=paper_file.stem,
        paper_content=content,
        embedding=emb,
        metadata={"source": "reference_collection"}
    )

# Save to disk
ltm.save_to_disk()
```

## Performance Considerations

### Embedding Cache
- Embeddings are automatically cached to avoid redundant API calls
- Cache is stored in `./embedding_cache/embeddings.pkl`
- Significant cost savings for repeated papers/code

### Memory Size
- Long-term memory grows with each processed paper
- Periodic cleanup recommended for large deployments
- Consider implementing eviction policies for very large systems

### Retrieval Speed
- In-memory similarity search is fast for < 1000 papers
- For larger scale, consider:
  - Vector databases (Pinecone, Weaviate, Milvus)
  - Approximate nearest neighbors (FAISS, Annoy)
  - Batched retrieval

## Best Practices

1. **Start Session Early:** Initialize memory at the beginning of processing
2. **Update Regularly:** Add entries to node memory throughout processing
3. **Save Frequently:** Persist short-term memory at key checkpoints
4. **Curate Long-Term Memory:** Periodically review and clean stored patterns
5. **Monitor Costs:** Track embedding API usage with cost monitoring
6. **Validate Retrievals:** Check relevance scores of retrieved items
7. **Document Metadata:** Add rich metadata for better filtering and search

## Future Enhancements

Potential improvements to the memory system:

1. **Reinforcement Learning:** Track which retrievals lead to successful code generation
2. **Active Learning:** Identify gaps in memory and suggest papers to add
3. **Multi-Modal Memory:** Store diagrams, equations, and visualizations
4. **Distributed Memory:** Scale across multiple machines
5. **Version Control:** Track evolution of code patterns over time
6. **Collaborative Memory:** Share memory across team members
7. **Personalization:** User-specific memory preferences and patterns

## Troubleshooting

### Common Issues

**Q: Embeddings are slow to generate**
A: Use caching and batch processing. Consider switching to local model for development.

**Q: Memory files are large**
A: Implement periodic cleanup. Consider compressing old entries.

**Q: Retrieval returns irrelevant results**
A: Tune similarity thresholds. Add more metadata for filtering. Improve keyword extraction.

**Q: Out of memory errors**
A: Limit long-term memory size. Implement eviction policies. Use disk-based storage for large embeddings.

## References

1. Nijkamp, E., et al. (2022). "CodeGen: An Open Large Language Model for Code"
2. Li, Y., et al. (2022). "Competition-Level Code Generation with AlphaCode"
3. The PaperCoder Team (2025). "PaperCoder: Automated Code Generation from Papers"

## Additional Resources

- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)
- [Sentence Transformers](https://www.sbert.net/)
- [Vector Search Best Practices](https://www.pinecone.io/learn/vector-search/)
