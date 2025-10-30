"""Memory system for Paper2Code multi-agent architecture.

This module implements a hierarchical memory system inspired by:
- CodeGen: Multi-turn conversation and context management
- AlphaCode: Test case and solution retrieval
- PaperCoder: Paper and code pattern matching

Components:
- NodeMemory: Per-agent memory for local context
- ShortTermMemory: Session-level memory for current paper processing
- LongTermMemory: Persistent memory across sessions for similar papers and code patterns
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np


@dataclass
class MemoryEntry:
    """A single memory entry with metadata.
    
    Attributes:
        id: Unique identifier for the entry
        content: The actual content (paper, code, analysis, etc.)
        entry_type: Type of entry (paper, code, planning, analysis, etc.)
        timestamp: When the entry was created
        metadata: Additional metadata (tags, paper_name, file_name, etc.)
        embedding: Optional vector embedding for similarity search
        access_count: Number of times this entry has been accessed
        relevance_score: Dynamic relevance score based on usage
    """
    id: str
    content: Union[str, Dict[str, Any]]
    entry_type: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    access_count: int = 0
    relevance_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy arrays."""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary, handling numpy arrays."""
        if data.get('embedding') is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)


class NodeMemory:
    """Per-agent memory for local context during processing.
    
    Each agent (planning, analyzing, coding) maintains its own node memory
    to track its current context, recent actions, and intermediate results.
    
    Inspired by CodeGen's turn-level context management.
    """
    
    def __init__(self, agent_name: str, max_size: int = 50):
        """Initialize node memory for an agent.
        
        Args:
            agent_name: Name of the agent (e.g., 'planning', 'analyzing', 'coding')
            max_size: Maximum number of recent entries to keep
        """
        self.agent_name = agent_name
        self.max_size = max_size
        self.recent_entries: deque = deque(maxlen=max_size)
        self.current_context: Dict[str, Any] = {}
        
    def add_entry(self, content: Any, entry_type: str, metadata: Optional[Dict] = None) -> None:
        """Add an entry to node memory.
        
        Args:
            content: Content to store
            entry_type: Type of content (e.g., 'input', 'output', 'decision')
            metadata: Optional metadata
        """
        entry = {
            'content': content,
            'entry_type': entry_type,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.recent_entries.append(entry)
        
    def update_context(self, key: str, value: Any) -> None:
        """Update current context state.
        
        Args:
            key: Context key
            value: Context value
        """
        self.current_context[key] = value
        
    def get_context(self, key: Optional[str] = None) -> Any:
        """Get context value(s).
        
        Args:
            key: Optional specific key to retrieve
            
        Returns:
            Context value or entire context dict
        """
        if key is None:
            return self.current_context
        return self.current_context.get(key)
    
    def get_recent_entries(self, n: Optional[int] = None, 
                          entry_type: Optional[str] = None) -> List[Dict]:
        """Get recent entries, optionally filtered by type.
        
        Args:
            n: Number of recent entries to return (None for all)
            entry_type: Optional filter by entry type
            
        Returns:
            List of recent entries
        """
        entries = list(self.recent_entries)
        
        if entry_type:
            entries = [e for e in entries if e['entry_type'] == entry_type]
            
        if n is not None:
            entries = entries[-n:]
            
        return entries
    
    def clear(self) -> None:
        """Clear node memory."""
        self.recent_entries.clear()
        self.current_context.clear()
        
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            'agent_name': self.agent_name,
            'recent_entries': list(self.recent_entries),
            'current_context': self.current_context
        }


class ShortTermMemory:
    """Session-level memory for current paper processing.
    
    Maintains context for the current paper being processed, including:
    - Original paper content
    - Planning decisions and architecture
    - Generated code and analysis
    - Intermediate artifacts
    
    Inspired by AlphaCode's per-problem context management.
    """
    
    def __init__(self, session_id: str):
        """Initialize short-term memory for a session.
        
        Args:
            session_id: Unique identifier for this session
        """
        self.session_id = session_id
        self.paper_content: Optional[Dict[str, Any]] = None
        self.planning: Dict[str, Any] = {}
        self.analysis: Dict[str, List[str]] = {}
        self.code: Dict[str, str] = {}
        self.artifacts: Dict[str, Any] = {}
        self.node_memories: Dict[str, NodeMemory] = {}
        self.session_start = datetime.now().isoformat()
        
    def set_paper(self, paper_content: Dict[str, Any]) -> None:
        """Store the paper being processed.
        
        Args:
            paper_content: Paper content in JSON or LaTeX format
        """
        self.paper_content = paper_content
        
    def add_planning(self, stage: str, content: Any) -> None:
        """Add planning stage output.
        
        Args:
            stage: Planning stage (e.g., 'overview', 'architecture', 'task_list')
            content: Planning content
        """
        self.planning[stage] = content
        
    def add_analysis(self, file_name: str, analysis: str) -> None:
        """Add analysis for a file.
        
        Args:
            file_name: Name of file being analyzed
            analysis: Analysis content
        """
        if file_name not in self.analysis:
            self.analysis[file_name] = []
        self.analysis[file_name].append(analysis)
        
    def add_code(self, file_name: str, code: str) -> None:
        """Add generated code.
        
        Args:
            file_name: Name of file
            code: Generated code content
        """
        self.code[file_name] = code
        
    def add_artifact(self, name: str, content: Any) -> None:
        """Add an artifact (config, diagram, etc.).
        
        Args:
            name: Artifact name
            content: Artifact content
        """
        self.artifacts[name] = content
        
    def get_node_memory(self, agent_name: str) -> NodeMemory:
        """Get or create node memory for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            NodeMemory instance for the agent
        """
        if agent_name not in self.node_memories:
            self.node_memories[agent_name] = NodeMemory(agent_name)
        return self.node_memories[agent_name]
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get complete session context.
        
        Returns:
            Dictionary with all session data
        """
        return {
            'session_id': self.session_id,
            'session_start': self.session_start,
            'paper_content': self.paper_content,
            'planning': self.planning,
            'analysis': self.analysis,
            'code': self.code,
            'artifacts': self.artifacts,
            'node_memories': {
                name: mem.to_dict() 
                for name, mem in self.node_memories.items()
            }
        }
    
    def save(self, output_dir: str) -> None:
        """Save short-term memory to disk.
        
        Args:
            output_dir: Directory to save memory
        """
        os.makedirs(output_dir, exist_ok=True)
        memory_path = os.path.join(output_dir, f"short_term_memory_{self.session_id}.json")
        
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_full_context(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, session_id: str, output_dir: str) -> 'ShortTermMemory':
        """Load short-term memory from disk.
        
        Args:
            session_id: Session identifier
            output_dir: Directory containing saved memory
            
        Returns:
            ShortTermMemory instance
        """
        memory = cls(session_id)
        memory_path = os.path.join(output_dir, f"short_term_memory_{session_id}.json")
        
        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                memory.paper_content = data.get('paper_content')
                memory.planning = data.get('planning', {})
                memory.analysis = data.get('analysis', {})
                memory.code = data.get('code', {})
                memory.artifacts = data.get('artifacts', {})
                
        return memory


class LongTermMemory:
    """Persistent memory across sessions for learning and retrieval.
    
    Stores and retrieves:
    - Similar papers and their implementations
    - Code patterns and solutions
    - Successful planning strategies
    - Common issues and resolutions
    
    Inspired by:
    - AlphaCode's training data clustering and retrieval
    - PaperCoder's cross-paper learning
    """
    
    def __init__(self, storage_dir: str = "./memory_store"):
        """Initialize long-term memory.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.papers: Dict[str, MemoryEntry] = {}
        self.code_patterns: Dict[str, MemoryEntry] = {}
        self.planning_strategies: Dict[str, MemoryEntry] = {}
        
        self._load_from_disk()
        
    def add_paper(self, paper_name: str, paper_content: Dict[str, Any], 
                  metadata: Optional[Dict] = None,
                  embedding: Optional[np.ndarray] = None) -> str:
        """Add a paper to long-term memory.
        
        Args:
            paper_name: Name/identifier for the paper
            paper_content: Paper content
            metadata: Optional metadata (domain, keywords, etc.)
            embedding: Optional vector embedding for similarity search
            
        Returns:
            Entry ID
        """
        entry_id = f"paper_{paper_name}_{datetime.now().timestamp()}"
        entry = MemoryEntry(
            id=entry_id,
            content=paper_content,
            entry_type='paper',
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
            embedding=embedding
        )
        self.papers[entry_id] = entry
        return entry_id
    
    def add_code_pattern(self, pattern_name: str, code: str, 
                        description: str, metadata: Optional[Dict] = None,
                        embedding: Optional[np.ndarray] = None) -> str:
        """Add a code pattern to long-term memory.
        
        Args:
            pattern_name: Name of the pattern
            code: Code implementation
            description: Description of what the pattern does
            metadata: Optional metadata (language, framework, etc.)
            embedding: Optional vector embedding
            
        Returns:
            Entry ID
        """
        entry_id = f"pattern_{pattern_name}_{datetime.now().timestamp()}"
        entry = MemoryEntry(
            id=entry_id,
            content={'code': code, 'description': description},
            entry_type='code_pattern',
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
            embedding=embedding
        )
        self.code_patterns[entry_id] = entry
        return entry_id
    
    def add_planning_strategy(self, strategy_name: str, strategy: Dict[str, Any],
                            metadata: Optional[Dict] = None) -> str:
        """Add a planning strategy to long-term memory.
        
        Args:
            strategy_name: Name of the strategy
            strategy: Strategy details
            metadata: Optional metadata
            
        Returns:
            Entry ID
        """
        entry_id = f"strategy_{strategy_name}_{datetime.now().timestamp()}"
        entry = MemoryEntry(
            id=entry_id,
            content=strategy,
            entry_type='planning_strategy',
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.planning_strategies[entry_id] = entry
        return entry_id
    
    def retrieve_similar_papers(self, query_embedding: np.ndarray, 
                               top_k: int = 5) -> List[Tuple[str, MemoryEntry, float]]:
        """Retrieve similar papers using embedding similarity.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of similar papers to retrieve
            
        Returns:
            List of (entry_id, entry, similarity_score) tuples
        """
        if not self.papers:
            return []
        
        similarities = []
        for entry_id, entry in self.papers.items():
            if entry.embedding is not None:
                # Cosine similarity
                similarity = np.dot(query_embedding, entry.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
                )
                similarities.append((entry_id, entry, float(similarity)))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def retrieve_code_patterns(self, query_embedding: Optional[np.ndarray] = None,
                              keywords: Optional[List[str]] = None,
                              top_k: int = 5) -> List[Tuple[str, MemoryEntry, float]]:
        """Retrieve relevant code patterns.
        
        Args:
            query_embedding: Optional query vector embedding
            keywords: Optional keywords for filtering
            top_k: Number of patterns to retrieve
            
        Returns:
            List of (entry_id, entry, relevance_score) tuples
        """
        if not self.code_patterns:
            return []
        
        if query_embedding is not None:
            # Use embedding similarity
            similarities = []
            for entry_id, entry in self.code_patterns.items():
                if entry.embedding is not None:
                    similarity = np.dot(query_embedding, entry.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
                    )
                    similarities.append((entry_id, entry, float(similarity)))
            similarities.sort(key=lambda x: x[2], reverse=True)
            return similarities[:top_k]
        
        elif keywords:
            # Use keyword matching
            matches = []
            for entry_id, entry in self.code_patterns.items():
                score = 0.0
                content_str = json.dumps(entry.content).lower()
                metadata_str = json.dumps(entry.metadata).lower()
                
                for keyword in keywords:
                    if keyword.lower() in content_str or keyword.lower() in metadata_str:
                        score += 1.0
                
                if score > 0:
                    matches.append((entry_id, entry, score / len(keywords)))
            
            matches.sort(key=lambda x: x[2], reverse=True)
            return matches[:top_k]
        
        else:
            # Return most accessed patterns
            patterns = [(entry_id, entry, entry.relevance_score) 
                       for entry_id, entry in self.code_patterns.items()]
            patterns.sort(key=lambda x: x[2], reverse=True)
            return patterns[:top_k]
    
    def update_access(self, entry_id: str) -> None:
        """Update access count for an entry.
        
        Args:
            entry_id: ID of the entry being accessed
        """
        # Check all memory stores
        for store in [self.papers, self.code_patterns, self.planning_strategies]:
            if entry_id in store:
                store[entry_id].access_count += 1
                # Update relevance score based on recency and frequency
                store[entry_id].relevance_score = (
                    store[entry_id].access_count * 0.7 + 
                    (1.0 / (1.0 + (datetime.now().timestamp() - 
                            datetime.fromisoformat(store[entry_id].timestamp).timestamp()) / 86400))
                )
                break
    
    def save_to_disk(self) -> None:
        """Persist long-term memory to disk."""
        # Save papers
        papers_path = os.path.join(self.storage_dir, "papers.pkl")
        with open(papers_path, 'wb') as f:
            pickle.dump(self.papers, f)
        
        # Save code patterns
        patterns_path = os.path.join(self.storage_dir, "code_patterns.pkl")
        with open(patterns_path, 'wb') as f:
            pickle.dump(self.code_patterns, f)
        
        # Save planning strategies
        strategies_path = os.path.join(self.storage_dir, "planning_strategies.pkl")
        with open(strategies_path, 'wb') as f:
            pickle.dump(self.planning_strategies, f)
    
    def _load_from_disk(self) -> None:
        """Load long-term memory from disk."""
        # Load papers
        papers_path = os.path.join(self.storage_dir, "papers.pkl")
        if os.path.exists(papers_path):
            with open(papers_path, 'rb') as f:
                self.papers = pickle.load(f)
        
        # Load code patterns
        patterns_path = os.path.join(self.storage_dir, "code_patterns.pkl")
        if os.path.exists(patterns_path):
            with open(patterns_path, 'rb') as f:
                self.code_patterns = pickle.load(f)
        
        # Load planning strategies
        strategies_path = os.path.join(self.storage_dir, "planning_strategies.pkl")
        if os.path.exists(strategies_path):
            with open(strategies_path, 'rb') as f:
                self.planning_strategies = pickle.load(f)
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics.
        
        Returns:
            Dictionary with counts of different memory types
        """
        return {
            'total_papers': len(self.papers),
            'total_code_patterns': len(self.code_patterns),
            'total_planning_strategies': len(self.planning_strategies)
        }


class MemoryManager:
    """Unified memory management for Paper2Code system.
    
    Coordinates between node memory, short-term memory, and long-term memory
    to provide contextual information during code generation.
    """
    
    def __init__(self, storage_dir: str = "./memory_store"):
        """Initialize memory manager.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        self.long_term_memory = LongTermMemory(storage_dir)
        self.short_term_memory: Optional[ShortTermMemory] = None
        
    def start_session(self, session_id: str) -> ShortTermMemory:
        """Start a new session and initialize short-term memory.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            ShortTermMemory instance for this session
        """
        self.short_term_memory = ShortTermMemory(session_id)
        return self.short_term_memory
    
    def end_session(self, output_dir: str, save_to_long_term: bool = True) -> None:
        """End current session and optionally save to long-term memory.
        
        Args:
            output_dir: Directory to save short-term memory
            save_to_long_term: Whether to save successful patterns to long-term memory
        """
        if self.short_term_memory:
            # Save short-term memory
            self.short_term_memory.save(output_dir)
            
            # Optionally extract and save patterns to long-term memory
            if save_to_long_term:
                self._extract_patterns_to_long_term()
            
            self.short_term_memory = None
    
    def _extract_patterns_to_long_term(self) -> None:
        """Extract successful patterns from current session to long-term memory."""
        if not self.short_term_memory:
            return
        
        # Extract successful code patterns
        for file_name, code in self.short_term_memory.code.items():
            if code and len(code) > 100:  # Only save substantial code
                pattern_name = f"{self.short_term_memory.session_id}_{file_name}"
                description = f"Code from {file_name} in {self.short_term_memory.session_id}"
                self.long_term_memory.add_code_pattern(
                    pattern_name=pattern_name,
                    code=code,
                    description=description,
                    metadata={
                        'file_name': file_name,
                        'session_id': self.short_term_memory.session_id
                    }
                )
        
        # Save planning strategies
        if self.short_term_memory.planning:
            strategy_name = f"strategy_{self.short_term_memory.session_id}"
            self.long_term_memory.add_planning_strategy(
                strategy_name=strategy_name,
                strategy=self.short_term_memory.planning,
                metadata={'session_id': self.short_term_memory.session_id}
            )
        
        # Persist to disk
        self.long_term_memory.save_to_disk()
    
    def retrieve_similar_context(self, query_embedding: Optional[np.ndarray] = None,
                                keywords: Optional[List[str]] = None,
                                top_k: int = 3) -> Dict[str, List[Any]]:
        """Retrieve similar papers and code patterns for context.
        
        Args:
            query_embedding: Optional query embedding
            keywords: Optional keywords for search
            top_k: Number of items to retrieve
            
        Returns:
            Dictionary with similar papers and code patterns
        """
        result = {
            'similar_papers': [],
            'code_patterns': []
        }
        
        if query_embedding is not None:
            result['similar_papers'] = self.long_term_memory.retrieve_similar_papers(
                query_embedding, top_k
            )
        
        result['code_patterns'] = self.long_term_memory.retrieve_code_patterns(
            query_embedding=query_embedding,
            keywords=keywords,
            top_k=top_k
        )
        
        return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about all memory systems.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            'long_term': self.long_term_memory.get_stats()
        }
        
        if self.short_term_memory:
            stats['short_term'] = {
                'session_id': self.short_term_memory.session_id,
                'num_planning_stages': len(self.short_term_memory.planning),
                'num_analyses': len(self.short_term_memory.analysis),
                'num_code_files': len(self.short_term_memory.code),
                'num_artifacts': len(self.short_term_memory.artifacts)
            }
        
        return stats
