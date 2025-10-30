"""Unit tests for memory system in Paper2Code.

Tests the hierarchical memory system including:
- NodeMemory for per-agent context
- ShortTermMemory for session-level storage
- LongTermMemory for persistent learning
- MemoryManager for coordination
"""

import pytest
import numpy as np
import tempfile
import os
import shutil
from codes.memory import (
    MemoryEntry,
    NodeMemory,
    ShortTermMemory,
    LongTermMemory,
    MemoryManager
)


class TestMemoryEntry:
    """Tests for MemoryEntry data class."""
    
    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            id="test_001",
            content="Test content",
            entry_type="paper",
            timestamp="2025-01-01T00:00:00",
            metadata={"key": "value"}
        )
        
        assert entry.id == "test_001"
        assert entry.content == "Test content"
        assert entry.entry_type == "paper"
        assert entry.access_count == 0
        assert entry.relevance_score == 1.0
    
    def test_memory_entry_with_embedding(self):
        """Test entry with numpy embedding."""
        embedding = np.array([0.1, 0.2, 0.3])
        entry = MemoryEntry(
            id="test_002",
            content={"code": "print('hello')"},
            entry_type="code",
            timestamp="2025-01-01T00:00:00",
            embedding=embedding
        )
        
        assert np.array_equal(entry.embedding, embedding)
    
    def test_memory_entry_serialization(self):
        """Test to_dict and from_dict."""
        embedding = np.array([0.1, 0.2, 0.3])
        entry = MemoryEntry(
            id="test_003",
            content="Test",
            entry_type="test",
            timestamp="2025-01-01T00:00:00",
            embedding=embedding
        )
        
        # Convert to dict
        data = entry.to_dict()
        assert isinstance(data['embedding'], list)
        
        # Convert back
        restored = MemoryEntry.from_dict(data)
        assert restored.id == entry.id
        assert np.array_equal(restored.embedding, embedding)


class TestNodeMemory:
    """Tests for NodeMemory class."""
    
    def test_node_memory_initialization(self):
        """Test creating node memory."""
        nm = NodeMemory("planning", max_size=10)
        
        assert nm.agent_name == "planning"
        assert nm.max_size == 10
        assert len(nm.recent_entries) == 0
    
    def test_add_entry(self):
        """Test adding entries to node memory."""
        nm = NodeMemory("analyzing")
        
        nm.add_entry("Analysis 1", "analysis", {"file": "test.py"})
        nm.add_entry("Analysis 2", "analysis", {"file": "main.py"})
        
        entries = nm.get_recent_entries()
        assert len(entries) == 2
        assert entries[0]['content'] == "Analysis 1"
    
    def test_max_size_enforcement(self):
        """Test that max_size is enforced."""
        nm = NodeMemory("coding", max_size=3)
        
        for i in range(5):
            nm.add_entry(f"Entry {i}", "test")
        
        entries = nm.get_recent_entries()
        assert len(entries) == 3
        assert entries[-1]['content'] == "Entry 4"
    
    def test_context_management(self):
        """Test context update and retrieval."""
        nm = NodeMemory("planning")
        
        nm.update_context("framework", "pytorch")
        nm.update_context("model", "transformer")
        
        assert nm.get_context("framework") == "pytorch"
        assert nm.get_context("model") == "transformer"
        assert nm.get_context() == {"framework": "pytorch", "model": "transformer"}
    
    def test_filter_by_entry_type(self):
        """Test filtering entries by type."""
        nm = NodeMemory("coding")
        
        nm.add_entry("Decision 1", "decision")
        nm.add_entry("Code 1", "code")
        nm.add_entry("Decision 2", "decision")
        
        decisions = nm.get_recent_entries(entry_type="decision")
        assert len(decisions) == 2
        assert all(e['entry_type'] == "decision" for e in decisions)


class TestShortTermMemory:
    """Tests for ShortTermMemory class."""
    
    def test_short_term_memory_initialization(self):
        """Test creating short-term memory."""
        stm = ShortTermMemory("session_123")
        
        assert stm.session_id == "session_123"
        assert stm.paper_content is None
        assert len(stm.planning) == 0
    
    def test_set_paper(self):
        """Test storing paper content."""
        stm = ShortTermMemory("session_123")
        paper = {"title": "Test Paper", "abstract": "Test abstract"}
        
        stm.set_paper(paper)
        
        assert stm.paper_content == paper
    
    def test_add_planning_stages(self):
        """Test adding planning stages."""
        stm = ShortTermMemory("session_123")
        
        stm.add_planning("overview", "Overall plan")
        stm.add_planning("architecture", "System design")
        
        assert len(stm.planning) == 2
        assert stm.planning["overview"] == "Overall plan"
    
    def test_add_code(self):
        """Test adding generated code."""
        stm = ShortTermMemory("session_123")
        
        stm.add_code("model.py", "class Model: pass")
        stm.add_code("trainer.py", "def train(): pass")
        
        assert len(stm.code) == 2
        assert "class Model" in stm.code["model.py"]
    
    def test_node_memory_integration(self):
        """Test getting node memories."""
        stm = ShortTermMemory("session_123")
        
        nm1 = stm.get_node_memory("planning")
        nm2 = stm.get_node_memory("coding")
        nm3 = stm.get_node_memory("planning")  # Should return same instance
        
        assert nm1.agent_name == "planning"
        assert nm2.agent_name == "coding"
        assert nm1 is nm3
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading short-term memory."""
        stm = ShortTermMemory("session_456")
        stm.set_paper({"title": "Test"})
        stm.add_planning("overview", "Plan")
        stm.add_code("test.py", "code")
        
        # Save
        stm.save(str(tmp_path))
        
        # Load
        loaded = ShortTermMemory.load("session_456", str(tmp_path))
        
        assert loaded.session_id == "session_456"
        assert loaded.paper_content["title"] == "Test"
        assert "overview" in loaded.planning


class TestLongTermMemory:
    """Tests for LongTermMemory class."""
    
    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage = tmp_path / "ltm_test"
        storage.mkdir()
        yield str(storage)
        shutil.rmtree(storage, ignore_errors=True)
    
    def test_long_term_memory_initialization(self, temp_storage):
        """Test creating long-term memory."""
        ltm = LongTermMemory(temp_storage)
        
        assert ltm.storage_dir == temp_storage
        assert len(ltm.papers) == 0
        assert len(ltm.code_patterns) == 0
    
    def test_add_paper(self, temp_storage):
        """Test adding a paper to long-term memory."""
        ltm = LongTermMemory(temp_storage)
        
        paper_content = {"title": "Test Paper", "abstract": "Abstract"}
        embedding = np.array([0.1, 0.2, 0.3])
        
        entry_id = ltm.add_paper(
            paper_name="test_paper",
            paper_content=paper_content,
            metadata={"domain": "ML"},
            embedding=embedding
        )
        
        assert entry_id.startswith("paper_test_paper")
        assert entry_id in ltm.papers
        assert ltm.papers[entry_id].metadata["domain"] == "ML"
    
    def test_add_code_pattern(self, temp_storage):
        """Test adding a code pattern."""
        ltm = LongTermMemory(temp_storage)
        
        entry_id = ltm.add_code_pattern(
            pattern_name="attention",
            code="def attention(): pass",
            description="Attention mechanism",
            metadata={"framework": "pytorch"}
        )
        
        assert entry_id.startswith("pattern_attention")
        assert entry_id in ltm.code_patterns
    
    def test_retrieve_similar_papers(self, temp_storage):
        """Test retrieving similar papers by embedding."""
        ltm = LongTermMemory(temp_storage)
        
        # Add papers with embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        emb3 = np.array([0.9, 0.1, 0.0])  # Similar to emb1
        
        ltm.add_paper("paper1", {"title": "P1"}, embedding=emb1)
        ltm.add_paper("paper2", {"title": "P2"}, embedding=emb2)
        ltm.add_paper("paper3", {"title": "P3"}, embedding=emb3)
        
        # Query with embedding similar to emb1 and emb3
        query = np.array([0.95, 0.05, 0.0])
        similar = ltm.retrieve_similar_papers(query, top_k=2)
        
        assert len(similar) == 2
        # Most similar should be paper3 or paper1
        assert similar[0][2] > 0.8  # High similarity score
    
    def test_retrieve_code_patterns_by_keywords(self, temp_storage):
        """Test retrieving code patterns by keywords."""
        ltm = LongTermMemory(temp_storage)
        
        ltm.add_code_pattern(
            "transformer",
            "class Transformer: pass",
            "Transformer model",
            metadata={"type": "model"}
        )
        ltm.add_code_pattern(
            "lstm",
            "class LSTM: pass",
            "LSTM model",
            metadata={"type": "model"}
        )
        
        patterns = ltm.retrieve_code_patterns(
            keywords=["transformer"],
            top_k=1
        )
        
        assert len(patterns) == 1
        assert "transformer" in patterns[0][0].lower()
    
    def test_save_and_load(self, temp_storage):
        """Test persistence to disk."""
        ltm = LongTermMemory(temp_storage)
        
        # Add some data
        ltm.add_paper("test", {"title": "Test"}, embedding=np.array([1, 2, 3]))
        ltm.add_code_pattern("test_pattern", "code", "description")
        
        # Save
        ltm.save_to_disk()
        
        # Create new instance (should load from disk)
        ltm2 = LongTermMemory(temp_storage)
        
        assert len(ltm2.papers) == 1
        assert len(ltm2.code_patterns) == 1


class TestMemoryManager:
    """Tests for MemoryManager class."""
    
    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage = tmp_path / "manager_test"
        storage.mkdir()
        yield str(storage)
        shutil.rmtree(storage, ignore_errors=True)
    
    def test_memory_manager_initialization(self, temp_storage):
        """Test creating memory manager."""
        manager = MemoryManager(temp_storage)
        
        assert manager.long_term_memory is not None
        assert manager.short_term_memory is None
    
    def test_start_session(self, temp_storage):
        """Test starting a session."""
        manager = MemoryManager(temp_storage)
        
        stm = manager.start_session("session_789")
        
        assert manager.short_term_memory is not None
        assert stm.session_id == "session_789"
    
    def test_end_session(self, temp_storage, tmp_path):
        """Test ending a session."""
        manager = MemoryManager(temp_storage)
        
        stm = manager.start_session("session_abc")
        stm.add_code("test.py", "print('hello')")
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        manager.end_session(str(output_dir), save_to_long_term=False)
        
        assert manager.short_term_memory is None
    
    def test_retrieve_context(self, temp_storage):
        """Test retrieving similar context."""
        manager = MemoryManager(temp_storage)
        
        # Add some papers to long-term memory
        emb = np.array([1.0, 0.0, 0.0])
        manager.long_term_memory.add_paper(
            "test_paper",
            {"title": "Test"},
            embedding=emb
        )
        
        # Retrieve context
        query_emb = np.array([0.9, 0.1, 0.0])
        context = manager.retrieve_similar_context(
            query_embedding=query_emb,
            top_k=1
        )
        
        assert 'similar_papers' in context
        assert 'code_patterns' in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
