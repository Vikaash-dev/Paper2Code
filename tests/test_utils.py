"""Unit tests for utility functions in Paper2Code.

This module tests the core utility functions including:
- Planning extraction
- JSON parsing with fallbacks
- Code extraction
- Cost calculation
"""

import pytest
import json
import tempfile
import os
from codes.utils import (
    extract_planning,
    content_to_json,
    extract_code_from_content,
    extract_code_from_content2,
    format_json_data,
    get_now_str
)


class TestExtractPlanning:
    """Tests for extract_planning function."""
    
    def test_extract_planning_basic(self, tmp_path):
        """Test basic planning extraction."""
        # Create test trajectories
        trajectories = [
            {"role": "user", "content": "Please plan"},
            {"role": "assistant", "content": "Here is the plan"},
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": "Here is more detail"},
            {"role": "assistant", "content": "Final thoughts"},
            {"role": "assistant", "content": "Extra (should be ignored)"}
        ]
        
        # Write to temp file
        traj_file = tmp_path / "test_traj.json"
        with open(traj_file, 'w') as f:
            json.dump(trajectories, f)
        
        # Extract planning
        result = extract_planning(str(traj_file))
        
        # Should extract first 3 assistant responses
        assert len(result) == 3
        assert result[0] == "Here is the plan"
        assert result[1] == "Here is more detail"
        assert result[2] == "Final thoughts"
    
    def test_extract_planning_with_think_tags(self, tmp_path):
        """Test extraction with thinking tags."""
        trajectories = [
            {"role": "assistant", "content": "<think>Internal thought</think>Actual response"}
        ]
        
        traj_file = tmp_path / "test_traj_think.json"
        with open(traj_file, 'w') as f:
            json.dump(trajectories, f)
        
        result = extract_planning(str(traj_file))
        
        assert len(result) == 1
        assert result[0] == "Actual response"
        assert "<think>" not in result[0]


class TestContentToJson:
    """Tests for content_to_json function and its fallbacks."""
    
    def test_clean_json(self):
        """Test parsing clean JSON."""
        content = '[CONTENT]{"key": "value", "list": [1, 2, 3]}[/CONTENT]'
        result = content_to_json(content)
        
        assert result["key"] == "value"
        assert result["list"] == [1, 2, 3]
    
    def test_json_with_comments(self):
        """Test parsing JSON with inline comments."""
        content = '''[CONTENT]{
            "key": "value", # This is a comment
            "number": 42
        }[/CONTENT]'''
        
        result = content_to_json(content)
        
        assert result["key"] == "value"
        assert result["number"] == 42
    
    def test_json_with_trailing_comma(self):
        """Test parsing JSON with trailing commas."""
        content = '[CONTENT]{"key": "value", "list": [1, 2, 3,]}[/CONTENT]'
        result = content_to_json(content)
        
        assert "key" in result
        assert "list" in result
    
    def test_fallback_to_logic_analysis(self):
        """Test fallback to extracting Logic Analysis and Task list."""
        content = '''
        Some text before
        "Logic Analysis": [["file1.py", "Description 1"], ["file2.py", "Description 2"]],
        "Task list": ["file1.py", "file2.py"]
        Some text after
        '''
        
        result = content_to_json(content)
        
        assert "Logic Analysis" in result
        assert "Task list" in result
        assert len(result["Logic Analysis"]) == 2
        assert len(result["Task list"]) == 2


class TestCodeExtraction:
    """Tests for code extraction functions."""
    
    def test_extract_code_basic(self):
        """Test basic code extraction."""
        content = '''
        Some text before
        ```python
        def hello():
            print("Hello, world!")
        ```
        Some text after
        '''
        
        result = extract_code_from_content(content)
        
        assert "def hello():" in result
        assert "print" in result
    
    def test_extract_code_no_language(self):
        """Test extraction without language specifier."""
        content = '''
        ```
        generic code
        ```
        '''
        
        result = extract_code_from_content(content)
        assert "generic code" in result
    
    def test_extract_code_empty(self):
        """Test extraction with no code blocks."""
        content = "Just plain text with no code blocks"
        
        result = extract_code_from_content(content)
        assert result == ""
    
    def test_extract_python_code(self):
        """Test Python-specific extraction."""
        content = '''
        ```python
        import numpy as np
        
        def process():
            return np.array([1, 2, 3])
        ```
        '''
        
        result = extract_code_from_content2(content)
        
        assert "import numpy" in result
        assert "def process()" in result


class TestFormatting:
    """Tests for formatting functions."""
    
    def test_format_json_data(self):
        """Test JSON data formatting."""
        data = {
            "Section 1": "Single value",
            "Section 2": ["Item 1", "Item 2", "Item 3"]
        }
        
        result = format_json_data(data)
        
        assert "[Section 1]" in result
        assert "[Section 2]" in result
        assert "Single value" in result
        assert "- Item 1" in result
        assert "-" * 40 in result
    
    def test_get_now_str_format(self):
        """Test timestamp string format."""
        result = get_now_str()
        
        # Should be in format: YYYYMMdd_HHmmss
        assert len(result) == 15  # 8 digits + _ + 6 digits
        assert "_" in result
        assert result.replace("_", "").isdigit()


class TestCostCalculation:
    """Tests for cost calculation (requires API response format)."""
    
    def test_cal_cost_structure(self):
        """Test cost calculation with mock response."""
        from codes.utils import cal_cost
        
        # Mock API response structure
        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "prompt_tokens_details": {
                    "cached_tokens": 200
                }
            }
        }
        
        result = cal_cost(response, "o3-mini")
        
        assert "model_name" in result
        assert "total_cost" in result
        assert "input_cost" in result
        assert "output_cost" in result
        assert result["actual_input_tokens"] == 800  # 1000 - 200 cached
        assert result["cached_tokens"] == 200


# Fixtures
@pytest.fixture
def sample_paper_json():
    """Sample paper JSON for testing."""
    return {
        "title": "Test Paper",
        "abstract": "This is a test abstract",
        "sections": [
            {
                "heading": "Introduction",
                "text": "Introduction text"
            },
            {
                "heading": "Methodology",
                "text": "Methodology text"
            }
        ]
    }


@pytest.fixture
def sample_trajectories():
    """Sample trajectories for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Please plan"},
        {"role": "assistant", "content": "Here is the plan"},
        {"role": "user", "content": "Continue"},
        {"role": "assistant", "content": "More details"}
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
