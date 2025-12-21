"""
Job Hunter Agent - Unit Tests

This module contains comprehensive unit tests for the Job Hunter Agent system,
including tests for:
- LangChain tool wrappers
- Agent initialization and configuration
- Tool execution and error handling
- Memory management
- Session statistics

Author: Job Hunter Team
Date: December 21, 2025
"""

import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from tools.langchain_tools import (
    JobSearchTool,
    KeywordExtractionTool,
    SemanticMatchingTool,
    ResumeParsingTool,
    ResumeGenerationTool,
    get_all_tools,
    get_tool_descriptions
)
from agents.job_hunter_agent import JobHunterAgent, create_agent


# ============================================================================
# TOOL TESTS
# ============================================================================

class TestLangChainTools:
    """Test LangChain tool wrappers."""
    
    def test_get_all_tools(self):
        """Test getting all tools."""
        tools = get_all_tools()
        assert len(tools) == 5
        assert all(hasattr(tool, 'name') for tool in tools)
        assert all(hasattr(tool, 'description') for tool in tools)
    
    def test_get_tool_descriptions(self):
        """Test getting tool descriptions."""
        descriptions = get_tool_descriptions()
        assert len(descriptions) == 5
        assert 'job_search' in descriptions
        assert 'extract_keywords' in descriptions
        assert 'semantic_match' in descriptions
        assert 'parse_resume' in descriptions
        assert 'generate_tailored_resume' in descriptions
    
    def test_job_search_tool_initialization(self):
        """Test JobSearchTool initialization."""
        tool = JobSearchTool()
        assert tool.name == "job_search"
        assert "search for job postings" in tool.description.lower()
        assert hasattr(tool, '_run')
    
    def test_keyword_extraction_tool_initialization(self):
        """Test KeywordExtractionTool initialization."""
        tool = KeywordExtractionTool()
        assert tool.name == "extract_keywords"
        assert "extract" in tool.description.lower()
        assert "keywords" in tool.description.lower()
    
    def test_semantic_matching_tool_initialization(self):
        """Test SemanticMatchingTool initialization."""
        tool = SemanticMatchingTool()
        assert tool.name == "semantic_match"
        assert "similarity" in tool.description.lower()
    
    def test_resume_parsing_tool_initialization(self):
        """Test ResumeParsingTool initialization."""
        tool = ResumeParsingTool()
        assert tool.name == "parse_resume"
        assert "parse" in tool.description.lower()
    
    def test_resume_generation_tool_initialization(self):
        """Test ResumeGenerationTool initialization."""
        tool = ResumeGenerationTool()
        assert tool.name == "generate_tailored_resume"
        assert "generate" in tool.description.lower()
        assert "tailored" in tool.description.lower()


class TestToolExecution:
    """Test tool execution with mocked dependencies."""
    
    @patch('tools.langchain_tools.JobFetcher')
    def test_job_search_tool_success(self, mock_fetcher_class):
        """Test successful job search."""
        # Mock JobFetcher
        mock_fetcher = Mock()
        mock_fetcher.search_jobs.return_value = [
            {"title": "Senior Developer", "company": "TechCorp"}
        ]
        mock_fetcher_class.return_value = mock_fetcher
        
        # Run tool
        tool = JobSearchTool()
        result = tool._run(
            query="Python Developer",
            location="Remote",
            max_results=1
        )
        
        # Verify
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert result_dict["count"] == 1
        assert len(result_dict["jobs"]) == 1
    
    @patch('tools.langchain_tools.JobFetcher')
    def test_job_search_tool_error(self, mock_fetcher_class):
        """Test job search with error."""
        # Mock error
        mock_fetcher = Mock()
        mock_fetcher.search_jobs.side_effect = Exception("API Error")
        mock_fetcher_class.return_value = mock_fetcher
        
        # Run tool
        tool = JobSearchTool()
        result = tool._run(query="Test", location="Remote", max_results=1)
        
        # Verify error handling
        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "error" in result_dict
    
    @patch('tools.langchain_tools.ResumeParser')
    def test_resume_parsing_tool_file_not_found(self, mock_parser_class):
        """Test resume parsing with missing file."""
        tool = ResumeParsingTool()
        result = tool._run(file_path="/nonexistent/file.pdf")
        
        # Verify error handling
        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "not found" in result_dict["error"].lower()


# ============================================================================
# AGENT TESTS
# ============================================================================

class TestJobHunterAgent:
    """Test JobHunterAgent class."""
    
    def test_agent_initialization_without_api_key(self):
        """Test agent initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                JobHunterAgent()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    def test_agent_initialization_with_api_key(self, mock_llm):
        """Test successful agent initialization."""
        agent = JobHunterAgent()
        
        assert agent.model_name == "gpt-4o-mini"
        assert agent.temperature == 0.0
        assert agent.max_iterations == 15
        assert agent.max_retries == 3
        assert len(agent.tools) == 5
        assert agent.execution_count == 0
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    def test_agent_custom_configuration(self, mock_llm):
        """Test agent with custom configuration."""
        agent = JobHunterAgent(
            model_name="gpt-4",
            temperature=0.5,
            max_iterations=10,
            max_retries=5,
            verbose=False
        )
        
        assert agent.model_name == "gpt-4"
        assert agent.temperature == 0.5
        assert agent.max_iterations == 10
        assert agent.max_retries == 5
        assert agent.verbose is False
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    def test_agent_memory_initialization(self, mock_llm):
        """Test agent memory initialization."""
        agent = JobHunterAgent()
        
        assert agent.memory is not None
        assert hasattr(agent.memory, 'chat_memory')
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    def test_agent_reset_memory(self, mock_llm):
        """Test resetting agent memory."""
        agent = JobHunterAgent()
        
        # Add some fake messages
        agent.memory.save_context(
            {"input": "test question"},
            {"output": "test answer"}
        )
        
        # Reset
        agent.reset_memory()
        
        # Verify memory cleared
        history = agent.get_conversation_history()
        assert len(history) == 0
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    def test_get_session_stats(self, mock_llm):
        """Test getting session statistics."""
        agent = JobHunterAgent()
        stats = agent.get_session_stats()
        
        assert "session_id" in stats
        assert "model" in stats
        assert "executions" in stats
        assert "total_tokens_used" in stats
        assert "conversation_length" in stats
        assert "tools_available" in stats
        
        assert stats["executions"] == 0
        assert stats["total_tokens_used"] == 0
        assert stats["tools_available"] == 5
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    def test_token_estimation(self, mock_llm):
        """Test token usage estimation."""
        agent = JobHunterAgent()
        
        task = "This is a test task"
        output = "This is a test output"
        intermediate_steps = [
            ("Action: test", "Observation: result")
        ]
        
        tokens = agent._estimate_tokens(task, output, intermediate_steps)
        
        # Should be > 0 and include overhead
        assert tokens > 1000  # At least overhead
        assert isinstance(tokens, int)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    def test_create_agent_convenience_function(self, mock_llm):
        """Test create_agent convenience function."""
        agent = create_agent(verbose=False)
        
        assert isinstance(agent, JobHunterAgent)
        assert agent.verbose is False


class TestAgentExecution:
    """Test agent task execution."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    @patch('agents.job_hunter_agent.AgentExecutor')
    def test_successful_execution(self, mock_executor_class, mock_llm):
        """Test successful task execution."""
        # Mock agent executor
        mock_executor = Mock()
        mock_executor.invoke.return_value = {
            "output": "Task completed successfully",
            "intermediate_steps": [("test", "result")]
        }
        
        # Patch from_agent_and_tools to return our mock
        with patch('agents.job_hunter_agent.AgentExecutor.from_agent_and_tools', return_value=mock_executor):
            agent = JobHunterAgent()
            result = agent.run("Test task")
        
        assert result["success"] is True
        assert "output" in result
        assert "execution_time" in result
        assert "tokens_used" in result
        assert result["retry_count"] == 0
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    @patch('agents.job_hunter_agent.AgentExecutor')
    def test_execution_with_retry(self, mock_executor_class, mock_llm):
        """Test task execution with retries."""
        # Mock agent executor to fail once then succeed
        mock_executor = Mock()
        mock_executor.invoke.side_effect = [
            Exception("First attempt failed"),
            {
                "output": "Success on retry",
                "intermediate_steps": []
            }
        ]
        
        with patch('agents.job_hunter_agent.AgentExecutor.from_agent_and_tools', return_value=mock_executor):
            agent = JobHunterAgent()
            result = agent.run("Test task", retry_on_error=True)
        
        # Should succeed after retry
        assert result["success"] is True
        assert result["retry_count"] == 1
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('agents.job_hunter_agent.ChatOpenAI')
    @patch('agents.job_hunter_agent.AgentExecutor')
    def test_execution_max_retries_exceeded(self, mock_executor_class, mock_llm):
        """Test task execution when max retries exceeded."""
        # Mock agent executor to always fail
        mock_executor = Mock()
        mock_executor.invoke.side_effect = Exception("Always fails")
        
        with patch('agents.job_hunter_agent.AgentExecutor.from_agent_and_tools', return_value=mock_executor):
            agent = JobHunterAgent(max_retries=2)
            result = agent.run("Test task", retry_on_error=True)
        
        # Should fail after max retries
        assert result["success"] is False
        assert "error" in result
        assert result["retry_count"] == 3  # Initial + 2 retries


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests (require API keys)."""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key required"
    )
    def test_agent_initialization_real(self):
        """Test real agent initialization with API key."""
        agent = create_agent(verbose=False)
        
        assert agent is not None
        assert len(agent.tools) == 5
        assert agent.execution_count == 0
        
        # Cleanup
        agent.reset_memory()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
