"""
Job Hunter Agent - Main Orchestrator

This module implements the core agentic AI system using LangChain's ReAct pattern.
The agent orchestrates multiple tools to automate resume tailoring:

1. Job Search - Find relevant job postings
2. Keyword Extraction - Identify key requirements
3. Semantic Matching - Analyze resume-job fit
4. Resume Parsing - Extract structured resume data
5. Resume Generation - Create tailored resumes

The agent uses:
- ReAct (Reasoning + Acting) for task planning
- Conversation memory for context retention
- Error handling with retries for robustness
- Structured logging for observability

Author: Job Hunter Team
Date: December 21, 2025
"""

import os
import time
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish

from tools.langchain_tools import get_all_tools
from utils.logger import logger



# ============================================================================
# REACT PROMPT TEMPLATE
# ============================================================================

REACT_PROMPT = """You are Job Hunter, an expert AI assistant specialized in helping job seekers create perfectly tailored resumes for specific job postings.

You have access to the following tools:

{tools}

Tool Names: {tool_names}

Your primary goal is to help users create ATS-optimized, keyword-rich resumes that match specific job requirements.

WORKFLOW GUIDELINES:

When a user asks to tailor a resume for a job:
1. If given a job title/location, use job_search to find relevant postings
2. Use extract_keywords to identify key requirements from the job description
3. Use parse_resume to extract structured data from the master resume
4. Use semantic_match to analyze current resume-job fit and identify gaps
5. Use generate_tailored_resume to create the optimized resume
6. Provide the user with the file path and a summary of changes made

IMPORTANT RULES:
- Always think step-by-step before taking action
- Validate inputs before using tools
- If a tool fails, try an alternative approach
- Provide clear, actionable feedback to the user
- Never make up information - only use tool outputs
- Keep file paths absolute and properly formatted

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# ============================================================================
# AGENT ORCHESTRATOR
# ============================================================================

class JobHunterAgent:
    """
    Main agent orchestrator for Job Hunter system.
    
    This class manages the agentic AI workflow using LangChain's ReAct pattern.
    It coordinates multiple tools to automate resume tailoring from job search
    to final resume generation.
    
    Attributes:
        llm: Language model for agent reasoning (GPT-4)
        tools: List of available LangChain tools
        memory: Conversation memory for context retention
        agent_executor: LangChain agent executor
        max_iterations: Maximum reasoning steps per task
        max_retries: Maximum retry attempts for failed operations
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_iterations: int = 15,
        max_retries: int = 3,
        verbose: bool = True
    ):
        """
        Initialize Job Hunter Agent.
        
        Args:
            openai_api_key: OpenAI API key (defaults to env var)
            model_name: LLM model to use (default: gpt-4o-mini)
            temperature: LLM temperature for reasoning (default: 0.0 for determinism)
            max_iterations: Max agent reasoning steps (default: 15)
            max_retries: Max retry attempts for failed operations (default: 3)
            verbose: Enable verbose logging (default: True)
        """
        logger.info("Initializing Job Hunter Agent")
        
        # Get API key
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )
        
        # Configuration
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.verbose = verbose
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {model_name} (temp={temperature})")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self.api_key
        )
        
        # Load tools
        logger.info("Loading tools")
        self.tools = get_all_tools()
        logger.info(f"Loaded {len(self.tools)} tools: {[t.name for t in self.tools]}")
        
        # Initialize memory
        logger.info("Initializing conversation memory")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Create agent
        logger.info("Creating ReAct agent")
        self.agent = self._create_agent()
        
        # Create agent executor
        logger.info("Creating agent executor")
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.execution_count = 0
        self.total_tokens_used = 0
        
        logger.info(f"✅ Job Hunter Agent initialized (session: {self.session_id})")
    
    def _create_agent(self):
        """Create ReAct agent with custom prompt."""
        prompt = PromptTemplate(
            template=REACT_PROMPT,
            input_variables=["input", "agent_scratchpad", "chat_history"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
        
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    def run(self, task: str, retry_on_error: bool = True) -> Dict[str, Any]:
        """
        Execute a task using the agent.
        
        Args:
            task: Natural language task description
            retry_on_error: Whether to retry on failures (default: True)
        
        Returns:
            Dictionary with execution results:
            {
                "success": bool,
                "output": str,
                "intermediate_steps": List[Tuple[AgentAction, str]],
                "execution_time": float,
                "tokens_used": int (estimated),
                "error": Optional[str]
            }
        """
        self.execution_count += 1
        execution_id = f"{self.session_id}_{self.execution_count}"
        
        logger.info("=" * 80)
        logger.info(f"EXECUTING TASK #{self.execution_count}")
        logger.info(f"Task: {task[:100]}...")
        logger.info("=" * 80)
        
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Execute agent
                logger.info(f"Attempt {retry_count + 1}/{self.max_retries + 1}")
                result = self.agent_executor.invoke({"input": task})
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Extract results
                output = result.get("output", "")
                intermediate_steps = result.get("intermediate_steps", [])
                
                # Estimate token usage (rough approximation)
                estimated_tokens = self._estimate_tokens(task, output, intermediate_steps)
                self.total_tokens_used += estimated_tokens
                
                logger.info("=" * 80)
                logger.info(f"✅ TASK COMPLETED (ID: {execution_id})")
                logger.info(f"Execution time: {execution_time:.2f}s")
                logger.info(f"Estimated tokens: {estimated_tokens}")
                logger.info(f"Steps taken: {len(intermediate_steps)}")
                logger.info("=" * 80)
                
                return {
                    "success": True,
                    "output": output,
                    "intermediate_steps": intermediate_steps,
                    "execution_time": execution_time,
                    "tokens_used": estimated_tokens,
                    "execution_id": execution_id,
                    "retry_count": retry_count
                }
                
            except Exception as e:
                retry_count += 1
                last_error = str(e)
                logger.error(f"❌ Attempt {retry_count} failed: {last_error}")
                
                if not retry_on_error or retry_count > self.max_retries:
                    break
                
                # Exponential backoff
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        # All retries failed
        execution_time = time.time() - start_time
        logger.error("=" * 80)
        logger.error(f"❌ TASK FAILED (ID: {execution_id})")
        logger.error(f"Error: {last_error}")
        logger.error("=" * 80)
        
        return {
            "success": False,
            "output": "",
            "intermediate_steps": [],
            "execution_time": execution_time,
            "tokens_used": 0,
            "execution_id": execution_id,
            "retry_count": retry_count,
            "error": last_error
        }
    
    def _estimate_tokens(
        self,
        task: str,
        output: str,
        intermediate_steps: List[tuple]
    ) -> int:
        """
        Estimate total tokens used (rough approximation).
        
        Args:
            task: Input task
            output: Final output
            intermediate_steps: Intermediate reasoning steps
        
        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters
        task_tokens = len(task) // 4
        output_tokens = len(output) // 4
        
        # Estimate tokens from intermediate steps
        intermediate_tokens = 0
        for step in intermediate_steps:
            if len(step) >= 2:
                action = str(step[0])
                observation = str(step[1])
                intermediate_tokens += (len(action) + len(observation)) // 4
        
        # Add overhead for system prompts (~1000 tokens)
        overhead = 1000
        
        return task_tokens + output_tokens + intermediate_tokens + overhead
    
    def reset_memory(self):
        """Clear conversation memory."""
        logger.info("Clearing conversation memory")
        self.memory.clear()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation messages
        """
        messages = self.memory.chat_memory.messages
        return [{"role": msg.type, "content": msg.content} for msg in messages]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session stats
        """
        return {
            "session_id": self.session_id,
            "model": self.model_name,
            "executions": self.execution_count,
            "total_tokens_used": self.total_tokens_used,
            "conversation_length": len(self.memory.chat_memory.messages),
            "tools_available": len(self.tools)
        }
    
    def tailor_resume_for_job(
        self,
        master_resume_path: str,
        job_query: Optional[str] = None,
        job_location: str = "Remote",
        job_description: Optional[str] = None,
        company_name: Optional[str] = None,
        job_title: Optional[str] = None,
        output_dir: str = "./output"
    ) -> Dict[str, Any]:
        """
        High-level method to tailor resume for a job.
        
        This is a convenience method that constructs the appropriate task
        for the agent based on available information.
        
        Args:
            master_resume_path: Path to master resume file
            job_query: Job search query (e.g., "Senior Python Developer")
            job_location: Job location (default: "Remote")
            job_description: Full job description (if already have it)
            company_name: Company name (if known)
            job_title: Job title (if known)
            output_dir: Directory to save tailored resume
        
        Returns:
            Agent execution results
        """
        # Validate master resume exists
        if not os.path.exists(master_resume_path):
            return {
                "success": False,
                "error": f"Master resume not found: {master_resume_path}"
            }
        
        # Construct task based on available information
        if job_description and company_name and job_title:
            # Have all information - direct generation
            task = f"""
I need you to create a tailored resume for this job:

Company: {company_name}
Job Title: {job_title}
Master Resume: {master_resume_path}
Output Directory: {output_dir}

Job Description:
{job_description}

Please:
1. Parse my master resume
2. Extract keywords from the job description
3. Analyze the match between my resume and the job
4. Generate a tailored resume optimized for this position
5. Save it to the output directory with proper naming

Provide me with the file path and a summary of the changes made.
"""
        elif job_query:
            # Need to search for jobs first
            task = f"""
I'm looking for {job_query} positions in {job_location}.

Please:
1. Search for relevant job postings
2. Show me the top 3 matches with company names and job titles
3. For the best match, extract keywords and analyze my resume fit
4. Generate a tailored resume for that position

My master resume is at: {master_resume_path}
Save tailored resume to: {output_dir}

Show me the match analysis and file path for the generated resume.
"""
        else:
            return {
                "success": False,
                "error": "Either provide (job_query) or (job_description + company_name + job_title)"
            }
        
        # Execute task
        return self.run(task)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_agent(**kwargs) -> JobHunterAgent:
    """
    Create a Job Hunter Agent with default settings.
    
    Args:
        **kwargs: Optional overrides for agent configuration
    
    Returns:
        Initialized JobHunterAgent instance
    """
    return JobHunterAgent(**kwargs)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test agent initialization and basic functionality."""
    
    print("=" * 80)
    print("JOB HUNTER AGENT TEST")
    print("=" * 80)
    
    try:
        # Initialize agent
        print("\n1. Initializing agent...")
        agent = create_agent(verbose=True)
        
        print("\n2. Agent initialized successfully!")
        print(f"   Session ID: {agent.session_id}")
        print(f"   Model: {agent.model_name}")
        print(f"   Tools: {len(agent.tools)}")
        
        # Show session stats
        print("\n3. Session Statistics:")
        stats = agent.get_session_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n4. Available Tools:")
        for tool in agent.tools:
            print(f"   - {tool.name}: {tool.description[:80]}...")
        
        print("\n" + "=" * 80)
        print("✅ Agent test completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise
