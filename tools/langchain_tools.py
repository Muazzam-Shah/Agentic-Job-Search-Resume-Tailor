"""
LangChain Tool Wrappers for Job Hunter Components

This module wraps existing Job Hunter components (JobFetcher, KeywordExtractor, 
SemanticMatcher, ResumeParser, ResumeGenerator) as LangChain Tools for agent use.

Each tool is designed to be:
1. Descriptive - Clear purpose and usage instructions
2. Robust - Comprehensive error handling
3. Observable - Detailed logging of operations
4. Composable - Can be chained together in agent workflows

Author: Job Hunter Team
Date: December 21, 2025
"""

import os
import json
from typing import Optional, Dict, List, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from tools.job_fetcher import JobFetcher
from tools.keyword_extractor import KeywordExtractor
from tools.semantic_matcher import SemanticMatcher
from parsers.resume_parser import ResumeParser
from generators.resume_generator import ResumeGenerator
from rag.rag_retriever import RAGRetriever, RetrievalConfig, SearchStrategy, RetrievalMode
from rag.job_corpus import JobCorpusManager
from rag.resume_history import ResumeHistoryTracker
from utils.logger import logger



# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class JobSearchInput(BaseModel):
    """Input schema for job search tool."""
    query: str = Field(description="Job title to search for (e.g., 'Senior Python Developer')")
    location: str = Field(default="Remote", description="Job location (e.g., 'San Francisco, CA' or 'Remote')")
    max_results: int = Field(default=5, description="Maximum number of job results to return (1-20)")


class KeywordExtractionInput(BaseModel):
    """Input schema for keyword extraction tool."""
    job_description: str = Field(description="Full job description text to extract keywords from")


class SemanticMatchInput(BaseModel):
    """Input schema for semantic matching tool."""
    resume_text: str = Field(description="Resume text or summary to match")
    job_description: str = Field(description="Job description text to match against")


class ResumeParseInput(BaseModel):
    """Input schema for resume parsing tool."""
    file_path: str = Field(description="Absolute path to resume file (PDF or DOCX)")


class ResumeGenerationInput(BaseModel):
    """Input schema for resume generation tool."""
    resume_file_path: str = Field(description="Path to master resume file")
    job_description: str = Field(description="Target job description")
    company_name: str = Field(description="Company name for the target job")
    job_title: str = Field(description="Job title for the target role")
    output_dir: Optional[str] = Field(default="./output", description="Directory to save generated resume")


# ============================================================================
# LANGCHAIN TOOL IMPLEMENTATIONS
# ============================================================================

class JobSearchTool(BaseTool):
    """
    Tool for searching job postings using JSearch and Adzuna APIs.
    
    This tool aggregates job listings from multiple sources including LinkedIn,
    Indeed, Glassdoor, and company career pages.
    
    Returns: JSON string with list of job postings including title, company,
             location, description, requirements, and application links.
    """
    
    name: str = "job_search"
    description: str = """
    Search for job postings by title and location. Use this when you need to find
    jobs matching specific criteria. Returns detailed job information including
    descriptions, requirements, and application links.
    
    Input should include:
    - query: Job title (e.g., "Senior Python Developer")
    - location: Location or "Remote" (e.g., "San Francisco, CA")
    - max_results: Number of results (1-20, default 5)
    
    Example: {"query": "Data Scientist", "location": "New York", "max_results": 3}
    """
    args_schema: type[BaseModel] = JobSearchInput
    
    def _run(self, query: str, location: str = "Remote", max_results: int = 5) -> str:
        """Execute job search."""
        try:
            logger.info(f"Searching jobs: query='{query}', location='{location}', max={max_results}")
            
            # Initialize JobFetcher
            fetcher = JobFetcher()
            
            # Validate max_results
            if max_results < 1 or max_results > 20:
                max_results = min(max(max_results, 1), 20)
                logger.warning(f"Adjusted max_results to valid range: {max_results}")
            
            # Search jobs
            jobs = fetcher.search_jobs(
                query=query,
                location=location,
                max_results=max_results
            )
            
            if not jobs:
                return json.dumps({
                    "success": False,
                    "message": "No jobs found matching the criteria",
                    "jobs": []
                })
            
            logger.info(f"Found {len(jobs)} jobs")
            
            return json.dumps({
                "success": True,
                "count": len(jobs),
                "jobs": jobs
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Job search failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "jobs": []
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version - not implemented yet."""
        raise NotImplementedError("Async execution not supported yet")


class KeywordExtractionTool(BaseTool):
    """
    Tool for extracting keywords and requirements from job descriptions.
    
    Uses GPT-4o-mini to intelligently identify:
    - Required skills and technologies
    - Preferred/nice-to-have skills
    - Tools and platforms
    - Certifications
    - Soft skills
    - ATS optimization keywords
    
    Returns: Structured JSON with categorized keywords and importance rankings.
    """
    
    name: str = "extract_keywords"
    description: str = """
    Extract and categorize keywords from a job description. Use this to understand
    what skills, technologies, and qualifications are required for a position.
    Returns structured data with keywords ranked by importance.
    
    Input should be the full job description text.
    
    The output includes:
    - Required skills (must-have)
    - Preferred skills (nice-to-have)
    - Tools and technologies
    - Certifications
    - Soft skills
    - ATS optimization keywords
    """
    args_schema: type[BaseModel] = KeywordExtractionInput
    
    def _run(self, job_description: str) -> str:
        """Extract keywords from job description."""
        try:
            logger.info("Extracting keywords from job description")
            
            # Initialize extractor
            extractor = KeywordExtractor()
            
            # Extract keywords
            keywords = extractor.extract_keywords(job_description)
            
            # Convert to dict for JSON serialization
            result = {
                "success": True,
                "keywords": keywords.model_dump(),
                "summary": {
                    "required_skills_count": len(keywords.required_skills),
                    "preferred_skills_count": len(keywords.preferred_skills),
                    "tools_count": len(keywords.tools),
                    "certifications_count": len(keywords.certifications),
                    "soft_skills_count": len(keywords.soft_skills)
                }
            }
            
            logger.info(f"Extracted {sum(result['summary'].values())} total keywords")
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version - not implemented yet."""
        raise NotImplementedError("Async execution not supported yet")


class SemanticMatchingTool(BaseTool):
    """
    Tool for calculating semantic similarity between resume and job description.
    
    Uses OpenAI embeddings to compute:
    - Overall similarity score (0-100)
    - Section-wise analysis (summary, experience, skills)
    - ATS compatibility score
    - Missing keyword identification
    - Actionable recommendations
    
    Returns: Detailed match analysis with composite score and improvement suggestions.
    """
    
    name: str = "semantic_match"
    description: str = """
    Calculate how well a resume matches a job description using semantic analysis.
    Use this to evaluate resume-job fit and identify gaps.
    
    Input should include:
    - resume_text: The resume content or summary
    - job_description: The target job description
    
    Returns a comprehensive match analysis including:
    - Overall similarity score (0-100)
    - ATS compatibility score
    - Section-wise breakdown
    - Missing keywords
    - Recommendations for improvement
    """
    args_schema: type[BaseModel] = SemanticMatchInput
    
    def _run(self, resume_text: str, job_description: str) -> str:
        """Calculate semantic match between resume and job."""
        try:
            logger.info("Calculating semantic match")
            
            # Initialize matcher
            matcher = SemanticMatcher()
            
            # Calculate match
            match_result = matcher.calculate_match(
                resume_text=resume_text,
                job_description=job_description
            )
            
            # Convert to dict for JSON serialization
            result = {
                "success": True,
                "match_analysis": match_result.model_dump()
            }
            
            logger.info(f"Match score: {match_result.composite_score:.1f}% ({match_result.match_strength})")
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version - not implemented yet."""
        raise NotImplementedError("Async execution not supported yet")


class ResumeParsingTool(BaseTool):
    """
    Tool for parsing PDF and DOCX resumes into structured data.
    
    Uses GPT-4o-mini with regex fallbacks to extract:
    - Contact information (name, email, phone, LinkedIn, GitHub)
    - Professional summary
    - Work experience with dates and achievements
    - Education details
    - Skills and certifications
    - Projects and awards
    
    Returns: Structured JSON with all resume sections.
    """
    
    name: str = "parse_resume"
    description: str = """
    Parse a resume file (PDF or DOCX) into structured data. Use this to extract
    information from a master resume before tailoring it.
    
    Input should be the absolute file path to the resume.
    
    Returns structured data including:
    - Contact information
    - Professional summary
    - Work experience
    - Education
    - Skills
    - Certifications, projects, awards
    """
    args_schema: type[BaseModel] = ResumeParseInput
    
    def _run(self, file_path: str) -> str:
        """Parse resume file."""
        try:
            logger.info(f"Parsing resume: {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                return json.dumps({
                    "success": False,
                    "error": f"File not found: {file_path}"
                })
            
            # Initialize parser
            parser = ResumeParser()
            
            # Parse resume
            parsed_resume = parser.parse(file_path)
            
            # Convert to dict for JSON serialization
            result = {
                "success": True,
                "resume": parsed_resume.model_dump(),
                "summary": {
                    "has_contact": parsed_resume.contact_info is not None,
                    "has_summary": bool(parsed_resume.summary),
                    "experience_count": len(parsed_resume.experience),
                    "education_count": len(parsed_resume.education),
                    "skills_count": len(parsed_resume.skills)
                }
            }
            
            logger.info(f"Parsed resume: {result['summary']}")
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Resume parsing failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version - not implemented yet."""
        raise NotImplementedError("Async execution not supported yet")


class ResumeGenerationTool(BaseTool):
    """
    Tool for generating tailored resumes for specific job postings.
    
    Uses GPT-4o-mini to:
    - Select most relevant experiences based on job keywords
    - Optimize bullet points with natural keyword incorporation
    - Create tailored professional summary
    - Prioritize skills by job requirements
    - Generate ATS-friendly DOCX files
    
    Returns: Path to generated resume file with generation statistics.
    """
    
    name: str = "generate_tailored_resume"
    description: str = """
    Generate a tailored resume for a specific job posting. Use this after parsing
    the master resume and extracting job keywords.
    
    Input should include:
    - resume_file_path: Path to master resume
    - job_description: Target job description
    - company_name: Company name
    - job_title: Job title
    - output_dir: Directory to save resume (optional)
    
    Returns the path to the generated DOCX file and generation statistics.
    """
    args_schema: type[BaseModel] = ResumeGenerationInput
    
    def _run(
        self,
        resume_file_path: str,
        job_description: str,
        company_name: str,
        job_title: str,
        output_dir: str = "./output"
    ) -> str:
        """Generate tailored resume."""
        try:
            logger.info(f"Generating tailored resume for {company_name} - {job_title}")
            
            # Validate resume file exists
            if not os.path.exists(resume_file_path):
                return json.dumps({
                    "success": False,
                    "error": f"Resume file not found: {resume_file_path}"
                })
            
            # Create output directory if needed
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize components
            resume_parser = ResumeParser()
            keyword_extractor = KeywordExtractor()
            semantic_matcher = SemanticMatcher()
            resume_generator = ResumeGenerator(
                resume_parser=resume_parser,
                keyword_extractor=keyword_extractor,
                semantic_matcher=semantic_matcher
            )
            
            # Generate resume
            output_path = resume_generator.generate_tailored_resume(
                master_resume_path=resume_file_path,
                job_description=job_description,
                company_name=company_name,
                job_title=job_title,
                output_dir=output_dir
            )
            
            result = {
                "success": True,
                "output_file": output_path,
                "company": company_name,
                "job_title": job_title,
                "message": f"Successfully generated tailored resume: {os.path.basename(output_path)}"
            }
            
            logger.info(f"Resume generated: {output_path}")
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Resume generation failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version - not implemented yet."""
        raise NotImplementedError("Async execution not supported yet")


# ============================================================================
# RAG TOOLS
# ============================================================================

class RAGSearchInput(BaseModel):
    """Input schema for RAG search tool."""
    query: str = Field(description="Search query for retrieving relevant job descriptions or resume examples")
    k: int = Field(default=5, description="Number of results to retrieve (1-10)")
    mode: str = Field(default="jobs", description="Search mode: 'jobs', 'resumes', 'bullets', or 'mixed'")


class SuccessfulBulletsInput(BaseModel):
    """Input schema for successful bullets retrieval."""
    query: str = Field(description="Search query for finding proven bullet points from successful applications")
    k: int = Field(default=3, description="Number of bullets to retrieve (1-10)")


class RAGSearchTool(BaseTool):
    """
    RAG-powered search tool for retrieving relevant context.
    
    Uses vector similarity search to find:
    - Similar job descriptions from corpus
    - Successful resume examples
    - Proven bullet points from past applications
    
    Returns: JSON with retrieved documents and relevance scores
    """
    
    name: str = "rag_search"
    description: str = """
    Search the knowledge base for relevant job descriptions, resumes, or bullet points.
    Use this to find similar examples, proven patterns, or relevant context.
    
    Input should include:
    - query: What to search for (e.g., "Python AI developer jobs")
    - k: How many results (1-10, default 5)
    - mode: What to search - 'jobs', 'resumes', 'bullets', or 'mixed'
    
    Example: {"query": "machine learning engineer", "k": 3, "mode": "jobs"}
    """
    args_schema: type[BaseModel] = RAGSearchInput
    
    def _run(
        self,
        query: str,
        k: int = 5,
        mode: str = "jobs"
    ) -> str:
        """Execute RAG search."""
        try:
            logger.info(f"RAG Search: query='{query}', k={k}, mode={mode}")
            
            # Parse mode
            if mode == "jobs":
                retrieval_mode = RetrievalMode.JOBS
            elif mode == "resumes":
                retrieval_mode = RetrievalMode.RESUMES
            elif mode == "bullets":
                retrieval_mode = RetrievalMode.SUCCESSFUL_BULLETS
            elif mode == "mixed":
                retrieval_mode = RetrievalMode.MIXED
            else:
                retrieval_mode = RetrievalMode.JOBS
            
            # Create retriever (with default stores)
            from rag.vector_store import JobVectorStore
            
            job_store = JobVectorStore(collection_name="jobs", use_chromadb=True)
            resume_store = JobVectorStore(collection_name="resumes", use_chromadb=True)
            bullet_store = JobVectorStore(collection_name="bullets", use_chromadb=True)
            
            retriever = RAGRetriever(
                job_store=job_store,
                resume_store=resume_store,
                bullet_store=bullet_store
            )
            
            # Configure retrieval
            config = RetrievalConfig(
                k=min(k, 10),
                similarity_threshold=0.6,
                strategy=SearchStrategy.SEMANTIC
            )
            
            # Retrieve
            results = retriever.retrieve(query, config, retrieval_mode)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": round(result.score, 3),
                    "rank": result.rank,
                    "source": result.source
                })
            
            logger.info(f"Retrieved {len(formatted_results)} results")
            
            return json.dumps({
                "success": True,
                "query": query,
                "mode": mode,
                "count": len(formatted_results),
                "results": formatted_results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"RAG search failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version - not implemented yet."""
        raise NotImplementedError("Async execution not supported yet")


class SuccessfulBulletsTool(BaseTool):
    """
    Retrieve proven bullet points from successful applications.
    
    Searches historical resume data to find bullet points that led to
    interviews or job offers. Great for inspiration and proven patterns.
    
    Returns: JSON with successful bullet points and their context
    """
    
    name: str = "get_successful_bullets"
    description: str = """
    Find proven bullet points from past successful job applications.
    Use this to get inspiration from resume sections that led to interviews/offers.
    
    Input should include:
    - query: What kind of bullet point (e.g., "AI project achievements")
    - k: How many bullets to retrieve (1-10, default 3)
    
    Example: {"query": "leadership experience", "k": 5}
    """
    args_schema: type[BaseModel] = SuccessfulBulletsInput
    
    def _run(
        self,
        query: str,
        k: int = 3
    ) -> str:
        """Retrieve successful bullets."""
        try:
            logger.info(f"Retrieving successful bullets: query='{query}', k={k}")
            
            # Initialize tracker
            tracker = ResumeHistoryTracker()
            
            # Get successful bullets
            bullets = tracker.get_successful_bullets(
                query=query,
                k=min(k, 10),
                min_similarity=0.65,
                outcome_filter=["interview", "offer"]
            )
            
            # Format results
            formatted_bullets = []
            for bullet in bullets:
                formatted_bullets.append({
                    "bullet": bullet["content"],
                    "metadata": bullet["metadata"],
                    "similarity": round(bullet["similarity"], 3)
                })
            
            logger.info(f"Found {len(formatted_bullets)} successful bullets")
            
            return json.dumps({
                "success": True,
                "query": query,
                "count": len(formatted_bullets),
                "bullets": formatted_bullets
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to retrieve successful bullets: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version - not implemented yet."""
        raise NotImplementedError("Async execution not supported yet")


# ============================================================================
# TOOL REGISTRY
# ============================================================================

def get_all_tools() -> List[BaseTool]:
    """
    Get all available Job Hunter tools.
    
    Returns:
        List of initialized LangChain tools
    """
    return [
        JobSearchTool(),
        KeywordExtractionTool(),
        SemanticMatchingTool(),
        ResumeParsingTool(),
        ResumeGenerationTool(),
        RAGSearchTool(),
        SuccessfulBulletsTool()
    ]


def get_tool_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all tools.
    
    Returns:
        Dictionary mapping tool names to descriptions
    """
    tools = get_all_tools()
    return {tool.name: tool.description for tool in tools}


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test tool functionality."""
    
    print("=" * 80)
    print("JOB HUNTER LANGCHAIN TOOLS TEST")
    print("=" * 80)
    
    # Get all tools
    tools = get_all_tools()
    
    print(f"\n✅ Loaded {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:100]}...")
    
    print("\n" + "=" * 80)
    print("Tool descriptions loaded successfully!")
    print("=" * 80)
