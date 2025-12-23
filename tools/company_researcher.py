"""
Company Research Tool for Cover Letter Generation

This module provides intelligent company research capabilities for tailoring
cover letters. It gathers information from multiple sources including:
- Web search for company overview and recent news
- Company website analysis
- Industry and culture insights
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

# For web search - we'll use free alternatives or OpenAI for now
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Structured Data
# ============================================================================

class CompanyNews(BaseModel):
    """Recent news or achievement about the company."""
    title: str = Field(description="News headline")
    summary: str = Field(description="Brief summary of the news")
    date: Optional[str] = Field(default=None, description="Publication date")
    source: Optional[str] = Field(default=None, description="News source")


class CompanyInfo(BaseModel):
    """Structured company information."""
    name: str = Field(description="Company name")
    industry: Optional[str] = Field(default=None, description="Industry or sector")
    size: Optional[str] = Field(default=None, description="Company size (employees)")
    description: str = Field(description="Company overview and mission")
    values: List[str] = Field(default_factory=list, description="Core company values")
    culture: Optional[str] = Field(default=None, description="Company culture description")
    recent_news: List[CompanyNews] = Field(default_factory=list, description="Recent news items")
    products_services: List[str] = Field(default_factory=list, description="Main products/services")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies used (if tech company)")
    achievements: List[str] = Field(default_factory=list, description="Recent achievements or milestones")


# ============================================================================
# Company Researcher Class
# ============================================================================

class CompanyResearcher:
    """
    Intelligent company research tool using GPT-4 and web search.
    
    This tool gathers comprehensive company information to enable
    personalized cover letter generation.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the company researcher.
        
        Args:
            model: OpenAI model to use for research analysis
            temperature: Temperature for LLM (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        # Cache for company research results
        self.cache_dir = "data/company_research"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"CompanyResearcher initialized with model: {model}")
    
    
    def research_company(
        self,
        company_name: str,
        job_description: Optional[str] = None,
        use_cache: bool = True
    ) -> CompanyInfo:
        """
        Research a company and gather comprehensive information.
        
        Args:
            company_name: Name of the company to research
            job_description: Optional job description for context
            use_cache: Whether to use cached results if available
        
        Returns:
            CompanyInfo object with structured company data
        """
        logger.info(f"Researching company: {company_name}")
        
        # Check cache first
        if use_cache:
            cached_info = self._load_from_cache(company_name)
            if cached_info:
                logger.info(f"Using cached research for {company_name}")
                return cached_info
        
        # Perform research using GPT-4
        company_info = self._research_with_llm(company_name, job_description)
        
        # Save to cache
        self._save_to_cache(company_name, company_info)
        
        return company_info
    
    
    def _research_with_llm(
        self,
        company_name: str,
        job_description: Optional[str] = None
    ) -> CompanyInfo:
        """
        Use GPT-4 to research company information.
        
        This uses the LLM's knowledge base to gather company information.
        For real-world use, this should be enhanced with web search APIs.
        
        Args:
            company_name: Company name
            job_description: Optional job description for context
        
        Returns:
            CompanyInfo object
        """
        # Create structured output parser
        parser = JsonOutputParser(pydantic_object=CompanyInfo)
        
        # Create prompt for company research
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional company researcher. Provide comprehensive, 
            accurate information about companies to help job seekers write personalized cover letters.
            
            {format_instructions}"""),
            ("user", """Research the following company and provide detailed information:
            
            Company Name: {company_name}
            
            {job_context}
            
            Provide:
            1. Company overview and mission
            2. Industry and company size
            3. Core values and culture
            4. Recent news or achievements (if known)
            5. Main products/services
            6. Technology stack (if tech company)
            7. Notable achievements or milestones
            
            Be factual and accurate. If you don't have certain information, leave those fields empty or minimal.
            Focus on information that would be useful for writing a compelling cover letter.""")
        ])
        
        # Prepare job context
        job_context = ""
        if job_description:
            job_context = f"Job Description Context:\n{job_description[:500]}\n\nUse this to focus your research on relevant aspects."
        
        # Format prompt
        formatted_prompt = prompt.format_messages(
            format_instructions=parser.get_format_instructions(),
            company_name=company_name,
            job_context=job_context
        )
        
        # Get LLM response
        try:
            response = self.llm.invoke(formatted_prompt)
            # Parse response - handle markdown code blocks
            import json
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                # Remove first line (```json or ```) and last line (```)
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines)
            
            # Parse JSON and create CompanyInfo object
            company_dict = json.loads(content)
            company_data = CompanyInfo(**company_dict)
            
            logger.info(f"Successfully researched {company_name}")
            return company_data
            
        except Exception as e:
            logger.error(f"Error researching company: {e}")
            # Return minimal company info on error
            return CompanyInfo(
                name=company_name,
                description=f"A company in the {company_name} industry.",
                values=[],
                recent_news=[],
                products_services=[],
                tech_stack=[],
                achievements=[]
            )
    
    
    def generate_company_insights(
        self,
        company_info: CompanyInfo,
        job_requirements: List[str]
    ) -> Dict[str, Any]:
        """
        Generate insights about how to align with the company.
        
        Args:
            company_info: Researched company information
            job_requirements: Key requirements from job description
        
        Returns:
            Dictionary with alignment insights
        """
        logger.info(f"Generating insights for {company_info.name if hasattr(company_info, 'name') else company_info.get('name', 'company')}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a career coach helping job seekers align their 
            background with company culture and values. Provide specific, actionable 
            insights for cover letter writing."""),
            ("user", """Given this company information:
            
            Company: {company_name}
            Industry: {industry}
            Description: {description}
            Values: {values}
            Culture: {culture}
            Recent News: {news}
            
            And these job requirements:
            {requirements}
            
            Provide:
            1. Key alignment points (3-5 specific ways the candidate can align with the company)
            2. Opening hook suggestions (2-3 compelling opening sentences mentioning recent news or achievements)
            3. Value proposition angles (how to frame your experience for this specific company)
            4. Tone recommendation (formal, conversational, innovative, etc.)
            
            Return as JSON with keys: alignment_points, opening_hooks, value_angles, tone""")
        ])
        
        # Handle both dict and CompanyInfo object
        if isinstance(company_info, dict):
            company_data = company_info
        else:
            company_data = company_info.dict() if hasattr(company_info, 'dict') else company_info.model_dump()
        
        # Format company values and news
        values_str = ", ".join(company_data.get('values', [])) if company_data.get('values') else "Not available"
        recent_news = company_data.get('recent_news', [])
        news_str = "; ".join([n.get('title', '') if isinstance(n, dict) else n.title for n in recent_news[:3]]) if recent_news else "Not available"
        requirements_str = "\n".join([f"- {req}" for req in job_requirements])
        
        formatted_prompt = prompt.format_messages(
            company_name=company_data.get('name', 'the company'),
            industry=company_data.get('industry') or "Not specified",
            description=company_data.get('description', ''),
            values=values_str,
            culture=company_data.get('culture') or "Not available",
            news=news_str,
            requirements=requirements_str
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            # Parse JSON response - handle markdown code blocks
            content = response.content.strip()
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            insights = json.loads(content)
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "alignment_points": ["Research the company's mission and values"],
                "opening_hooks": [f"I am excited about the opportunity at {company_info.name}"],
                "value_angles": ["Highlight your relevant experience"],
                "tone": "professional"
            }
    
    
    def _load_from_cache(self, company_name: str) -> Optional[CompanyInfo]:
        """Load company info from cache if available."""
        cache_file = os.path.join(self.cache_dir, f"{self._normalize_name(company_name)}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return CompanyInfo(**data)
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        
        return None
    
    
    def _save_to_cache(self, company_name: str, company_info: CompanyInfo):
        """Save company info to cache."""
        cache_file = os.path.join(self.cache_dir, f"{self._normalize_name(company_name)}.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                # Convert Pydantic model to dict
                data = company_info.dict() if hasattr(company_info, 'dict') else company_info.model_dump()
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached research for {company_name}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    
    @staticmethod
    def _normalize_name(company_name: str) -> str:
        """Normalize company name for file naming."""
        return company_name.lower().replace(" ", "_").replace(".", "").replace(",", "")


# ============================================================================
# Convenience Functions
# ============================================================================

def research_company(
    company_name: str,
    job_description: Optional[str] = None,
    use_cache: bool = True
) -> CompanyInfo:
    """
    Quick function to research a company.
    
    Args:
        company_name: Name of company to research
        job_description: Optional job description for context
        use_cache: Whether to use cached results
    
    Returns:
        CompanyInfo object
    """
    researcher = CompanyResearcher()
    return researcher.research_company(company_name, job_description, use_cache)
