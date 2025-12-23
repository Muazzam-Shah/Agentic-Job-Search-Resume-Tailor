"""
Cover Letter Generator with Company Research Integration

This module generates personalized, professional cover letters tailored to
specific job postings and companies using GPT-4 and company research.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import company researcher
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.company_researcher import CompanyResearcher, CompanyInfo
from parsers.resume_parser import ParsedResume


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Cover Letter Structure
# ============================================================================

class CoverLetterHeader(BaseModel):
    """Cover letter header information."""
    candidate_name: str
    candidate_address: Optional[str] = None
    candidate_phone: str
    candidate_email: str
    date: str
    hiring_manager: Optional[str] = Field(default="Hiring Manager", description="Hiring manager name")
    company_name: str
    company_address: Optional[str] = None


class CoverLetterContent(BaseModel):
    """Structured cover letter content."""
    opening_paragraph: str = Field(description="Compelling opening with company-specific hook")
    body_paragraph_1: str = Field(description="Why this company - alignment with mission/values")
    body_paragraph_2: str = Field(description="Why you - relevant achievements and skills")
    body_paragraph_3: Optional[str] = Field(default=None, description="Value proposition - what you bring")
    closing_paragraph: str = Field(description="Strong closing with call-to-action")
    salutation: str = Field(default="Sincerely", description="Closing salutation")


class GeneratedCoverLetter(BaseModel):
    """Complete generated cover letter."""
    header: CoverLetterHeader
    content: CoverLetterContent
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")


# ============================================================================
# Cover Letter Generator Class
# ============================================================================

class CoverLetterGenerator:
    """
    Intelligent cover letter generator with company research.
    
    This class generates personalized cover letters by:
    1. Researching the target company
    2. Analyzing job requirements
    3. Mapping resume achievements to requirements
    4. Crafting compelling, tailored content
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the cover letter generator.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for creative writing (0.7 recommended)
        """
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.researcher = CompanyResearcher()
        
        logger.info(f"CoverLetterGenerator initialized with model: {model}")
    
    
    def generate_cover_letter(
        self,
        parsed_resume: ParsedResume,
        company_name: str,
        job_title: str,
        job_description: str,
        hiring_manager: Optional[str] = None,
        company_address: Optional[str] = None,
        style: str = "professional"
    ) -> GeneratedCoverLetter:
        """
        Generate a tailored cover letter.
        
        Args:
            parsed_resume: Parsed resume data
            company_name: Target company name
            job_title: Job title applying for
            job_description: Full job description
            hiring_manager: Hiring manager name (optional)
            company_address: Company address (optional)
            style: Cover letter style (professional, creative, technical)
        
        Returns:
            GeneratedCoverLetter object with complete cover letter
        """
        logger.info(f"Generating cover letter for {company_name} - {job_title}")
        
        # Step 1: Research company
        company_info = self.researcher.research_company(company_name, job_description)
        
        # Step 2: Extract job requirements
        requirements = self._extract_requirements(job_description)
        
        # Step 3: Generate company insights
        insights = self.researcher.generate_company_insights(company_info, requirements)
        
        # Step 4: Map achievements to requirements
        achievement_mapping = self._map_achievements(parsed_resume, requirements)
        
        # Step 5: Generate cover letter content
        content = self._generate_content(
            parsed_resume=parsed_resume,
            company_info=company_info,
            job_title=job_title,
            job_description=job_description,
            insights=insights,
            achievement_mapping=achievement_mapping,
            style=style
        )
        
        # Step 6: Create header
        header = CoverLetterHeader(
            candidate_name=parsed_resume.contact_info.name,
            candidate_phone=parsed_resume.contact_info.phone or "",
            candidate_email=parsed_resume.contact_info.email or "",
            date=datetime.now().strftime("%B %d, %Y"),
            hiring_manager=hiring_manager or "Hiring Manager",
            company_name=company_name,
            company_address=company_address
        )
        
        # Step 7: Assemble complete cover letter
        cover_letter = GeneratedCoverLetter(
            header=header,
            content=content,
            metadata={
                "job_title": job_title,
                "company_name": company_name,
                "generation_date": datetime.now().isoformat(),
                "model": self.model,
                "style": style
            }
        )
        
        logger.info("Cover letter generation complete")
        return cover_letter
    
    
    def _extract_requirements(self, job_description: str) -> List[str]:
        """Extract key requirements from job description."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing job descriptions and extracting key requirements."),
            ("user", """Analyze this job description and extract the top 5-7 most important requirements:
            
            {job_description}
            
            Return as a JSON array of strings: ["requirement 1", "requirement 2", ...]
            Focus on skills, experience, and qualifications that should be highlighted in a cover letter.""")
        ])
        
        try:
            formatted = prompt.format_messages(job_description=job_description)
            response = self.llm.invoke(formatted)
            
            # Parse JSON array - handle potential markdown code blocks
            import json
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            requirements = json.loads(content)
            return requirements[:7]  # Top 7 max
            
        except Exception as e:
            logger.error(f"Error extracting requirements: {e}")
            return ["Relevant experience in the field"]
    
    
    def _map_achievements(
        self,
        parsed_resume: ParsedResume,
        requirements: List[str]
    ) -> Dict[str, List[str]]:
        """
        Map resume achievements to job requirements.
        
        Returns:
            Dictionary mapping requirements to relevant achievements
        """
        # Collect all achievements from resume
        all_achievements = []
        for exp in parsed_resume.experience:
            all_achievements.extend(exp.description)  # Changed from exp.bullets to exp.description
        
        # Use GPT to map achievements to requirements
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a career coach mapping resume achievements to job requirements.
            For each requirement, identify the 2-3 most relevant achievements from the resume."""),
            ("user", """Job Requirements:
            {requirements}
            
            Resume Achievements:
            {achievements}
            
            For each requirement, select the 2-3 most relevant achievements.
            Return as JSON: {{"requirement": ["achievement 1", "achievement 2"], ...}}""")
        ])
        
        try:
            formatted = prompt.format_messages(
                requirements="\n".join([f"{i+1}. {req}" for i, req in enumerate(requirements)]),
                achievements="\n".join([f"- {ach}" for ach in all_achievements])
            )
            response = self.llm.invoke(formatted)
            
            import json
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            mapping = json.loads(content)
            return mapping
            
        except Exception as e:
            logger.error(f"Error mapping achievements: {e}")
            return {req: all_achievements[:2] for req in requirements}
    
    
    def _generate_content(
        self,
        parsed_resume: ParsedResume,
        company_info: CompanyInfo,
        job_title: str,
        job_description: str,
        insights: Dict[str, Any],
        achievement_mapping: Dict[str, List[str]],
        style: str
    ) -> CoverLetterContent:
        """Generate the main content of the cover letter."""
        
        # Create parser for structured output
        parser = JsonOutputParser(pydantic_object=CoverLetterContent)
        
        # Handle both dict and CompanyInfo object
        company_data = company_info if isinstance(company_info, dict) else company_info.dict()
        
        # Prepare context
        recent_news = ""
        news_list = company_data.get('recent_news', [])
        if news_list:
            first_news = news_list[0]
            news_title = first_news.get('title', '') if isinstance(first_news, dict) else first_news.title
            recent_news = f"Recent News: {news_title}"
        
        values_str = ", ".join(company_data.get('values', [])[:3]) if company_data.get('values') else "innovation and excellence"
        
        # Create comprehensive prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert cover letter writer who creates compelling, 
            personalized cover letters that stand out. Write in a {style} tone.
            
            {format_instructions}"""),
            ("user", """Write a compelling cover letter for:
            
            CANDIDATE: {candidate_name}
            POSITION: {job_title} at {company_name}
            
            COMPANY INFORMATION:
            - Industry: {industry}
            - Mission: {description}
            - Values: {values}
            - {recent_news}
            
            JOB REQUIREMENTS:
            {requirements}
            
            CANDIDATE'S RELEVANT ACHIEVEMENTS:
            {achievements}
            
            INSIGHTS:
            - Opening Hooks: {opening_hooks}
            - Alignment Points: {alignment_points}
            
            Create a cover letter with:
            
            1. OPENING PARAGRAPH (3-4 sentences):
               - Hook with company-specific reference (recent news, mission, or achievement)
               - State position and express genuine enthusiasm
               - Brief value proposition
            
            2. BODY PARAGRAPH 1 (4-5 sentences):
               - Why THIS company specifically
               - Align with company values and mission
               - Show you've researched them
               - Connect your background to their goals
            
            3. BODY PARAGRAPH 2 (4-5 sentences):
               - Why YOU specifically
               - Highlight 2-3 most relevant achievements with metrics
               - Connect achievements directly to job requirements
               - Demonstrate impact and value
            
            4. BODY PARAGRAPH 3 (3-4 sentences) [OPTIONAL]:
               - Future value proposition
               - What you will bring to the role
               - Specific contributions you can make
            
            5. CLOSING PARAGRAPH (2-3 sentences):
               - Reiterate enthusiasm
               - Clear call-to-action (interview request)
               - Professional thanks
            
            Guidelines:
            - Be specific and avoid generic phrases
            - Use concrete examples and numbers
            - Show genuine interest and research
            - Keep professional but personable
            - Vary sentence structure
            - Total length: 300-400 words
            """)
        ])
        
        # Format achievements for prompt
        achievement_list = []
        for req, achs in achievement_mapping.items():
            achievement_list.extend(achs[:2])  # Top 2 per requirement
        achievement_str = "\n".join([f"- {ach}" for ach in list(set(achievement_list))[:6]])
        
        # Format requirements
        requirements_str = "\n".join([f"- {req}" for req in list(achievement_mapping.keys())[:5]])
        
        # Format opening hooks
        opening_hooks_str = "\n".join([f"- {hook}" for hook in insights.get("opening_hooks", [])[:2]])
        alignment_str = "\n".join([f"- {point}" for point in insights.get("alignment_points", [])[:3]])
        
        # Generate content
        formatted_prompt = prompt.format_messages(
            style=style,
            format_instructions=parser.get_format_instructions(),
            candidate_name=parsed_resume.contact_info.name,
            job_title=job_title,
            company_name=company_data.get('name', 'the company'),
            industry=company_data.get('industry') or "the industry",
            description=company_data.get('description', '')[:200],
            values=values_str,
            recent_news=recent_news,
            requirements=requirements_str,
            achievements=achievement_str,
            opening_hooks=opening_hooks_str,
            alignment_points=alignment_str
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            content_data = parser.parse(response.content)
            logger.info("Successfully generated cover letter content")
            return content_data
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            # Return basic cover letter on error
            company_name = company_data.get('name', 'the company')
            return CoverLetterContent(
                opening_paragraph=f"I am writing to express my strong interest in the {job_title} position at {company_name}.",
                body_paragraph_1=f"I am impressed by {company_name}'s commitment to {values_str}.",
                body_paragraph_2="My background and experience make me an ideal candidate for this role.",
                closing_paragraph="Thank you for considering my application. I look forward to discussing this opportunity further."
            )


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_cover_letter(
    parsed_resume: ParsedResume,
    company_name: str,
    job_title: str,
    job_description: str,
    **kwargs
) -> GeneratedCoverLetter:
    """
    Quick function to generate a cover letter.
    
    Args:
        parsed_resume: Parsed resume data
        company_name: Target company
        job_title: Job title
        job_description: Full job description
        **kwargs: Additional arguments (hiring_manager, style, etc.)
    
    Returns:
        GeneratedCoverLetter object
    """
    generator = CoverLetterGenerator()
    return generator.generate_cover_letter(
        parsed_resume=parsed_resume,
        company_name=company_name,
        job_title=job_title,
        job_description=job_description,
        **kwargs
    )
