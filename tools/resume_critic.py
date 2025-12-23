"""
Resume Critic - Intelligent Resume Analysis and Feedback

This module provides comprehensive resume critique capabilities including
content analysis, formatting feedback, keyword optimization, and ATS scoring.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.resume_parser import ParsedResume


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Feedback Categories
# ============================================================================

class ContentFeedback(BaseModel):
    """Feedback on resume content quality."""
    category: str = Field(default="content", description="Feedback category")
    issues: List[str] = Field(default_factory=list, description="Content issues found")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    score: int = Field(ge=0, le=100, description="Content quality score (0-100)")


class FormatFeedback(BaseModel):
    """Feedback on resume formatting."""
    category: str = Field(default="format", description="Feedback category")
    issues: List[str] = Field(default_factory=list, description="Formatting issues")
    suggestions: List[str] = Field(default_factory=list, description="Format improvements")
    score: int = Field(ge=0, le=100, description="Format quality score (0-100)")


class KeywordFeedback(BaseModel):
    """Feedback on keyword optimization."""
    category: str = Field(default="keywords", description="Feedback category")
    missing_keywords: List[str] = Field(default_factory=list, description="Important missing keywords")
    keyword_density: float = Field(ge=0, le=1, description="Keyword density (0-1)")
    suggestions: List[str] = Field(default_factory=list, description="Keyword optimization tips")
    score: int = Field(ge=0, le=100, description="Keyword optimization score (0-100)")


class ATSFeedback(BaseModel):
    """Feedback on ATS compatibility."""
    category: str = Field(default="ats", description="Feedback category")
    compatibility_score: int = Field(ge=0, le=100, description="ATS compatibility (0-100)")
    issues: List[str] = Field(default_factory=list, description="ATS compatibility issues")
    suggestions: List[str] = Field(default_factory=list, description="ATS optimization tips")


class ComprehensiveFeedback(BaseModel):
    """Complete resume feedback across all categories."""
    content: ContentFeedback
    format: FormatFeedback
    keywords: KeywordFeedback
    ats: ATSFeedback
    overall_score: int = Field(ge=0, le=100, description="Overall resume quality (0-100)")
    top_priorities: List[str] = Field(description="Top 3-5 improvement priorities")
    strengths: List[str] = Field(description="Resume strengths to maintain")


# ============================================================================
# Resume Critic Class
# ============================================================================

class ResumeCritic:
    """
    Intelligent resume critic providing multi-category feedback.
    
    Analyzes resumes across content, format, keywords, and ATS compatibility,
    providing specific, actionable improvement suggestions.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Initialize the resume critic.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for feedback generation (0.3 for balanced)
        """
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        logger.info(f"ResumeCritic initialized with model: {model}")
    
    
    def critique_resume(
        self,
        parsed_resume: ParsedResume,
        job_description: Optional[str] = None,
        focus_areas: Optional[List[str]] = None
    ) -> ComprehensiveFeedback:
        """
        Provide comprehensive resume critique.
        
        Args:
            parsed_resume: Parsed resume data
            job_description: Optional job description for targeted feedback
            focus_areas: Optional list of specific areas to focus on
        
        Returns:
            ComprehensiveFeedback with detailed analysis
        """
        logger.info("Generating comprehensive resume critique")
        
        # Create parser for structured output
        parser = JsonOutputParser(pydantic_object=ComprehensiveFeedback)
        
        # Prepare resume text summary
        resume_summary = self._create_resume_summary(parsed_resume)
        
        # Create critique prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume reviewer and career coach with 15+ years of experience. 
            Provide specific, actionable feedback that helps job seekers improve their resumes.
            Be constructive but honest. Focus on concrete improvements.
            
            {format_instructions}"""),
            ("user", """Analyze this resume and provide comprehensive feedback:
            
            RESUME:
            {resume_summary}
            
            {job_context}
            {focus_context}
            
            Provide detailed feedback in these categories:
            
            1. CONTENT (score 0-100):
               - Are achievements quantified with metrics?
               - Are bullet points action-oriented and impactful?
               - Is the summary compelling and tailored?
               - Are there any gaps or weak sections?
               - Specific issues and suggestions
            
            2. FORMAT (score 0-100):
               - Is the layout clean and professional?
               - Is information well-organized?
               - Are sections clearly defined?
               - Is there good use of white space?
               - Specific formatting improvements
            
            3. KEYWORDS (score 0-100):
               - {keyword_context}
               - Are industry-standard keywords present?
               - Is keyword density appropriate?
               - Missing important keywords
               - Keyword optimization suggestions
            
            4. ATS COMPATIBILITY (score 0-100):
               - Would this pass ATS screening?
               - Are there any ATS-unfriendly elements?
               - Suggestions for ATS optimization
            
            5. OVERALL ASSESSMENT:
               - Calculate overall_score (weighted average)
               - Identify top 3-5 priorities for improvement
               - List 3-5 strengths to maintain
            
            Be specific with examples. Quote actual text when suggesting improvements.""")
        ])
        
        # Prepare context
        job_context = ""
        if job_description:
            job_context = f"TARGET JOB:\n{job_description[:500]}\n\nProvide feedback relevant to this role."
        
        keyword_context = "Does the resume include relevant keywords for the candidate's field?"
        if job_description:
            keyword_context = "Does the resume include keywords from the job description?"
        
        focus_context = ""
        if focus_areas:
            focus_context = f"FOCUS ON: {', '.join(focus_areas)}"
        
        # Generate critique
        formatted_prompt = prompt.format_messages(
            format_instructions=parser.get_format_instructions(),
            resume_summary=resume_summary,
            job_context=job_context,
            keyword_context=keyword_context,
            focus_context=focus_context
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            
            # Parse JSON response
            content = response.content.strip()
            if content.startswith('```'):
                lines = content.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines)
            
            feedback_dict = json.loads(content)
            feedback = ComprehensiveFeedback(**feedback_dict)
            
            logger.info(f"Generated comprehensive feedback (Overall score: {feedback.overall_score})")
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating critique: {e}")
            # Return basic feedback on error
            return ComprehensiveFeedback(
                content=ContentFeedback(
                    issues=["Unable to analyze content"],
                    suggestions=["Review resume content for clarity and impact"],
                    score=70
                ),
                format=FormatFeedback(
                    issues=["Unable to analyze format"],
                    suggestions=["Ensure clean, professional formatting"],
                    score=70
                ),
                keywords=KeywordFeedback(
                    missing_keywords=[],
                    keyword_density=0.5,
                    suggestions=["Add relevant industry keywords"],
                    score=70
                ),
                ats=ATSFeedback(
                    compatibility_score=70,
                    issues=["Unable to fully assess ATS compatibility"],
                    suggestions=["Use standard section headings and simple formatting"]
                ),
                overall_score=70,
                top_priorities=["Review resume comprehensively"],
                strengths=["Resume structure present"]
            )
    
    
    def suggest_improvements(
        self,
        parsed_resume: ParsedResume,
        feedback_category: str,
        specific_section: Optional[str] = None
    ) -> List[str]:
        """
        Generate specific improvement suggestions for a category.
        
        Args:
            parsed_resume: Parsed resume data
            feedback_category: Category to focus on (content/format/keywords/ats)
            specific_section: Optional specific section to improve
        
        Returns:
            List of specific improvement suggestions
        """
        logger.info(f"Generating {feedback_category} improvement suggestions")
        
        # Extract relevant section
        section_text = ""
        if specific_section:
            if specific_section.lower() == "summary" and parsed_resume.summary:
                section_text = parsed_resume.summary
            elif specific_section.lower() == "experience":
                section_text = "\n".join([
                    f"{exp.title} at {exp.company}: {', '.join(exp.description[:3])}"
                    for exp in parsed_resume.experience[:3]
                ])
        else:
            section_text = self._create_resume_summary(parsed_resume)[:1000]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert resume coach providing specific, actionable improvement suggestions."),
            ("user", """Provide 5-7 specific improvements for this resume section:
            
            CATEGORY: {category}
            SECTION: {section}
            
            CURRENT TEXT:
            {text}
            
            Provide concrete, actionable suggestions. Be specific with examples.
            Return as JSON array: ["suggestion 1", "suggestion 2", ...]""")
        ])
        
        try:
            formatted = prompt.format_messages(
                category=feedback_category,
                section=specific_section or "overall",
                text=section_text
            )
            
            response = self.llm.invoke(formatted)
            content = response.content.strip()
            
            # Remove markdown if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            
            suggestions = json.loads(content)
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [f"Review {specific_section or 'resume'} for {feedback_category} improvements"]
    
    
    def compare_versions(
        self,
        original_resume: ParsedResume,
        improved_resume: ParsedResume
    ) -> Dict[str, Any]:
        """
        Compare two resume versions and highlight improvements.
        
        Args:
            original_resume: Original resume
            improved_resume: Improved version
        
        Returns:
            Dictionary with comparison metrics and highlights
        """
        logger.info("Comparing resume versions")
        
        # Get scores for both
        original_feedback = self.critique_resume(original_resume)
        improved_feedback = self.critique_resume(improved_resume)
        
        # Calculate improvements
        comparison = {
            "overall_improvement": improved_feedback.overall_score - original_feedback.overall_score,
            "content_improvement": improved_feedback.content.score - original_feedback.content.score,
            "format_improvement": improved_feedback.format.score - original_feedback.format.score,
            "keyword_improvement": improved_feedback.keywords.score - original_feedback.keywords.score,
            "ats_improvement": improved_feedback.ats.compatibility_score - original_feedback.ats.compatibility_score,
            "original_score": original_feedback.overall_score,
            "improved_score": improved_feedback.overall_score,
            "improvements_made": [],
            "remaining_issues": improved_feedback.top_priorities
        }
        
        # Identify specific improvements
        if comparison["content_improvement"] > 5:
            comparison["improvements_made"].append(f"Content quality improved by {comparison['content_improvement']} points")
        if comparison["keyword_improvement"] > 5:
            comparison["improvements_made"].append(f"Keyword optimization improved by {comparison['keyword_improvement']} points")
        if comparison["ats_improvement"] > 5:
            comparison["improvements_made"].append(f"ATS compatibility improved by {comparison['ats_improvement']} points")
        
        return comparison
    
    
    def _create_resume_summary(self, parsed_resume: ParsedResume) -> str:
        """Create a text summary of resume for analysis."""
        summary_parts = []
        
        # Contact info
        summary_parts.append(f"Name: {parsed_resume.contact_info.name}")
        if parsed_resume.contact_info.email:
            summary_parts.append(f"Email: {parsed_resume.contact_info.email}")
        
        # Summary
        if parsed_resume.summary:
            summary_parts.append(f"\nPROFESSIONAL SUMMARY:\n{parsed_resume.summary}")
        
        # Experience
        if parsed_resume.experience:
            summary_parts.append("\nWORK EXPERIENCE:")
            for exp in parsed_resume.experience[:3]:  # Top 3
                summary_parts.append(f"\n{exp.title} at {exp.company}")
                if exp.start_date or exp.end_date:
                    dates = f"{exp.start_date or ''} - {exp.end_date or ''}"
                    summary_parts.append(f"  {dates}")
                for bullet in exp.description[:3]:  # Top 3 bullets
                    summary_parts.append(f"  - {bullet}")
        
        # Education
        if parsed_resume.education:
            summary_parts.append("\nEDUCATION:")
            for edu in parsed_resume.education:
                summary_parts.append(f"{edu.degree} - {edu.institution}")
        
        # Skills
        if parsed_resume.skills:
            summary_parts.append(f"\nSKILLS: {', '.join(parsed_resume.skills[:20])}")
        
        return "\n".join(summary_parts)


# ============================================================================
# Convenience Functions
# ============================================================================

def critique_resume(
    parsed_resume: ParsedResume,
    job_description: Optional[str] = None
) -> ComprehensiveFeedback:
    """
    Quick function to critique a resume.
    
    Args:
        parsed_resume: Parsed resume data
        job_description: Optional job description for context
    
    Returns:
        ComprehensiveFeedback object
    """
    critic = ResumeCritic()
    return critic.critique_resume(parsed_resume, job_description)
