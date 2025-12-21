"""
Keyword Extraction Tool
Extracts important keywords, skills, and requirements from job descriptions using GPT-4o-mini
"""

import os
import re
from typing import List, Dict, Set, Optional
from collections import Counter

# LLM for intelligent extraction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# OpenAI for embeddings
from openai import OpenAI

# Logging
from utils.logger import logger


# Pydantic models for structured output
class ExtractedKeywords(BaseModel):
    """Structured keywords extracted from job description"""
    required_skills: List[str] = Field(default_factory=list, description="Required technical skills and technologies")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred/nice-to-have skills")
    soft_skills: List[str] = Field(default_factory=list, description="Soft skills and interpersonal abilities")
    certifications: List[str] = Field(default_factory=list, description="Required or preferred certifications")
    experience_level: Optional[str] = Field(None, description="Required years of experience (e.g., '5+ years')")
    education: List[str] = Field(default_factory=list, description="Required education (degree, field)")
    responsibilities: List[str] = Field(default_factory=list, description="Key job responsibilities")
    tools_technologies: List[str] = Field(default_factory=list, description="Specific tools, frameworks, libraries")
    domains: List[str] = Field(default_factory=list, description="Industry domains or business areas")
    keywords_for_ats: List[str] = Field(default_factory=list, description="Critical keywords for ATS optimization")


class KeywordExtractor:
    """
    Extracts and analyzes keywords from job descriptions
    Uses GPT-4o-mini for intelligent extraction and OpenAI embeddings for semantic analysis
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize keyword extractor
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: Model temperature (0.0 for deterministic)
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info(f"KeywordExtractor initialized with model: {model}")
    
    def extract_keywords(self, job_description: str) -> ExtractedKeywords:
        """
        Extract structured keywords from job description using LLM
        
        Args:
            job_description: Job description text
            
        Returns:
            ExtractedKeywords object with categorized keywords
        """
        logger.info("Extracting keywords from job description")
        
        # Create JSON output parser
        parser = JsonOutputParser(pydantic_object=ExtractedKeywords)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing job descriptions and extracting key information.
            Extract all relevant keywords, skills, and requirements from the job description.
            
            Instructions:
            1. Identify REQUIRED technical skills (must-have)
            2. Identify PREFERRED skills (nice-to-have)
            3. Extract soft skills (communication, teamwork, etc.)
            4. List any certifications mentioned
            5. Identify experience level requirements
            6. Extract education requirements
            7. List key responsibilities
            8. Identify specific tools, frameworks, and technologies
            9. Identify industry domains or business areas
            10. Extract critical ATS keywords (most important terms for applicant tracking systems)
            
            Be thorough and specific. Use the exact terminology from the job description.
            
            {format_instructions}"""),
            ("user", "Job Description:\n\n{job_description}")
        ])
        
        # Create chain
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "job_description": job_description,
                "format_instructions": parser.get_format_instructions()
            })
            
            extracted = ExtractedKeywords(**result)
            
            logger.info(f"Extracted keywords - Required skills: {len(extracted.required_skills)}, "
                       f"Preferred: {len(extracted.preferred_skills)}, "
                       f"Tools: {len(extracted.tools_technologies)}, "
                       f"ATS keywords: {len(extracted.keywords_for_ats)}")
            
            return extracted
        
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return ExtractedKeywords()
    
    def extract_keywords_simple(self, text: str) -> Dict[str, List[str]]:
        """
        Simple regex-based keyword extraction (fallback method)
        
        Args:
            text: Job description text
            
        Returns:
            Dictionary with categorized keywords
        """
        keywords = {
            'technical_skills': [],
            'tools': [],
            'soft_skills': [],
            'years_experience': []
        }
        
        # Common technical skills patterns
        tech_patterns = [
            r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue|Node\.js|Django|Flask|FastAPI|Spring|Express)\b',
            r'\b(?:SQL|NoSQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|CI/CD)\b',
            r'\b(?:Machine Learning|Deep Learning|AI|NLP|Computer Vision)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords['technical_skills'].extend(matches)
        
        # Years of experience
        exp_pattern = r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience'
        exp_matches = re.findall(exp_pattern, text, re.IGNORECASE)
        keywords['years_experience'] = exp_matches
        
        # Remove duplicates
        for key in keywords:
            keywords[key] = list(set(keywords[key]))
        
        return keywords
    
    def get_keyword_embeddings(self, keywords: List[str]) -> List[List[float]]:
        """
        Get embeddings for keywords using OpenAI API
        
        Args:
            keywords: List of keywords
            
        Returns:
            List of embedding vectors
        """
        if not keywords:
            return []
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=keywords
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return []
    
    def rank_keywords_by_importance(self, keywords: List[str], job_description: str) -> List[Dict[str, any]]:
        """
        Rank keywords by importance using frequency and position in text
        
        Args:
            keywords: List of keywords to rank
            job_description: Original job description
            
        Returns:
            List of dicts with keyword and importance score
        """
        ranked = []
        
        for keyword in keywords:
            # Count occurrences
            count = len(re.findall(re.escape(keyword), job_description, re.IGNORECASE))
            
            # Check if in first 200 characters (likely in title/summary)
            in_beginning = keyword.lower() in job_description[:200].lower()
            
            # Calculate importance score
            score = count * 10
            if in_beginning:
                score += 20
            
            ranked.append({
                'keyword': keyword,
                'count': count,
                'score': score,
                'in_beginning': in_beginning
            })
        
        # Sort by score
        ranked.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked
    
    def analyze_keyword_density(self, text: str, keywords: List[str]) -> Dict[str, float]:
        """
        Calculate keyword density for ATS optimization
        
        Args:
            text: Resume or document text
            keywords: List of target keywords
            
        Returns:
            Dictionary with keyword density percentages
        """
        word_count = len(text.split())
        densities = {}
        
        for keyword in keywords:
            count = len(re.findall(re.escape(keyword), text, re.IGNORECASE))
            density = (count / word_count) * 100 if word_count > 0 else 0
            densities[keyword] = round(density, 2)
        
        return densities
    
    def get_missing_keywords(self, resume_text: str, job_keywords: ExtractedKeywords) -> Dict[str, List[str]]:
        """
        Identify keywords from job description missing in resume
        
        Args:
            resume_text: Resume text
            job_keywords: Extracted keywords from job description
            
        Returns:
            Dictionary of missing keywords by category
        """
        missing = {
            'required_skills': [],
            'preferred_skills': [],
            'tools_technologies': [],
            'certifications': [],
            'soft_skills': []
        }
        
        resume_lower = resume_text.lower()
        
        # Check required skills
        for skill in job_keywords.required_skills:
            if skill.lower() not in resume_lower:
                missing['required_skills'].append(skill)
        
        # Check preferred skills
        for skill in job_keywords.preferred_skills:
            if skill.lower() not in resume_lower:
                missing['preferred_skills'].append(skill)
        
        # Check tools/technologies
        for tool in job_keywords.tools_technologies:
            if tool.lower() not in resume_lower:
                missing['tools_technologies'].append(tool)
        
        # Check certifications
        for cert in job_keywords.certifications:
            if cert.lower() not in resume_lower:
                missing['certifications'].append(cert)
        
        # Check soft skills
        for skill in job_keywords.soft_skills:
            if skill.lower() not in resume_lower:
                missing['soft_skills'].append(skill)
        
        return missing
    
    def calculate_keyword_match_score(self, resume_text: str, job_keywords: ExtractedKeywords) -> float:
        """
        Calculate overall keyword match score between resume and job description
        
        Args:
            resume_text: Resume text
            job_keywords: Extracted keywords from job description
            
        Returns:
            Match score (0-100)
        """
        resume_lower = resume_text.lower()
        
        total_keywords = 0
        matched_keywords = 0
        
        # Weight different categories
        weights = {
            'required_skills': 3.0,  # Most important
            'tools_technologies': 2.0,
            'preferred_skills': 1.5,
            'soft_skills': 1.0,
            'certifications': 2.0
        }
        
        # Check required skills
        for skill in job_keywords.required_skills:
            total_keywords += weights['required_skills']
            if skill.lower() in resume_lower:
                matched_keywords += weights['required_skills']
        
        # Check tools/technologies
        for tool in job_keywords.tools_technologies:
            total_keywords += weights['tools_technologies']
            if tool.lower() in resume_lower:
                matched_keywords += weights['tools_technologies']
        
        # Check preferred skills
        for skill in job_keywords.preferred_skills:
            total_keywords += weights['preferred_skills']
            if skill.lower() in resume_lower:
                matched_keywords += weights['preferred_skills']
        
        # Check soft skills
        for skill in job_keywords.soft_skills:
            total_keywords += weights['soft_skills']
            if skill.lower() in resume_lower:
                matched_keywords += weights['soft_skills']
        
        # Check certifications
        for cert in job_keywords.certifications:
            total_keywords += weights['certifications']
            if cert.lower() in resume_lower:
                matched_keywords += weights['certifications']
        
        # Calculate percentage
        if total_keywords == 0:
            return 0.0
        
        score = (matched_keywords / total_keywords) * 100
        return round(min(score, 100), 2)


# Convenience function
def extract_job_keywords(job_description: str) -> ExtractedKeywords:
    """
    Convenience function to extract keywords from job description
    
    Args:
        job_description: Job description text
        
    Returns:
        ExtractedKeywords object
    """
    extractor = KeywordExtractor()
    return extractor.extract_keywords(job_description)


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    sample_job = """
    Senior Python Developer
    
    We are seeking an experienced Senior Python Developer to join our AI/ML team.
    
    Required Qualifications:
    - 5+ years of professional Python development experience
    - Strong experience with FastAPI, Django, or Flask
    - Proficiency in SQL and NoSQL databases (PostgreSQL, MongoDB)
    - Experience with cloud platforms (AWS, Azure, or GCP)
    - Knowledge of Docker and Kubernetes
    - Bachelor's degree in Computer Science or related field
    
    Preferred Qualifications:
    - Experience with LangChain and LLM applications
    - Knowledge of machine learning frameworks (TensorFlow, PyTorch)
    - AWS Certified Solutions Architect certification
    - Experience with CI/CD pipelines (Jenkins, GitLab CI)
    
    Responsibilities:
    - Design and implement scalable microservices architecture
    - Develop RESTful APIs for internal and external use
    - Collaborate with data scientists on ML model deployment
    - Mentor junior developers and conduct code reviews
    - Optimize application performance and database queries
    
    Skills:
    Strong communication skills, problem-solving abilities, team player, 
    attention to detail, ability to work in fast-paced environment.
    """
    
    print("🔍 Extracting Keywords from Job Description...\n")
    
    extractor = KeywordExtractor()
    keywords = extractor.extract_keywords(sample_job)
    
    print("✅ Keywords Extracted!\n")
    print("=" * 70)
    
    print("\n📌 REQUIRED SKILLS:")
    for skill in keywords.required_skills:
        print(f"   • {skill}")
    
    print("\n⭐ PREFERRED SKILLS:")
    for skill in keywords.preferred_skills:
        print(f"   • {skill}")
    
    print("\n🛠️  TOOLS & TECHNOLOGIES:")
    for tool in keywords.tools_technologies:
        print(f"   • {tool}")
    
    print("\n💬 SOFT SKILLS:")
    for skill in keywords.soft_skills:
        print(f"   • {skill}")
    
    print("\n🎓 EDUCATION:")
    for edu in keywords.education:
        print(f"   • {edu}")
    
    print("\n📜 CERTIFICATIONS:")
    for cert in keywords.certifications:
        print(f"   • {cert}")
    
    print(f"\n⏱️  EXPERIENCE LEVEL: {keywords.experience_level}")
    
    print("\n🎯 ATS KEYWORDS (Most Important):")
    for kw in keywords.keywords_for_ats[:10]:
        print(f"   • {kw}")
    
    print("\n" + "=" * 70)
    
    # Test keyword matching
    sample_resume = """
    John Doe - Python Developer
    5 years of experience in Python development
    Skills: Python, FastAPI, PostgreSQL, Docker, AWS, Git
    Built RESTful APIs and microservices
    """
    
    print("\n📊 Keyword Match Analysis:")
    score = extractor.calculate_keyword_match_score(sample_resume, keywords)
    print(f"   Match Score: {score}%")
    
    missing = extractor.get_missing_keywords(sample_resume, keywords)
    print("\n⚠️  Missing Keywords:")
    print(f"   Required Skills: {len(missing['required_skills'])}")
    print(f"   Tools: {len(missing['tools_technologies'])}")
    print(f"   Certifications: {len(missing['certifications'])}")
