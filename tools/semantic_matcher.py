"""
Semantic Matching Analyzer
Compares resumes and job descriptions using OpenAI embeddings
Calculates similarity scores and provides gap analysis
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from openai import OpenAI

# Import other tools
from tools.keyword_extractor import KeywordExtractor, ExtractedKeywords
from parsers.resume_parser import ParsedResume

# Logging
from utils.logger import logger



class SemanticMatcher:
    """
    Analyzes semantic similarity between resumes and job descriptions
    Uses OpenAI embeddings for vector-based comparison
    """
    
    def __init__(self):
        """Initialize semantic matcher with OpenAI client"""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.keyword_extractor = KeywordExtractor(model="gpt-4o-mini")
        logger.info("SemanticMatcher initialized")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using OpenAI API
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (1536 dimensions for text-embedding-3-small)
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        if not vec1 or not vec2:
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
    
    def calculate_overall_similarity(self, resume_text: str, job_description: str) -> float:
        """
        Calculate overall semantic similarity between resume and job description
        
        Args:
            resume_text: Full resume text
            job_description: Full job description
            
        Returns:
            Similarity score (0-1)
        """
        logger.info("Calculating overall semantic similarity")
        
        resume_embedding = self.get_embedding(resume_text)
        job_embedding = self.get_embedding(job_description)
        
        similarity = self.cosine_similarity(resume_embedding, job_embedding)
        logger.info(f"Overall similarity: {similarity:.3f}")
        
        return round(similarity, 3)
    
    def calculate_section_similarities(
        self, 
        parsed_resume: ParsedResume, 
        job_description: str
    ) -> Dict[str, float]:
        """
        Calculate similarity for different resume sections vs job description
        
        Args:
            parsed_resume: ParsedResume object
            job_description: Job description text
            
        Returns:
            Dictionary with section-wise similarities
        """
        logger.info("Calculating section-wise similarities")
        
        job_embedding = self.get_embedding(job_description)
        similarities = {}
        
        # Summary similarity
        if parsed_resume.summary:
            summary_emb = self.get_embedding(parsed_resume.summary)
            similarities['summary'] = self.cosine_similarity(summary_emb, job_embedding)
        
        # Experience similarity (combined)
        if parsed_resume.experience:
            exp_text = "\n".join([
                f"{exp.title} at {exp.company}: " + " ".join(exp.description)
                for exp in parsed_resume.experience
            ])
            exp_emb = self.get_embedding(exp_text)
            similarities['experience'] = self.cosine_similarity(exp_emb, job_embedding)
        
        # Skills similarity
        if parsed_resume.skills:
            skills_text = ", ".join(parsed_resume.skills)
            skills_emb = self.get_embedding(skills_text)
            similarities['skills'] = self.cosine_similarity(skills_emb, job_embedding)
        
        # Education similarity
        if parsed_resume.education:
            edu_text = " ".join([
                f"{edu.degree} in {edu.field or ''} from {edu.institution}"
                for edu in parsed_resume.education
            ])
            edu_emb = self.get_embedding(edu_text)
            similarities['education'] = self.cosine_similarity(edu_emb, job_embedding)
        
        # Round all scores
        for key in similarities:
            similarities[key] = round(similarities[key], 3)
        
        logger.info(f"Section similarities calculated: {list(similarities.keys())}")
        return similarities
    
    def analyze_match(
        self, 
        resume_text: str, 
        parsed_resume: ParsedResume,
        job_description: str
    ) -> Dict[str, any]:
        """
        Comprehensive matching analysis between resume and job description
        
        Args:
            resume_text: Full resume text
            parsed_resume: Structured resume data
            job_description: Job description text
            
        Returns:
            Comprehensive match analysis with scores and recommendations
        """
        logger.info("Starting comprehensive match analysis")
        
        # Extract keywords from job
        job_keywords = self.keyword_extractor.extract_keywords(job_description)
        
        # Calculate similarities
        overall_similarity = self.calculate_overall_similarity(resume_text, job_description)
        section_similarities = self.calculate_section_similarities(parsed_resume, job_description)
        
        # Calculate keyword match score
        keyword_score = self.keyword_extractor.calculate_keyword_match_score(
            resume_text, job_keywords
        )
        
        # Get missing keywords
        missing_keywords = self.keyword_extractor.get_missing_keywords(
            resume_text, job_keywords
        )
        
        # Calculate composite score (weighted average)
        composite_score = (
            overall_similarity * 0.3 +  # 30% overall similarity
            (keyword_score / 100) * 0.5 +  # 50% keyword match
            section_similarities.get('skills', 0) * 0.2  # 20% skills match
        )
        
        # Determine match strength
        if composite_score >= 0.8:
            match_strength = "Excellent"
        elif composite_score >= 0.65:
            match_strength = "Good"
        elif composite_score >= 0.5:
            match_strength = "Fair"
        else:
            match_strength = "Weak"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            composite_score,
            missing_keywords,
            section_similarities,
            job_keywords
        )
        
        analysis = {
            'overall_similarity': overall_similarity,
            'keyword_match_score': keyword_score,
            'section_similarities': section_similarities,
            'composite_score': round(composite_score, 3),
            'match_strength': match_strength,
            'missing_keywords': missing_keywords,
            'extracted_job_keywords': job_keywords.model_dump(),
            'recommendations': recommendations
        }
        
        logger.info(f"Match analysis complete - Composite score: {composite_score:.3f} ({match_strength})")
        return analysis
    
    def _generate_recommendations(
        self,
        composite_score: float,
        missing_keywords: Dict[str, List[str]],
        section_similarities: Dict[str, float],
        job_keywords: ExtractedKeywords
    ) -> List[str]:
        """
        Generate tailoring recommendations based on analysis
        
        Args:
            composite_score: Overall match score
            missing_keywords: Missing keywords by category
            section_similarities: Section similarity scores
            job_keywords: Extracted job keywords
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Overall score recommendations
        if composite_score < 0.5:
            recommendations.append("⚠️ Low match score - consider if this role aligns with your background")
        elif composite_score < 0.7:
            recommendations.append("📝 Moderate match - resume needs significant tailoring")
        else:
            recommendations.append("✅ Good match - minor tailoring recommended")
        
        # Missing required skills
        if missing_keywords['required_skills']:
            count = len(missing_keywords['required_skills'])
            recommendations.append(
                f"🔴 Add {count} missing REQUIRED skills: {', '.join(missing_keywords['required_skills'][:5])}"
            )
        
        # Missing tools
        if missing_keywords['tools_technologies']:
            count = len(missing_keywords['tools_technologies'])
            recommendations.append(
                f"🛠️ Include {count} missing tools/technologies: {', '.join(missing_keywords['tools_technologies'][:5])}"
            )
        
        # Missing certifications
        if missing_keywords['certifications']:
            recommendations.append(
                f"📜 Highlight if you have: {', '.join(missing_keywords['certifications'])}"
            )
        
        # Section-specific recommendations
        if section_similarities.get('summary', 0) < 0.6:
            recommendations.append("📋 Rewrite professional summary to better match job requirements")
        
        if section_similarities.get('experience', 0) < 0.6:
            recommendations.append("💼 Emphasize relevant experience that matches job responsibilities")
        
        if section_similarities.get('skills', 0) < 0.7:
            recommendations.append("💪 Add more relevant skills from job description to your skills section")
        
        # ATS optimization
        if job_keywords.keywords_for_ats:
            recommendations.append(
                f"🎯 Ensure these ATS keywords appear: {', '.join(job_keywords.keywords_for_ats[:5])}"
            )
        
        # Experience level
        if job_keywords.experience_level:
            recommendations.append(
                f"⏱️ Highlight experience matching: {job_keywords.experience_level}"
            )
        
        return recommendations
    
    def calculate_ats_score(self, resume_text: str, job_keywords: ExtractedKeywords) -> int:
        """
        Calculate ATS (Applicant Tracking System) compatibility score
        
        Args:
            resume_text: Resume text
            job_keywords: Extracted job keywords
            
        Returns:
            ATS score (0-100)
        """
        score = 0
        max_score = 100
        
        resume_lower = resume_text.lower()
        
        # Check for ATS critical keywords (30 points)
        if job_keywords.keywords_for_ats:
            matches = sum(1 for kw in job_keywords.keywords_for_ats if kw.lower() in resume_lower)
            score += (matches / len(job_keywords.keywords_for_ats)) * 30
        
        # Check for required skills (40 points)
        if job_keywords.required_skills:
            matches = sum(1 for skill in job_keywords.required_skills if skill.lower() in resume_lower)
            score += (matches / len(job_keywords.required_skills)) * 40
        
        # Check for tools/technologies (20 points)
        if job_keywords.tools_technologies:
            matches = sum(1 for tool in job_keywords.tools_technologies if tool.lower() in resume_lower)
            score += (matches / len(job_keywords.tools_technologies)) * 20
        
        # Check for preferred skills (10 points)
        if job_keywords.preferred_skills:
            matches = sum(1 for skill in job_keywords.preferred_skills if skill.lower() in resume_lower)
            score += (matches / len(job_keywords.preferred_skills)) * 10
        
        return min(int(round(score)), max_score)


# Convenience function
def analyze_resume_job_match(resume_text: str, parsed_resume: ParsedResume, job_description: str) -> Dict:
    """
    Convenience function for comprehensive match analysis
    
    Args:
        resume_text: Resume text
        parsed_resume: Parsed resume structure
        job_description: Job description
        
    Returns:
        Match analysis dictionary
    """
    matcher = SemanticMatcher()
    return matcher.analyze_match(resume_text, parsed_resume, job_description)


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    sample_job = """
    Senior Python Developer - AI/ML Focus
    
    We're looking for a Senior Python Developer with 5+ years experience 
    to join our AI team. Must have experience with FastAPI, LangChain, 
    PostgreSQL, and cloud platforms (AWS/Azure). 
    
    Knowledge of Docker, Kubernetes, and CI/CD required.
    """
    
    sample_resume = """
    John Doe - Python Developer
    
    5 years of Python development experience. Built RESTful APIs using 
    FastAPI and Flask. Worked with PostgreSQL databases and deployed 
    applications on AWS. Familiar with Docker containerization.
    
    Skills: Python, FastAPI, PostgreSQL, AWS, Docker, Git
    """
    
    print("🔍 Semantic Matching Analysis\n")
    print("=" * 70)
    
    matcher = SemanticMatcher()
    
    # Overall similarity
    similarity = matcher.calculate_overall_similarity(sample_resume, sample_job)
    print(f"\n📊 Overall Semantic Similarity: {similarity:.1%}")
    
    # Keyword analysis
    extractor = KeywordExtractor()
    job_keywords = extractor.extract_keywords(sample_job)
    keyword_score = extractor.calculate_keyword_match_score(sample_resume, job_keywords)
    print(f"🎯 Keyword Match Score: {keyword_score}%")
    
    # ATS score
    ats_score = matcher.calculate_ats_score(sample_resume, job_keywords)
    print(f"🤖 ATS Compatibility Score: {ats_score}/100")
    
    # Missing keywords
    missing = extractor.get_missing_keywords(sample_resume, job_keywords)
    print(f"\n⚠️  Missing Keywords:")
    print(f"   Required Skills: {missing['required_skills']}")
    print(f"   Tools: {missing['tools_technologies']}")
    
    print("\n" + "=" * 70)
