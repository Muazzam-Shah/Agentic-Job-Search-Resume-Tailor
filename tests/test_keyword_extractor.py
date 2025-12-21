"""
Unit tests for keyword extraction functionality
"""

import pytest
from unittest.mock import Mock, patch
from tools.keyword_extractor import KeywordExtractor, ExtractedKeywords


class TestKeywordExtractor:
    """Test suite for KeywordExtractor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = KeywordExtractor(model="gpt-4o-mini")
        
        self.sample_keywords = ExtractedKeywords(
            required_skills=["Python", "JavaScript", "SQL"],
            preferred_skills=["React", "Docker"],
            tools_technologies=["Git", "Jenkins", "AWS"],
            certifications=["AWS Certified Solutions Architect"],
            soft_skills=["Communication", "Problem Solving"],
            experience_level="5+ years",
            education=["Bachelor's in Computer Science"],
            keywords_for_ats=["Python", "JavaScript", "SQL", "React", "AWS", "Git"]
        )
        
        self.sample_job = """
        Senior Python Developer
        Required: Python, SQL, 5+ years experience
        Preferred: React, Docker
        Tools: Git, Jenkins, AWS
        """
        
        self.sample_resume = """
        Software Engineer with 6 years of experience.
        Skills: Python, SQL, Git, AWS, JavaScript
        Built applications using React and Docker.
        """
    
    def test_initialization(self):
        """Test KeywordExtractor initialization"""
        assert self.extractor.model == "gpt-4o-mini"
        assert self.extractor.temperature == 0.0
        assert hasattr(self.extractor, 'llm')
        assert hasattr(self.extractor, 'openai_client')
    
    def test_initialization_custom_temperature(self):
        """Test initialization with custom temperature"""
        extractor = KeywordExtractor(temperature=0.3)
        assert extractor.temperature == 0.3
    
    @patch('tools.keyword_extractor.ChatOpenAI')
    def test_extract_keywords_simple(self, mock_llm):
        """Test simple keyword extraction (regex-based)"""
        keywords = self.extractor.extract_keywords_simple(self.sample_job)
        
        assert isinstance(keywords, ExtractedKeywords)
        assert "Python" in keywords.required_skills
        assert "SQL" in keywords.required_skills
        assert len(keywords.keywords_for_ats) > 0
    
    def test_keyword_match_calculation(self):
        """Test keyword match score calculation"""
        score = self.extractor.calculate_keyword_match_score(
            self.sample_resume,
            self.sample_keywords
        )
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100
        # Should have good match (Python, SQL, Git, AWS present)
        assert score > 50
    
    def test_keyword_match_empty_resume(self):
        """Test keyword matching with empty resume"""
        score = self.extractor.calculate_keyword_match_score(
            "",
            self.sample_keywords
        )
        
        assert score == 0
    
    def test_keyword_match_perfect_match(self):
        """Test keyword matching with all keywords present"""
        perfect_resume = """
        Python JavaScript SQL React Docker Git Jenkins AWS
        Communication Problem Solving 5+ years
        Bachelor's in Computer Science
        AWS Certified Solutions Architect
        """
        
        score = self.extractor.calculate_keyword_match_score(
            perfect_resume,
            self.sample_keywords
        )
        
        # Should be very high score
        assert score > 85
    
    def test_missing_keywords_identification(self):
        """Test identification of missing keywords"""
        missing = self.extractor.get_missing_keywords(
            self.sample_resume,
            self.sample_keywords
        )
        
        assert isinstance(missing, dict)
        assert 'required_skills' in missing
        assert 'preferred_skills' in missing
        assert 'tools_technologies' in missing
        assert 'certifications' in missing
        
        # Resume doesn't have Jenkins
        assert "Jenkins" in missing['tools_technologies']
        
        # Resume doesn't have certification
        assert len(missing['certifications']) > 0
    
    def test_missing_keywords_complete_resume(self):
        """Test missing keywords with complete resume"""
        complete_resume = """
        Python JavaScript SQL React Docker Git Jenkins AWS
        Communication Problem Solving
        AWS Certified Solutions Architect
        """
        
        missing = self.extractor.get_missing_keywords(
            complete_resume,
            self.sample_keywords
        )
        
        # Should have minimal missing keywords
        assert len(missing['required_skills']) == 0
        assert len(missing['tools_technologies']) == 0
    
    def test_keyword_density_analysis(self):
        """Test keyword density calculation"""
        keywords = ["Python", "SQL", "JavaScript"]
        densities = self.extractor.analyze_keyword_density(
            self.sample_resume,
            keywords
        )
        
        assert isinstance(densities, dict)
        assert "Python" in densities
        assert "SQL" in densities
        
        for kw, density in densities.items():
            assert isinstance(density, float)
            assert density >= 0
    
    def test_keyword_density_case_insensitive(self):
        """Test that keyword density is case-insensitive"""
        resume = "Python python PYTHON sql SQL"
        keywords = ["Python", "SQL"]
        
        densities = self.extractor.analyze_keyword_density(resume, keywords)
        
        # Should count all case variations
        assert densities["Python"] > 0
        assert densities["SQL"] > 0
    
    def test_keyword_ranking(self):
        """Test keyword ranking by importance"""
        job_desc = """
        Python Developer Position
        Must have: Python Python Python
        Nice to have: Docker Docker
        Tools: Git
        """
        
        keywords = ["Python", "Docker", "Git"]
        ranked = self.extractor.rank_keywords_by_importance(job_desc, keywords)
        
        assert isinstance(ranked, list)
        assert len(ranked) > 0
        
        # Python should rank higher (mentioned more)
        python_rank = next(
            (i for i, item in enumerate(ranked) if item['keyword'] == 'Python'),
            -1
        )
        docker_rank = next(
            (i for i, item in enumerate(ranked) if item['keyword'] == 'Docker'),
            -1
        )
        
        assert python_rank < docker_rank  # Python appears first
    
    def test_keyword_ranking_empty_list(self):
        """Test keyword ranking with empty keyword list"""
        ranked = self.extractor.rank_keywords_by_importance(
            self.sample_job,
            []
        )
        
        assert ranked == []
    
    @patch('tools.keyword_extractor.OpenAI')
    def test_get_keyword_embeddings(self, mock_openai):
        """Test getting embeddings for keywords"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        extractor = KeywordExtractor()
        keywords = ["Python", "JavaScript"]
        embeddings = extractor.get_keyword_embeddings(keywords)
        
        assert isinstance(embeddings, dict)
        assert "Python" in embeddings
        assert "JavaScript" in embeddings
    
    def test_extracted_keywords_model(self):
        """Test ExtractedKeywords pydantic model"""
        keywords = ExtractedKeywords(
            required_skills=["Python"],
            preferred_skills=[],
            tools_technologies=["Git"],
            certifications=[],
            soft_skills=["Communication"],
            experience_level="3+ years",
            education=["Bachelor's degree"],
            keywords_for_ats=["Python", "Git"]
        )
        
        assert keywords.required_skills == ["Python"]
        assert keywords.tools_technologies == ["Git"]
        assert keywords.experience_level == "3+ years"
        assert len(keywords.keywords_for_ats) == 2
    
    def test_extracted_keywords_optional_fields(self):
        """Test ExtractedKeywords with minimal fields"""
        keywords = ExtractedKeywords(
            required_skills=["Python"],
            preferred_skills=[],
            tools_technologies=[],
            certifications=[],
            soft_skills=[],
            experience_level="",
            education=[],
            keywords_for_ats=["Python"]
        )
        
        assert len(keywords.preferred_skills) == 0
        assert len(keywords.certifications) == 0
        assert keywords.experience_level == ""


class TestKeywordMatching:
    """Test suite for keyword matching functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = KeywordExtractor(model="gpt-4o-mini")
    
    def test_weighted_scoring(self):
        """Test that required skills are weighted higher"""
        keywords = ExtractedKeywords(
            required_skills=["Python"],
            preferred_skills=["Docker"],
            tools_technologies=[],
            certifications=[],
            soft_skills=[],
            experience_level="",
            education=[],
            keywords_for_ats=[]
        )
        
        # Resume with only required skill
        resume_required = "Python developer"
        score_required = self.extractor.calculate_keyword_match_score(
            resume_required, keywords
        )
        
        # Resume with only preferred skill
        resume_preferred = "Docker expert"
        score_preferred = self.extractor.calculate_keyword_match_score(
            resume_preferred, keywords
        )
        
        # Required skill match should score higher
        assert score_required > score_preferred
    
    def test_partial_match_scoring(self):
        """Test scoring with partial keyword matches"""
        keywords = ExtractedKeywords(
            required_skills=["Python", "JavaScript", "SQL", "React"],
            preferred_skills=[],
            tools_technologies=[],
            certifications=[],
            soft_skills=[],
            experience_level="",
            education=[],
            keywords_for_ats=[]
        )
        
        # Resume with 50% of required skills
        resume_half = "Python SQL developer"
        score_half = self.extractor.calculate_keyword_match_score(
            resume_half, keywords
        )
        
        # Resume with all required skills
        resume_full = "Python JavaScript SQL React developer"
        score_full = self.extractor.calculate_keyword_match_score(
            resume_full, keywords
        )
        
        assert score_half < score_full
        assert 20 < score_half < 80  # Should be in middle range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
