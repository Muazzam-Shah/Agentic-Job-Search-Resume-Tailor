"""
Unit tests for semantic matching functionality
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from tools.semantic_matcher import SemanticMatcher
from tools.keyword_extractor import ExtractedKeywords
from parsers.resume_parser import ParsedResume, ContactInfo, Experience, Education


class TestSemanticMatcher:
    """Test suite for SemanticMatcher class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.matcher = SemanticMatcher()
        
        self.sample_resume_text = """
        Software Engineer with 5 years of experience in Python and web development.
        Built RESTful APIs using Django and Flask. Worked with PostgreSQL databases.
        Deployed applications on AWS using Docker containers.
        """
        
        self.sample_job = """
        Senior Python Developer position. Must have 5+ years experience with Python,
        Django/Flask frameworks, SQL databases, and cloud deployment experience.
        """
        
        self.parsed_resume = ParsedResume(
            contact_info=ContactInfo(
                name="John Doe",
                email="john@example.com",
                phone="555-1234"
            ),
            summary="Software Engineer with 5 years of Python experience",
            experience=[
                Experience(
                    title="Software Engineer",
                    company="Tech Corp",
                    start_date="2019",
                    end_date="Present",
                    description=["Built APIs with Django", "Deployed on AWS"]
                )
            ],
            education=[
                Education(
                    degree="Bachelor of Science",
                    field="Computer Science",
                    institution="State University"
                )
            ],
            skills=["Python", "Django", "Flask", "PostgreSQL", "AWS", "Docker"],
            raw_text=self.sample_resume_text
        )
    
    def test_initialization(self):
        """Test SemanticMatcher initialization"""
        assert hasattr(self.matcher, 'openai_client')
        assert hasattr(self.matcher, 'keyword_extractor')
        assert self.matcher.embedding_model == "text-embedding-3-small"
    
    @patch('tools.semantic_matcher.OpenAI')
    def test_get_embedding(self, mock_openai):
        """Test embedding generation"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_embedding = [0.1] * 1536
        mock_response.data = [Mock(embedding=mock_embedding)]
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        matcher = SemanticMatcher()
        embedding = matcher.get_embedding("test text")
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 1536
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])
        
        # Identical vectors
        sim_identical = self.matcher.cosine_similarity(vec1, vec2)
        assert abs(sim_identical - 1.0) < 0.001
        
        # Orthogonal vectors
        sim_orthogonal = self.matcher.cosine_similarity(vec1, vec3)
        assert abs(sim_orthogonal) < 0.001
    
    def test_cosine_similarity_partial(self):
        """Test cosine similarity with partial match"""
        vec1 = np.array([1.0, 1.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        sim = self.matcher.cosine_similarity(vec1, vec2)
        assert 0 < sim < 1  # Should be between 0 and 1
    
    @patch.object(SemanticMatcher, 'get_embedding')
    def test_calculate_overall_similarity(self, mock_embedding):
        """Test overall similarity calculation"""
        # Mock embeddings
        mock_embedding.side_effect = [
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0])
        ]
        
        similarity = self.matcher.calculate_overall_similarity(
            "resume text",
            "job description"
        )
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
    
    @patch.object(SemanticMatcher, 'get_embedding')
    def test_calculate_section_similarities(self, mock_embedding):
        """Test section-wise similarity calculation"""
        # Mock embeddings (return same for all for testing)
        mock_embedding.return_value = np.array([1.0] * 10)
        
        similarities = self.matcher.calculate_section_similarities(
            self.parsed_resume,
            self.sample_job
        )
        
        assert isinstance(similarities, dict)
        assert 'summary' in similarities
        assert 'experience' in similarities
        assert 'skills' in similarities
        assert 'education' in similarities
        
        for section, score in similarities.items():
            assert 0 <= score <= 1
    
    @patch.object(SemanticMatcher, 'get_embedding')
    def test_calculate_section_similarities_missing_sections(self, mock_embedding):
        """Test section similarities with missing sections"""
        mock_embedding.return_value = np.array([1.0] * 10)
        
        # Resume with no summary
        minimal_resume = ParsedResume(
            contact_info=ContactInfo(name="Jane Doe"),
            summary=None,
            experience=[],
            education=[],
            skills=["Python"],
            raw_text="Python developer"
        )
        
        similarities = self.matcher.calculate_section_similarities(
            minimal_resume,
            self.sample_job
        )
        
        # Should still return scores for all sections
        assert 'summary' in similarities
        assert similarities['summary'] == 0.0  # No summary present
    
    def test_calculate_ats_score(self):
        """Test ATS score calculation"""
        keywords = ExtractedKeywords(
            required_skills=["Python", "SQL"],
            preferred_skills=["Docker"],
            tools_technologies=["Git", "AWS"],
            certifications=[],
            soft_skills=["Communication"],
            experience_level="5+ years",
            education=["Bachelor's"],
            keywords_for_ats=["Python", "SQL", "Docker", "Git", "AWS"]
        )
        
        score = self.matcher.calculate_ats_score(self.sample_resume_text, keywords)
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100
        # Resume has Python, SQL, AWS, Docker
        assert score > 50
    
    def test_calculate_ats_score_no_match(self):
        """Test ATS score with no keyword matches"""
        keywords = ExtractedKeywords(
            required_skills=["Rust", "Haskell"],
            preferred_skills=["Erlang"],
            tools_technologies=["Nix"],
            certifications=[],
            soft_skills=[],
            experience_level="",
            education=[],
            keywords_for_ats=["Rust", "Haskell", "Erlang", "Nix"]
        )
        
        score = self.matcher.calculate_ats_score(self.sample_resume_text, keywords)
        
        # Should be very low score
        assert score < 20
    
    def test_match_strength_classification(self):
        """Test match strength classification"""
        # Test different score ranges
        assert self.matcher._get_match_strength(0.85) == "Excellent"
        assert self.matcher._get_match_strength(0.70) == "Good"
        assert self.matcher._get_match_strength(0.55) == "Fair"
        assert self.matcher._get_match_strength(0.35) == "Weak"
    
    def _get_match_strength(self, score):
        """Helper method for testing match strength"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.65:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        else:
            return "Weak"
    
    @patch.object(SemanticMatcher, 'get_embedding')
    @patch.object(SemanticMatcher, 'calculate_ats_score')
    def test_analyze_match_comprehensive(self, mock_ats, mock_embedding):
        """Test comprehensive match analysis"""
        # Mock embeddings and ATS score
        mock_embedding.return_value = np.array([1.0] * 10)
        mock_ats.return_value = 75
        
        analysis = self.matcher.analyze_match(
            self.sample_resume_text,
            self.parsed_resume,
            self.sample_job
        )
        
        # Check all required fields
        assert 'overall_similarity' in analysis
        assert 'keyword_match_score' in analysis
        assert 'section_similarities' in analysis
        assert 'composite_score' in analysis
        assert 'match_strength' in analysis
        assert 'missing_keywords' in analysis
        assert 'recommendations' in analysis
        
        # Check data types
        assert isinstance(analysis['overall_similarity'], float)
        assert isinstance(analysis['keyword_match_score'], (int, float))
        assert isinstance(analysis['section_similarities'], dict)
        assert isinstance(analysis['composite_score'], float)
        assert isinstance(analysis['match_strength'], str)
        assert isinstance(analysis['recommendations'], list)
    
    @patch.object(SemanticMatcher, 'get_embedding')
    def test_composite_score_calculation(self, mock_embedding):
        """Test composite score weighting"""
        # Mock high similarity
        mock_embedding.return_value = np.array([1.0] * 10)
        
        analysis = self.matcher.analyze_match(
            self.sample_resume_text,
            self.parsed_resume,
            self.sample_job
        )
        
        composite = analysis['composite_score']
        
        # Should be weighted average: 30% overall + 50% keywords + 20% skills
        assert isinstance(composite, float)
        assert 0 <= composite <= 1
    
    def test_recommendations_generation(self):
        """Test that recommendations are generated"""
        recommendations = self.matcher._generate_recommendations(
            overall_sim=0.7,
            keyword_score=60,
            missing_keywords={
                'required_skills': ["Kubernetes"],
                'preferred_skills': ["GraphQL"],
                'tools_technologies': ["Jenkins"],
                'certifications': ["AWS Certified"]
            },
            section_sims={
                'summary': 0.8,
                'experience': 0.6,
                'skills': 0.7,
                'education': 0.9
            }
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have recommendations for missing keywords
        rec_text = " ".join(recommendations)
        assert "Kubernetes" in rec_text or "missing" in rec_text.lower()
    
    def test_recommendations_with_no_missing_keywords(self):
        """Test recommendations when no keywords are missing"""
        recommendations = self.matcher._generate_recommendations(
            overall_sim=0.9,
            keyword_score=95,
            missing_keywords={
                'required_skills': [],
                'preferred_skills': [],
                'tools_technologies': [],
                'certifications': []
            },
            section_sims={
                'summary': 0.9,
                'experience': 0.9,
                'skills': 0.9,
                'education': 0.9
            }
        )
        
        # Should still have some recommendations
        assert isinstance(recommendations, list)
    
    @patch.object(SemanticMatcher, 'get_embedding')
    def test_analyze_match_edge_cases(self, mock_embedding):
        """Test match analysis with edge cases"""
        mock_embedding.return_value = np.array([0.0] * 10)
        
        # Empty resume
        minimal_resume = ParsedResume(
            contact_info=ContactInfo(name="Test"),
            skills=[],
            experience=[],
            education=[],
            raw_text=""
        )
        
        analysis = self.matcher.analyze_match(
            "",
            minimal_resume,
            self.sample_job
        )
        
        # Should handle gracefully
        assert analysis['composite_score'] >= 0
        assert isinstance(analysis['recommendations'], list)


class TestMatchAnalysis:
    """Test suite for comprehensive match analysis"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.matcher = SemanticMatcher()
    
    @patch.object(SemanticMatcher, 'get_embedding')
    @patch.object(SemanticMatcher, 'calculate_ats_score')
    def test_excellent_match(self, mock_ats, mock_embedding):
        """Test analysis with excellent match"""
        # Mock high similarity
        mock_embedding.return_value = np.array([1.0] * 10)
        mock_ats.return_value = 90
        
        resume = ParsedResume(
            contact_info=ContactInfo(name="Test"),
            skills=["Python", "Django", "AWS"],
            experience=[],
            education=[],
            raw_text="Python Django AWS expert"
        )
        
        analysis = self.matcher.analyze_match(
            "Python Django AWS expert",
            resume,
            "Looking for Python Django AWS developer"
        )
        
        assert analysis['match_strength'] in ["Excellent", "Good"]
        assert analysis['composite_score'] > 0.6
    
    @patch.object(SemanticMatcher, 'get_embedding')
    @patch.object(SemanticMatcher, 'calculate_ats_score')
    def test_weak_match(self, mock_ats, mock_embedding):
        """Test analysis with weak match"""
        # Mock low similarity
        mock_embedding.return_value = np.array([0.1] * 10)
        mock_ats.return_value = 20
        
        resume = ParsedResume(
            contact_info=ContactInfo(name="Test"),
            skills=["Java"],
            experience=[],
            education=[],
            raw_text="Java developer"
        )
        
        analysis = self.matcher.analyze_match(
            "Java developer",
            resume,
            "Looking for Python expert with machine learning"
        )
        
        # Should identify weak match
        assert len(analysis['recommendations']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
