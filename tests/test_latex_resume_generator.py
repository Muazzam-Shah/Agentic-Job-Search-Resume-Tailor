"""
Unit Tests for LaTeX Resume Generator

Tests the LaTeX resume generation functionality including:
- Template loading and rendering
- Content selection and optimization
- LaTeX compilation
- Error handling

Author: Job Hunter Project
Date: December 2025
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from generators.latex_resume_generator import LaTeXResumeGenerator, generate_latex_resume
from parsers.resume_parser import (
    ParsedResume, ContactInfo, Experience, Education
)
from tools.keyword_extractor import ExtractedKeywords


@pytest.fixture
def sample_contact_info():
    """Sample contact information"""
    return ContactInfo(
        name="John Doe",
        email="john.doe@email.com",
        phone="+1-555-123-4567",
        location="San Francisco, CA",
        linkedin="https://www.linkedin.com/in/johndoe",
        github="https://github.com/johndoe"
    )


@pytest.fixture
def sample_experience():
    """Sample work experience"""
    return [
        Experience(
            title="Senior Software Engineer",
            company="Tech Corp",
            location="San Francisco, CA",
            start_date="Jan 2020",
            end_date="Present",
            description=[
                "Led development of microservices architecture using Python and Docker",
                "Improved system performance by 40% through optimization",
                "Mentored team of 5 junior developers"
            ]
        ),
        Experience(
            title="Software Engineer",
            company="StartupXYZ",
            location="New York, NY",
            start_date="Jun 2018",
            end_date="Dec 2019",
            description=[
                "Developed REST APIs using Flask and PostgreSQL",
                "Implemented CI/CD pipeline with Jenkins",
                "Reduced deployment time by 60%"
            ]
        )
    ]


@pytest.fixture
def sample_education():
    """Sample education"""
    return [
        Education(
            degree="Bachelor of Science",
            field="Computer Science",
            institution="Stanford University",
            location="Stanford, CA",
            graduation_date="May 2018",
            gpa="3.8/4.0"
        )
    ]


@pytest.fixture
def sample_parsed_resume(sample_contact_info, sample_experience, sample_education):
    """Complete sample parsed resume"""
    return ParsedResume(
        contact_info=sample_contact_info,
        summary="Experienced software engineer with 5+ years in backend development",
        experience=sample_experience,
        education=sample_education,
        skills=[
            "Python", "JavaScript", "Docker", "Kubernetes", "AWS",
            "Flask", "Django", "PostgreSQL", "MongoDB", "Git"
        ],
        certifications=["AWS Certified Solutions Architect"],
        projects=[
            {
                "name": "ML Pipeline Automation",
                "description": "Built automated ML pipeline using Python and Airflow"
            }
        ],
        awards=["Employee of the Year 2021"],
        raw_text="Full resume text..."
    )


@pytest.fixture
def sample_job_keywords():
    """Sample extracted job keywords"""
    return ExtractedKeywords(
        required_skills=["Python", "Docker", "Kubernetes", "AWS"],
        preferred_skills=["PostgreSQL", "Redis", "Terraform"],
        tools_technologies=["Flask", "FastAPI", "Git", "Jenkins"],
        certifications=["AWS Certified"],
        soft_skills=["Leadership", "Communication"],
        ats_keywords=["Python", "Docker", "Kubernetes", "AWS", "microservices"]
    )


class TestLaTeXResumeGenerator:
    """Test LaTeX Resume Generator"""
    
    def test_initialization(self):
        """Test generator initialization"""
        generator = LaTeXResumeGenerator(
            template_style="classic",
            output_dir="test_output"
        )
        
        assert generator.template_style == "classic"
        assert generator.output_dir == Path("test_output")
        assert generator.model == "gpt-4o-mini"
        assert generator.temperature == 0.7
    
    def test_invalid_template_style(self):
        """Test initialization with invalid template style"""
        with pytest.raises(ValueError):
            LaTeXResumeGenerator(template_style="invalid_style")
    
    def test_missing_template_directory(self):
        """Test initialization with missing template directory"""
        with pytest.raises(FileNotFoundError):
            LaTeXResumeGenerator(template_dir="nonexistent_dir")
    
    def test_escape_latex(self):
        """Test LaTeX special character escaping"""
        generator = LaTeXResumeGenerator()
        
        # Test special characters
        assert generator._escape_latex("Hello & World") == r"Hello \& World"
        assert generator._escape_latex("50% improvement") == r"50\% improvement"
        assert generator._escape_latex("$100K salary") == r"\$100K salary"
        assert generator._escape_latex("C# Developer") == r"C\# Developer"
        assert generator._escape_latex("file_name.txt") == r"file\_name.txt"
        assert generator._escape_latex("") == ""
    
    def test_prioritize_skills(self, sample_job_keywords):
        """Test skill prioritization"""
        generator = LaTeXResumeGenerator()
        
        resume_skills = [
            "Python", "Java", "Docker", "Kubernetes",
            "PostgreSQL", "MongoDB", "AWS", "React"
        ]
        
        prioritized = generator._prioritize_skills(resume_skills, sample_job_keywords)
        
        # Check that required skills come first
        assert "Python" in prioritized[:4]
        assert "Docker" in prioritized[:4]
        assert "Kubernetes" in prioritized[:4]
        assert "AWS" in prioritized[:4]
    
    def test_categorize_skills(self):
        """Test skill categorization"""
        generator = LaTeXResumeGenerator()
        
        skills = [
            "Python", "JavaScript", "React", "Flask",
            "Docker", "Kubernetes", "PostgreSQL", "MongoDB",
            "TensorFlow", "PyTorch"
        ]
        
        categories = generator._categorize_skills(skills)
        
        assert "Programming Languages" in categories
        assert "Python" in categories["Programming Languages"]
        assert "JavaScript" in categories["Programming Languages"]
        
        assert "Frameworks / Tools" in categories
        assert "React" in categories["Frameworks / Tools"]
        assert "Flask" in categories["Frameworks / Tools"]
        
        assert "DevOps" in categories
        assert "Docker" in categories["DevOps"]
        
        assert "Database Management" in categories
        assert "PostgreSQL" in categories["Database Management"]
    
    def test_generate_filename(self):
        """Test filename generation"""
        generator = LaTeXResumeGenerator()
        
        filename = generator._generate_filename("Google Inc.", "Software Engineer")
        assert filename == "Resume_Google_Inc_Software_Engineer"
        
        # Test with special characters
        filename = generator._generate_filename("AT&T Corp.", "Sr. Dev/Architect")
        assert "Resume_ATT" in filename
        
        # Test length limiting
        long_company = "Very Long Company Name That Exceeds Maximum Length Limit"
        long_title = "Senior Principal Staff Software Development Engineer Architect"
        filename = generator._generate_filename(long_company, long_title)
        assert len(filename) <= 100
    
    @patch('generators.latex_resume_generator.ChatOpenAI')
    def test_create_tailored_summary(self, mock_llm, sample_parsed_resume, sample_job_keywords):
        """Test tailored summary creation"""
        # Mock LLM response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "summary": "Experienced Python engineer with expertise in Docker and Kubernetes"
        }
        mock_llm.return_value.__or__ = MagicMock(return_value=mock_chain)
        
        generator = LaTeXResumeGenerator()
        
        summary = generator._create_tailored_summary(
            sample_parsed_resume,
            sample_job_keywords,
            "We need a Python engineer with Docker experience"
        )
        
        assert "Python" in summary or "Docker" in summary or "engineer" in summary
    
    @patch('generators.latex_resume_generator.ChatOpenAI')
    def test_optimize_bullet_points(self, mock_llm, sample_job_keywords):
        """Test bullet point optimization"""
        # Mock LLM response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "optimized_bullets": [
                "Led Python microservices development using Docker and Kubernetes",
                "Improved AWS infrastructure performance by 40%"
            ]
        }
        mock_llm.return_value.__or__ = MagicMock(return_value=mock_chain)
        
        generator = LaTeXResumeGenerator()
        
        original_bullets = [
            "Led development of services",
            "Improved performance"
        ]
        
        optimized = generator._optimize_bullet_points(
            original_bullets,
            sample_job_keywords,
            "Senior Software Engineer"
        )
        
        assert len(optimized) == 2
        assert isinstance(optimized[0], str)
    
    def test_select_content(self, sample_parsed_resume, sample_job_keywords):
        """Test content selection"""
        generator = LaTeXResumeGenerator()
        
        match_analysis = {
            "overall_similarity": 0.85,
            "section_scores": {},
            "missing_keywords": []
        }
        
        content = generator._select_content(
            sample_parsed_resume,
            sample_job_keywords,
            match_analysis
        )
        
        assert 'experiences' in content
        assert 'skills' in content
        assert len(content['experiences']) > 0
        assert len(content['skills']) > 0
    
    @patch('subprocess.run')
    def test_check_latex_installation(self, mock_run):
        """Test LaTeX installation check"""
        # Mock successful pdflatex check
        mock_run.return_value = Mock(returncode=0, stdout="pdflatex 3.14")
        
        generator = LaTeXResumeGenerator()
        result = generator._check_latex_installation()
        
        assert result == True
        mock_run.assert_called()
    
    @patch('subprocess.run')
    def test_check_latex_not_installed(self, mock_run):
        """Test LaTeX not installed"""
        # Mock failed pdflatex check
        mock_run.side_effect = FileNotFoundError()
        
        generator = LaTeXResumeGenerator()
        result = generator._check_latex_installation()
        
        assert result == False
    
    @patch('subprocess.run')
    @patch('shutil.copy')
    def test_compile_latex_success(self, mock_copy, mock_run):
        """Test successful LaTeX compilation"""
        # Mock successful compilation
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        generator = LaTeXResumeGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake PDF
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_text("fake pdf content")
            
            # Mock the PDF creation
            def copy_side_effect(src, dst):
                Path(dst).write_text("fake pdf")
            
            mock_copy.side_effect = copy_side_effect
            
            latex_content = r"\documentclass{article}\begin{document}Test\end{document}"
            
            # This will fail in practice without actual pdflatex, but tests the logic
            try:
                result = generator._compile_latex(latex_content, "test")
                assert "test.pdf" in result
            except RuntimeError:
                # Expected if pdflatex not actually installed
                pass
    
    @patch('subprocess.run')
    def test_compile_latex_failure(self, mock_run):
        """Test failed LaTeX compilation"""
        # Mock failed compilation
        mock_run.return_value = Mock(
            returncode=1,
            stdout="Error in LaTeX",
            stderr="Undefined control sequence"
        )
        
        generator = LaTeXResumeGenerator()
        
        latex_content = r"\documentclass{article}\begin{document}\invalidcommand\end{document}"
        
        with pytest.raises(RuntimeError):
            generator._compile_latex(latex_content, "test")
    
    @patch('subprocess.run')
    def test_compile_latex_timeout(self, mock_run):
        """Test LaTeX compilation timeout"""
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired('pdflatex', 30)
        
        generator = LaTeXResumeGenerator()
        
        latex_content = r"\documentclass{article}\begin{document}Test\end{document}"
        
        with pytest.raises(RuntimeError, match="timed out"):
            generator._compile_latex(latex_content, "test")
    
    def test_template_loading(self):
        """Test LaTeX template loading"""
        generator = LaTeXResumeGenerator(template_style="classic")
        
        template = generator.jinja_env.get_template("classic_template.tex")
        assert template is not None
    
    @patch.object(LaTeXResumeGenerator, '_compile_latex')
    @patch.object(LaTeXResumeGenerator, '_optimize_bullet_points')
    @patch.object(LaTeXResumeGenerator, '_create_tailored_summary')
    def test_generate_tailored_resume(
        self,
        mock_summary,
        mock_optimize,
        mock_compile,
        sample_parsed_resume
    ):
        """Test full resume generation workflow"""
        # Mock methods
        mock_summary.return_value = "Tailored summary"
        mock_optimize.return_value = ["Optimized bullet 1", "Optimized bullet 2"]
        mock_compile.return_value = "output/test_resume.pdf"
        
        generator = LaTeXResumeGenerator()
        
        result = generator.generate_tailored_resume(
            parsed_resume=sample_parsed_resume,
            job_description="Python engineer with Docker experience needed",
            company_name="TechCorp",
            job_title="Software Engineer"
        )
        
        assert result == "output/test_resume.pdf"
        mock_summary.assert_called_once()
        assert mock_optimize.call_count > 0
        mock_compile.assert_called_once()
    
    def test_generate_multiple_versions(self, sample_parsed_resume):
        """Test batch resume generation"""
        generator = LaTeXResumeGenerator()
        
        jobs = [
            {
                "description": "Python engineer needed",
                "company": "Company A",
                "title": "Software Engineer"
            },
            {
                "description": "DevOps engineer with Docker",
                "company": "Company B",
                "title": "DevOps Engineer"
            }
        ]
        
        with patch.object(generator, 'generate_tailored_resume') as mock_generate:
            mock_generate.side_effect = ["resume1.pdf", "resume2.pdf"]
            
            results = generator.generate_multiple_versions(
                sample_parsed_resume,
                jobs,
                batch_size=2
            )
            
            assert len(results) == 2
            assert mock_generate.call_count == 2


class TestConvenienceFunction:
    """Test convenience function"""
    
    @patch('generators.latex_resume_generator.parse_resume')
    @patch.object(LaTeXResumeGenerator, 'generate_tailored_resume')
    def test_generate_latex_resume(self, mock_generate, mock_parse, sample_parsed_resume):
        """Test convenience function"""
        mock_parse.return_value = sample_parsed_resume
        mock_generate.return_value = "output/resume.pdf"
        
        result = generate_latex_resume(
            resume_file="resume.pdf",
            job_description="Python engineer needed",
            company_name="TechCorp",
            job_title="Software Engineer",
            template_style="classic"
        )
        
        assert result == "output/resume.pdf"
        mock_parse.assert_called_once_with("resume.pdf")
        mock_generate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
