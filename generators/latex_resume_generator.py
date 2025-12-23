"""
LaTeX Resume Generator

Generates professional, ATS-optimized resumes using LaTeX templates.
Provides superior typography and formatting compared to DOCX approach.

Features:
- Multiple professional templates (modern, classic, academic)
- Direct PDF generation with pdflatex
- Jinja2 template engine for dynamic content
- Content selection based on job matching
- GPT-4o-mini bullet point optimization
- ATS-friendly formatting
- Automated LaTeX error handling

Author: Job Hunter Project
Date: December 2025
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from parsers.resume_parser import ParsedResume, Experience, Education
from tools.keyword_extractor import KeywordExtractor, ExtractedKeywords
from tools.semantic_matcher import SemanticMatcher
from utils.logger import logger


class LaTeXResumeGenerator:
    """
    Generates tailored, ATS-optimized resumes using LaTeX
    
    Features:
    - Professional LaTeX templates with superior typography
    - Content selection based on job match
    - Bullet point optimization with GPT-4o-mini
    - Direct PDF generation (no intermediate files)
    - Multiple template styles
    - ATS-friendly formatting
    """
    
    TEMPLATE_STYLES = {
        'modern': 'modern_template.tex',
        'classic': 'classic_template.tex',
        'academic': 'academic_template.tex'
    }
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        output_dir: str = "output/resumes",
        template_dir: str = "templates/latex",
        template_style: str = "classic"
    ):
        """
        Initialize LaTeX resume generator
        
        Args:
            model: LLM model for bullet point optimization
            temperature: Temperature for creative rewriting (0.7 for balance)
            output_dir: Directory for generated PDFs
            template_dir: Directory containing LaTeX templates
            template_style: Template style (modern, classic, academic)
        """
        self.model = model
        self.temperature = temperature
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.template_dir = Path(template_dir)
        if not self.template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")
        
        if template_style not in self.TEMPLATE_STYLES:
            raise ValueError(f"Invalid template style. Choose from: {list(self.TEMPLATE_STYLES.keys())}")
        
        self.template_style = template_style
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['tex']),
            block_start_string='\\BLOCK{',
            block_end_string='}',
            variable_start_string='\\VAR{',
            variable_end_string='}',
            comment_start_string='\\#{',
            comment_end_string='}',
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filter for LaTeX escaping
        self.jinja_env.filters['escape_latex'] = self._escape_latex
        
        # Initialize LLM for bullet point optimization
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        # Initialize helper tools
        self.keyword_extractor = KeywordExtractor(model=model)
        self.semantic_matcher = SemanticMatcher()
        
        # Check for LaTeX installation
        self._check_latex_installation()
        
        logger.info(f"LaTeXResumeGenerator initialized with {template_style} template")
    
    def _check_latex_installation(self) -> bool:
        """
        Check if pdflatex is installed and available
        
        Returns:
            True if pdflatex is available, False otherwise
        """
        try:
            result = subprocess.run(
                ['pdflatex', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("pdflatex is available")
                return True
            else:
                logger.warning("pdflatex not found. Install TeX Live or MiKTeX to generate PDFs.")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"pdflatex check failed: {e}")
            logger.warning("Install TeX Live (Linux/Mac) or MiKTeX (Windows) to enable PDF generation.")
            return False
    
    def _escape_latex(self, text: str) -> str:
        """
        Escape special LaTeX characters
        
        Args:
            text: Text to escape
        
        Returns:
            LaTeX-safe text
        """
        if not text:
            return ""
        
        # Special character replacements
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
        }
        
        result = str(text)
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)
        
        return result
    
    def generate_tailored_resume(
        self,
        parsed_resume: ParsedResume,
        job_description: str,
        company_name: str,
        job_title: str,
        match_analysis: Optional[Dict] = None,
        template_style: Optional[str] = None
    ) -> str:
        """
        Generate a tailored resume for specific job
        
        Args:
            parsed_resume: Parsed master resume data
            job_description: Target job description
            company_name: Company name for file naming
            job_title: Job title for file naming
            match_analysis: Pre-computed match analysis (optional)
            template_style: Override default template style (optional)
        
        Returns:
            Path to generated PDF file
        """
        logger.info(f"Generating LaTeX resume for {company_name} - {job_title}")
        
        # Extract job keywords
        job_keywords = self.keyword_extractor.extract_keywords(job_description)
        
        # Get match analysis if not provided
        if match_analysis is None:
            match_analysis = self.semantic_matcher.analyze_match(
                parsed_resume,
                job_description,
                job_keywords
            )
        
        # Select and optimize content
        selected_content = self._select_content(
            parsed_resume,
            job_keywords,
            match_analysis
        )
        
        # Create tailored summary
        tailored_summary = self._create_tailored_summary(
            parsed_resume,
            job_keywords,
            job_description
        )
        
        # Optimize experiences
        optimized_experiences = self._optimize_experiences(
            selected_content['experiences'],
            job_keywords,
            job_description
        )
        
        # Generate PDF
        output_path = self._generate_pdf(
            parsed_resume=parsed_resume,
            tailored_summary=tailored_summary,
            optimized_experiences=optimized_experiences,
            selected_skills=selected_content['skills'],
            company_name=company_name,
            job_title=job_title,
            template_style=template_style or self.template_style
        )
        
        logger.info(f"LaTeX resume generated: {output_path}")
        return output_path
    
    def _select_content(
        self,
        parsed_resume: ParsedResume,
        job_keywords: ExtractedKeywords,
        match_analysis: Dict
    ) -> Dict:
        """
        Select most relevant content based on job requirements
        
        Args:
            parsed_resume: Parsed resume data
            job_keywords: Extracted job keywords
            match_analysis: Match analysis results
        
        Returns:
            Dictionary with selected content
        """
        logger.info("Selecting relevant content for job")
        
        # Score each experience by relevance
        experience_scores = []
        for exp in parsed_resume.experience:
            exp_text = f"{exp.title} {exp.company} {' '.join(exp.description or [])}"
            
            # Calculate keyword overlap
            all_keywords = (
                job_keywords.required_skills +
                job_keywords.preferred_skills +
                job_keywords.tools_technologies
            )
            
            keyword_matches = sum(
                1 for kw in all_keywords
                if kw.lower() in exp_text.lower()
            )
            
            relevance_score = keyword_matches / max(len(all_keywords), 1)
            experience_scores.append((exp, relevance_score))
        
        # Sort by relevance
        experience_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top experiences (at least top 3, all if score > 0.2)
        selected_experiences = []
        for exp, score in experience_scores:
            if len(selected_experiences) < 3 or score > 0.2:
                selected_experiences.append((exp, exp.description or []))
        
        # Prioritize skills based on job keywords
        selected_skills = self._prioritize_skills(
            parsed_resume.skills,
            job_keywords
        )
        
        return {
            'experiences': selected_experiences,
            'skills': selected_skills,
            'experience_scores': experience_scores
        }
    
    def _prioritize_skills(
        self,
        resume_skills: List[str],
        job_keywords: ExtractedKeywords
    ) -> List[str]:
        """
        Prioritize skills based on job requirements
        
        Args:
            resume_skills: List of skills from resume
            job_keywords: Job keywords
        
        Returns:
            Prioritized skill list
        """
        if not resume_skills:
            return []
        
        # Create priority tiers
        required = set(kw.lower() for kw in job_keywords.required_skills)
        preferred = set(kw.lower() for kw in job_keywords.preferred_skills)
        tools = set(kw.lower() for kw in job_keywords.tools_technologies)
        
        # Categorize resume skills
        priority_1 = []  # Required skills
        priority_2 = []  # Tools and preferred
        priority_3 = []  # Other skills
        
        for skill in resume_skills:
            skill_lower = skill.lower()
            if skill_lower in required:
                priority_1.append(skill)
            elif skill_lower in preferred or skill_lower in tools:
                priority_2.append(skill)
            else:
                priority_3.append(skill)
        
        # Combine with priorities first
        return priority_1 + priority_2 + priority_3[:10]
    
    def _create_tailored_summary(
        self,
        parsed_resume: ParsedResume,
        job_keywords: ExtractedKeywords,
        job_description: str
    ) -> str:
        """
        Create job-specific professional summary using GPT-4o-mini
        
        Args:
            parsed_resume: Parsed resume
            job_keywords: Job keywords
            job_description: Full job description
        
        Returns:
            Tailored summary text
        """
        logger.info("Creating tailored professional summary")
        
        # Build prompt for summary generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume writer. Create a compelling professional summary 
            tailored to the specific job. The summary should:
            1. Highlight relevant skills and experience matching the job
            2. Incorporate key job requirements naturally
            3. Be concise (2-3 sentences, ~50 words)
            4. Use action-oriented language
            5. Emphasize quantifiable achievements when possible
            
            Return JSON: {"summary": "tailored summary text"}"""),
            ("user", """Original Summary: {original_summary}
            
Job Title/Description: {job_description}

Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Tools/Technologies: {tools}

Create a tailored professional summary that positions the candidate perfectly for this role.""")
        ])
        
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "original_summary": parsed_resume.summary or "Professional with diverse experience",
                "job_description": job_description[:500],
                "required_skills": ", ".join(job_keywords.required_skills[:10]),
                "preferred_skills": ", ".join(job_keywords.preferred_skills[:5]),
                "tools": ", ".join(job_keywords.tools_technologies[:5])
            })
            
            return result.get('summary', parsed_resume.summary or "")
            
        except Exception as e:
            logger.error(f"Failed to generate tailored summary: {e}")
            return parsed_resume.summary or ""
    
    def _optimize_experiences(
        self,
        experiences: List[Tuple[Experience, List[str]]],
        job_keywords: ExtractedKeywords,
        job_description: str
    ) -> List[Tuple[Experience, List[str]]]:
        """
        Optimize experience bullet points with job keywords
        
        Args:
            experiences: List of (Experience, bullet_points) tuples
            job_keywords: Job keywords
            job_description: Job description
        
        Returns:
            List of (Experience, optimized_bullets) tuples
        """
        logger.info(f"Optimizing {len(experiences)} experience sections")
        
        optimized = []
        for exp, bullets in experiences:
            if not bullets:
                optimized.append((exp, []))
                continue
            
            # Optimize bullets
            optimized_bullets = self._optimize_bullet_points(
                bullets,
                job_keywords,
                exp.title
            )
            
            optimized.append((exp, optimized_bullets))
        
        return optimized
    
    def _optimize_bullet_points(
        self,
        bullets: List[str],
        job_keywords: ExtractedKeywords,
        role_title: str
    ) -> List[str]:
        """
        Optimize bullet points using GPT-4o-mini
        
        Args:
            bullets: Original bullet points
            job_keywords: Job keywords to incorporate
            role_title: Role title for context
        
        Returns:
            Optimized bullet points
        """
        if not bullets:
            return []
        
        # Build optimization prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume writer. Optimize bullet points to:
            1. Incorporate relevant job keywords naturally
            2. Use strong action verbs
            3. Quantify achievements when possible
            4. Keep concise (1-2 lines each)
            5. Maintain truthfulness (enhance, don't fabricate)
            6. Use past tense for completed roles
            
            Return JSON: {"optimized_bullets": ["bullet 1", "bullet 2", ...]}"""),
            ("user", """Role: {role_title}

Original Bullets:
{bullets}

Job Keywords to Incorporate:
Required Skills: {required_skills}
Tools/Technologies: {tools}
Preferred Skills: {preferred_skills}

Optimize these bullet points to better match the job requirements while staying truthful.""")
        ])
        
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "role_title": role_title,
                "bullets": "\n".join(f"- {b}" for b in bullets),
                "required_skills": ", ".join(job_keywords.required_skills[:8]),
                "tools": ", ".join(job_keywords.tools_technologies[:6]),
                "preferred_skills": ", ".join(job_keywords.preferred_skills[:4])
            })
            
            optimized = result.get('optimized_bullets', bullets)
            logger.info(f"Optimized {len(bullets)} bullets for {role_title}")
            return optimized
            
        except Exception as e:
            logger.error(f"Failed to optimize bullets: {e}")
            return bullets
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Categorize skills by type for organized display
        
        Args:
            skills: List of skills
        
        Returns:
            Dictionary mapping categories to skills
        """
        categories = {
            'Programming Languages': [],
            'Frameworks / Tools': [],
            'Machine Learning & AI': [],
            'Database Management': [],
            'DevOps': []
        }
        
        # Keywords for categorization
        prog_langs = ['python', 'c++', 'java', 'javascript', 'c#', 'kotlin', 'c ', 'swift', 'go', 'rust']
        frameworks = ['react', 'angular', 'node', 'flask', 'django', 'spring', 'tailwind', 'qt', 'vue']
        ml_ai = ['tensorflow', 'pytorch', 'keras', 'scikit', 'llm', 'gpt', 'whisper', 'langchain', 'numpy', 'pandas']
        databases = ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'firebase', 'dynamodb']
        devops = ['docker', 'kubernetes', 'git', 'aws', 'azure', 'gcp', 'jenkins', 'terraform']
        
        # Categorize each skill
        for skill in skills[:25]:
            skill_lower = skill.lower()
            
            if any(lang in skill_lower for lang in prog_langs):
                categories['Programming Languages'].append(skill)
            elif any(fw in skill_lower for fw in frameworks):
                categories['Frameworks / Tools'].append(skill)
            elif any(ml in skill_lower for ml in ml_ai):
                categories['Machine Learning & AI'].append(skill)
            elif any(db in skill_lower for db in databases):
                categories['Database Management'].append(skill)
            elif any(dv in skill_lower for dv in devops):
                categories['DevOps'].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _generate_pdf(
        self,
        parsed_resume: ParsedResume,
        tailored_summary: str,
        optimized_experiences: List[Tuple[Experience, List[str]]],
        selected_skills: List[str],
        company_name: str,
        job_title: str,
        template_style: str
    ) -> str:
        """
        Generate PDF from LaTeX template
        
        Args:
            parsed_resume: Original parsed resume
            tailored_summary: Tailored summary
            optimized_experiences: Optimized experiences
            selected_skills: Prioritized skills
            company_name: Company name
            job_title: Job title
            template_style: Template style to use
        
        Returns:
            Path to generated PDF file
        """
        logger.info(f"Generating PDF with {template_style} template")
        
        # Prepare template data
        skills_by_category = self._categorize_skills(selected_skills)
        
        template_data = {
            'contact_info': parsed_resume.contact_info,
            'summary': tailored_summary,
            'education': parsed_resume.education,
            'experiences': optimized_experiences,
            'projects': parsed_resume.projects[:4] if parsed_resume.projects else [],
            'skills': selected_skills,
            'skills_by_category': skills_by_category,
            'certifications': parsed_resume.certifications,
            'awards': parsed_resume.awards,
            'publications': getattr(parsed_resume, 'publications', []),
            'presentations': getattr(parsed_resume, 'presentations', []),
            'service': getattr(parsed_resume, 'service', [])
        }
        
        # Load and render template
        template_file = self.TEMPLATE_STYLES[template_style]
        template = self.jinja_env.get_template(template_file)
        latex_content = template.render(**template_data)
        
        # Generate filename
        filename = self._generate_filename(company_name, job_title)
        
        # Compile LaTeX to PDF
        pdf_path = self._compile_latex(latex_content, filename)
        
        return pdf_path
    
    def _compile_latex(self, latex_content: str, filename: str) -> str:
        """
        Compile LaTeX content to PDF using pdflatex
        
        Args:
            latex_content: LaTeX source code
            filename: Base filename (without extension)
        
        Returns:
            Path to generated PDF
        """
        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Write LaTeX content to file
            tex_file = tmpdir_path / f"{filename}.tex"
            tex_file.write_text(latex_content, encoding='utf-8')
            
            # Compile with pdflatex (run twice for references)
            for i in range(2):
                try:
                    result = subprocess.run(
                        ['pdflatex', '-interaction=nonstopmode', f'{filename}.tex'],
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode != 0:
                        logger.error(f"pdflatex compilation failed (run {i+1}):")
                        logger.error(result.stdout)
                        logger.error(result.stderr)
                        if i == 1:  # Only raise on second run
                            raise RuntimeError(f"LaTeX compilation failed: {result.stderr}")
                
                except subprocess.TimeoutExpired:
                    logger.error("pdflatex compilation timed out")
                    raise RuntimeError("LaTeX compilation timed out after 30 seconds")
                
                except FileNotFoundError:
                    logger.error("pdflatex not found. Please install TeX Live or MiKTeX.")
                    raise RuntimeError("pdflatex not installed. Install TeX Live (Linux/Mac) or MiKTeX (Windows).")
            
            # Copy PDF to output directory
            pdf_source = tmpdir_path / f"{filename}.pdf"
            if not pdf_source.exists():
                logger.error("PDF file was not generated")
                # Save .log file for debugging
                log_file = tmpdir_path / f"{filename}.log"
                if log_file.exists():
                    debug_log = self.output_dir / f"{filename}_debug.log"
                    shutil.copy(log_file, debug_log)
                    logger.error(f"LaTeX log saved to: {debug_log}")
                raise RuntimeError("PDF generation failed. Check LaTeX log for errors.")
            
            pdf_destination = self.output_dir / f"{filename}.pdf"
            shutil.copy(pdf_source, pdf_destination)
            
            logger.info(f"PDF compiled successfully: {pdf_destination}")
            return str(pdf_destination)
    
    def _generate_filename(self, company_name: str, job_title: str) -> str:
        """
        Generate filename: Resume_[Company]_[Role]
        
        Args:
            company_name: Company name
            job_title: Job title
        
        Returns:
            Sanitized filename (without extension)
        """
        # Sanitize strings
        company_clean = re.sub(r'[^\w\s-]', '', company_name)
        company_clean = re.sub(r'[-\s]+', '_', company_clean)
        
        job_clean = re.sub(r'[^\w\s-]', '', job_title)
        job_clean = re.sub(r'[-\s]+', '_', job_clean)
        
        # Build filename
        filename = f"Resume_{company_clean}_{job_clean}"
        
        # Ensure not too long (max 100 chars)
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename
    
    def generate_multiple_versions(
        self,
        parsed_resume: ParsedResume,
        jobs: List[Dict[str, str]],
        batch_size: int = 5
    ) -> List[str]:
        """
        Generate multiple tailored resumes for different jobs
        
        Args:
            parsed_resume: Parsed master resume
            jobs: List of job dicts with 'description', 'company', 'title'
            batch_size: Number of resumes to generate
        
        Returns:
            List of generated file paths
        """
        logger.info(f"Generating {len(jobs)} LaTeX resumes")
        
        generated_files = []
        for job in jobs[:batch_size]:
            try:
                pdf_path = self.generate_tailored_resume(
                    parsed_resume=parsed_resume,
                    job_description=job['description'],
                    company_name=job['company'],
                    job_title=job['title']
                )
                generated_files.append(pdf_path)
            except Exception as e:
                logger.error(f"Failed to generate resume for {job['company']}: {e}")
        
        logger.info(f"Successfully generated {len(generated_files)} LaTeX resumes")
        return generated_files


# Convenience function
def generate_latex_resume(
    resume_file: str,
    job_description: str,
    company_name: str,
    job_title: str,
    template_style: str = "classic",
    output_dir: str = "output/resumes"
) -> str:
    """
    Convenience function to generate a LaTeX resume
    
    Args:
        resume_file: Path to master resume (PDF/DOCX)
        job_description: Job description text
        company_name: Company name
        job_title: Job title
        template_style: Template style (modern/classic/academic)
        output_dir: Output directory
    
    Returns:
        Path to generated PDF
    """
    from parsers.resume_parser import parse_resume
    
    # Parse resume
    parsed_resume = parse_resume(resume_file)
    
    # Generate tailored resume
    generator = LaTeXResumeGenerator(
        output_dir=output_dir,
        template_style=template_style
    )
    
    return generator.generate_tailored_resume(
        parsed_resume=parsed_resume,
        job_description=job_description,
        company_name=company_name,
        job_title=job_title
    )
