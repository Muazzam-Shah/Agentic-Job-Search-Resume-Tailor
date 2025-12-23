"""
Simple PDF Resume Generator - FPDF2 + JSON Approach
No HTML/CSS messiness. Pure Python. Pixel-perfect formatting.
"""

import os
import json
from typing import Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from fpdf import FPDF

# Load environment variables
load_dotenv()


class PDFResume(FPDF):
    """
    Custom PDF class using FPDF2 for FAANG-style formatting.
    Direct layout control - no HTML/CSS conversion.
    """
    
    def header(self):
        # Header is handled manually in generate_body
        pass

    def footer(self):
        """Page numbers at bottom (optional - FAANG resumes often skip this)"""
        pass  # No footer for clean 1-page resume

    def section_title(self, label: str):
        """Creates FAANG-style section header with underline."""
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0)
        self.cell(0, 6, label.upper(), align="L")
        self.ln(6)
        # Draw line under section
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)


class SimplePDFGenerator:
    """
    FAANG-style PDF resume generator.
    Uses GPT-4 to create structured JSON → renders with fpdf2.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize with OpenAI model."""
        self.llm = ChatOpenAI(model=model, temperature=0.0)
        self.output_dir = "output/resumes/pdf"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_resume_pdf(
        self,
        parsed_resume: Dict,
        job_description: str,
        company_name: str,
        job_title: str,
        style: str = "professional"
    ) -> str:
        """
        Generate PDF resume in 2 steps:
        1. GPT-4 creates structured JSON
        2. Render PDF with fpdf2
        
        Args:
            parsed_resume: Master resume data (dict or ParsedResume)
            job_description: Target job description
            company_name: Company name
            job_title: Job title
            style: Ignored (always FAANG style)
        
        Returns:
            Path to generated PDF file
        """
        
        # Convert ParsedResume object to dict if needed
        if hasattr(parsed_resume, 'dict'):
            resume_data = parsed_resume.dict()
        else:
            resume_data = parsed_resume
        
        # Step 1: GPT-4 generates structured JSON
        print("🤖 Generating resume content with GPT-4...")
        json_data = self._generate_json(
            resume_data, job_description, company_name, job_title
        )
        
        # Step 2: Render PDF directly with fpdf2
        print("📄 Rendering PDF...")
        pdf_path = self._render_pdf(json_data, company_name, job_title)
        
        print(f"✅ Resume generated: {pdf_path}")
        return pdf_path
    
    def _generate_json(
        self,
        resume_data: Dict,
        job_description: str,
        company_name: str,
        job_title: str
    ) -> Dict:
        """Use GPT-4 to generate structured JSON resume data."""
        
        system_prompt = """You are an expert FAANG resume writer. Output ONLY valid JSON.

OUTPUT SCHEMA (strict):
{
    "header": {
        "name": "Full Name",
        "email": "email@example.com",
        "phone": "(555) 123-4567",
        "location": "City, State",
        "links": ["linkedin.com/in/username", "github.com/username"]
    },
    "summary": "2-3 sentence professional summary highlighting key achievements and perfect fit for this role",
    "skills": {
        "Languages": ["Python", "JavaScript", "Java"],
        "Frameworks": ["React", "Django", "FastAPI"],
        "Cloud/Tools": ["AWS", "Docker", "Kubernetes"],
        "Databases": ["PostgreSQL", "MongoDB", "Redis"]
    },
    "experience": [
        {
            "title": "Senior Software Engineer",
            "company": "Company Name",
            "location": "City, State",
            "date": "Jan 2020 - Present",
            "bullets": [
                "Led X project using Y technology, resulting in Z% improvement",
                "Developed A which reduced B by C% and saved $D annually",
                "Architected E system serving F users with G% uptime"
            ]
        }
    ],
    "education": [
        {
            "school": "University Name",
            "degree": "Bachelor of Science in Computer Science",
            "location": "City, State",
            "year": "2020",
            "gpa": "3.8/4.0",
            "relevant": "Relevant Coursework: Algorithms, Distributed Systems"
        }
    ],
    "projects": [
        {
            "name": "Project Name",
            "description": "Built X using Y technologies, achieving Z impact",
            "tech": ["Python", "AWS", "Docker"]
        }
    ]
}

CRITICAL RULES:
1. Output ONLY JSON (no markdown, no explanations)
2. Quantify EVERYTHING (X% growth, $Y saved, Z users)
3. Use action verbs: Led, Developed, Architected, Reduced, Increased
4. Each bullet = ONE strong achievement (not responsibilities)
5. Maximum 3-4 bullets per role
6. Skills: prioritize those matching job description
7. Summary: show PERFECT fit for target role
8. Projects: only if impressive/relevant"""

        user_prompt = f"""Create a tailored resume for this job:

**TARGET POSITION:**
Company: {company_name}
Role: {job_title}
Requirements: {job_description}

**CANDIDATE DATA:**
{self._format_resume_data(resume_data)}

**INSTRUCTIONS:**
- Tailor everything to match job requirements
- Highlight only relevant experiences
- Use technologies from job description
- Quantify all achievements
- Keep professional and concise"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        content = response.content.strip()
        
        # Clean up markdown code blocks if GPT adds them
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"GPT Response:\n{content[:500]}...")
            raise
    
    def _format_resume_data(self, resume_data: Dict) -> str:
        """Format resume data for GPT-4 prompt."""
        formatted = []
        
        # Contact info
        if 'contact_info' in resume_data and resume_data['contact_info']:
            contact = resume_data['contact_info']
            formatted.append(f"Name: {contact.get('name', 'N/A')}")
            formatted.append(f"Email: {contact.get('email', 'N/A')}")
            formatted.append(f"Phone: {contact.get('phone', 'N/A')}")
            if contact.get('linkedin'):
                formatted.append(f"LinkedIn: {contact['linkedin']}")
            if contact.get('github'):
                formatted.append(f"GitHub: {contact['github']}")
        
        # Summary
        if 'summary' in resume_data and resume_data['summary']:
            formatted.append(f"\nSummary: {resume_data['summary']}")
        
        # Experience
        if 'experience' in resume_data and resume_data['experience']:
            formatted.append("\nExperience:")
            for exp in resume_data['experience']:
                formatted.append(f"  - {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")
                formatted.append(f"    {exp.get('start_date', 'N/A')} - {exp.get('end_date', 'Present')}")
                if exp.get('achievements'):
                    for achievement in exp['achievements'][:4]:
                        formatted.append(f"    • {achievement}")
        
        # Education
        if 'education' in resume_data and resume_data['education']:
            formatted.append("\nEducation:")
            for edu in resume_data['education']:
                formatted.append(f"  - {edu.get('degree', 'N/A')} in {edu.get('field', 'N/A')}")
                formatted.append(f"    {edu.get('institution', 'N/A')} ({edu.get('graduation_date', 'N/A')})")
                if edu.get('gpa'):
                    formatted.append(f"    GPA: {edu['gpa']}")
        
        # Skills
        if 'skills' in resume_data and resume_data['skills']:
            skills = resume_data['skills']
            formatted.append("\nSkills:")
            if isinstance(skills, dict):
                for category, skill_list in skills.items():
                    formatted.append(f"  {category}: {', '.join(skill_list)}")
            elif isinstance(skills, list):
                formatted.append(f"  {', '.join(skills)}")
        
        return '\n'.join(formatted)
    
    def _render_pdf(self, data: Dict, company_name: str, job_title: str) -> str:
        """Render PDF using fpdf2 with pixel-perfect FAANG formatting."""
        
        pdf = PDFResume(orientation='P', unit='mm', format='Letter')
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.add_page()
        
        # FAANG-style margins: 0.4 inches = 10.16mm ≈ 10mm
        pdf.set_margins(10, 10, 10)
        
        # --- 1. HEADER (Name + Contact) ---
        header = data.get("header", {})
        
        # Name - centered, 20pt, bold
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(0)
        pdf.cell(0, 8, header.get("name", ""), align="C")
        pdf.ln(8)
        
        # Contact Info - centered, 10pt
        pdf.set_font("Helvetica", "", 10)
        contact_parts = []
        if header.get("email"):
            contact_parts.append(header["email"])
        if header.get("phone"):
            contact_parts.append(header["phone"])
        if header.get("location"):
            contact_parts.append(header["location"])
        
        # Links on same line
        links = header.get("links", [])
        contact_parts.extend(links)
        
        contact_line = "  |  ".join(contact_parts)
        pdf.cell(0, 5, contact_line, align="C")
        pdf.ln(8)
        
        # --- 2. PROFESSIONAL SUMMARY ---
        if data.get("summary"):
            pdf.section_title("Professional Summary")
            pdf.set_font("Helvetica", "", 10.5)
            pdf.set_text_color(0)
            pdf.multi_cell(0, 5, data["summary"])
        
        # --- 3. TECHNICAL SKILLS ---
        if data.get("skills"):
            pdf.section_title("Technical Skills")
            pdf.set_font("Helvetica", "", 10)
            
            for category, items in data["skills"].items():
                # Bold category name
                pdf.set_font("Helvetica", "B", 10)
                pdf.write(5, f"{category}: ")
                
                # Regular items
                pdf.set_font("Helvetica", "", 10)
                pdf.write(5, ", ".join(items))
                pdf.ln(5)
        
        # --- 4. EXPERIENCE ---
        if data.get("experience"):
            pdf.section_title("Experience")
            
            for role in data["experience"]:
                # Job title + Company (left) | Date (right)
                pdf.set_font("Helvetica", "B", 11)
                
                # Calculate width for left part (title + company)
                title_text = f"{role.get('title', '')} - {role.get('company', '')}"
                date_text = role.get('date', '')
                
                # Split the line: title on left, date on right
                pdf.cell(140, 6, title_text, align="L")
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 6, date_text, align="R")
                pdf.ln(6)
                
                # Location (if provided)
                if role.get('location'):
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(60)
                    pdf.cell(0, 4, role['location'], align="L")
                    pdf.ln(4)
                    pdf.set_text_color(0)
                
                # Bullet points
                pdf.set_font("Helvetica", "", 10.5)
                for bullet in role.get("bullets", []):
                    pdf.set_x(15)  # Indent
                    # Using simple dash for bullets (Latin-1 compatible)
                    pdf.multi_cell(0, 5, f"-  {bullet}")
                
                pdf.ln(2)
        
        # --- 5. EDUCATION ---
        if data.get("education"):
            pdf.section_title("Education")
            
            for edu in data["education"]:
                # School name (left) | Year (right)
                pdf.set_font("Helvetica", "B", 10.5)
                pdf.cell(140, 6, edu.get('school', ''), align="L")
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 6, edu.get('year', ''), align="R")
                pdf.ln(6)
                
                # Degree
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 5, edu.get('degree', ''), align="L")
                pdf.ln(5)
                
                # GPA / Relevant coursework
                details = []
                if edu.get('gpa'):
                    details.append(f"GPA: {edu['gpa']}")
                if edu.get('relevant'):
                    details.append(edu['relevant'])
                
                if details:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(60)
                    pdf.cell(0, 4, " | ".join(details), align="L")
                    pdf.ln(4)
                    pdf.set_text_color(0)
                
                pdf.ln(2)
        
        # --- 6. PROJECTS (if any) ---
        if data.get("projects"):
            pdf.section_title("Projects")
            
            for project in data["projects"]:
                # Project name (bold)
                pdf.set_font("Helvetica", "B", 10.5)
                pdf.write(5, project.get('name', '') + ": ")
                
                # Description
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 5, project.get('description', ''))
                
                # Technologies
                if project.get('tech'):
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(60)
                    pdf.cell(0, 4, f"Technologies: {', '.join(project['tech'])}", align="L")
                    pdf.ln(4)
                    pdf.set_text_color(0)
                
                pdf.ln(2)
        
        # Generate filename and save
        safe_company = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = "".join(c for c in job_title if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"Resume_{safe_company}_{safe_title}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        pdf.output(filepath)
        return filepath


# Convenience function
def generate_simple_pdf(
    parsed_resume: Dict,
    job_description: str,
    company_name: str,
    job_title: str,
    style: str = "professional"
) -> str:
    """
    Generate PDF resume - simple interface.
    
    Example:
        from parsers.resume_parser import parse_resume
        resume = parse_resume("my_resume.pdf")
        pdf = generate_simple_pdf(
            resume,
            "Looking for Python engineer...",
            "Google",
            "Software Engineer"
        )
    """
    generator = SimplePDFGenerator()
    return generator.generate_resume_pdf(
        parsed_resume, job_description, company_name, job_title, style
    )


if __name__ == "__main__":
    print("Simple PDF Generator - FPDF2 + JSON Approach")
    print("=" * 50)
    
    # Mock resume data
    mock_resume = {
        'contact_info': {
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '(555) 123-4567',
            'linkedin': 'linkedin.com/in/johndoe',
            'github': 'github.com/johndoe'
        },
        'summary': 'Experienced software engineer with 5+ years in Python and cloud technologies.',
        'experience': [
            {
                'title': 'Senior Software Engineer',
                'company': 'Tech Corp',
                'start_date': '2020-01',
                'end_date': 'Present',
                'achievements': [
                    'Led team of 5 engineers in building microservices platform',
                    'Reduced API latency by 40% through caching and optimization',
                    'Implemented CI/CD pipeline reducing deployment time by 75%',
                    'Architected system serving 1M+ daily active users'
                ]
            },
            {
                'title': 'Software Engineer',
                'company': 'StartupXYZ',
                'start_date': '2018-06',
                'end_date': '2020-01',
                'achievements': [
                    'Built REST APIs using Django serving 100K+ requests/day',
                    'Developed real-time analytics dashboard with React',
                    'Reduced database query time by 60% through indexing'
                ]
            }
        ],
        'education': [
            {
                'degree': 'Bachelor of Science',
                'field': 'Computer Science',
                'institution': 'University XYZ',
                'graduation_date': '2018',
                'gpa': '3.8'
            }
        ],
        'skills': {
            'Programming': ['Python', 'JavaScript', 'SQL', 'Go'],
            'Cloud': ['AWS', 'Docker', 'Kubernetes', 'Terraform'],
            'Tools': ['Git', 'Jenkins', 'PostgreSQL', 'Redis']
        }
    }
    
    # Generate PDF
    pdf_path = generate_simple_pdf(
        mock_resume,
        "Looking for senior Python engineer with cloud experience and proven track record of building scalable systems",
        "Example Company",
        "Senior Python Engineer"
    )
    
    print(f"\n✅ Generated: {pdf_path}")
