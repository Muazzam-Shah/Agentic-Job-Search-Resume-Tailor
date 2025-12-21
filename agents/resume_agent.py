"""
Resume Tailor Agent using LangChain and OpenAI
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import MessagesPlaceholder
from typing import List, Dict
import os


class ResumeTailorAgent:
    """Agent for tailoring resumes to job descriptions"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the resume tailor agent
        
        Args:
            model: OpenAI model to use (gpt-4o-mini is cheaper, gpt-4o for better quality)
            temperature: Model temperature (0-1)
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def tailor_resume(self, resume_text: str, job_description: str) -> str:
        """
        Tailor a resume to match a job description
        
        Args:
            resume_text: Original resume content
            job_description: Target job description
            
        Returns:
            Tailored resume suggestions
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume consultant. Analyze the resume and job description, 
            then provide specific suggestions to tailor the resume to the job.
            
            Focus on:
            1. Keywords from job description to add
            2. Skills to emphasize
            3. Experience to highlight
            4. Achievements to quantify
            5. Format improvements
            
            Be specific and actionable."""),
            ("user", """Resume:
{resume_text}

Job Description:
{job_description}

Provide tailored suggestions:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description
        })
        
        return result
    
    def generate_cover_letter(self, resume_text: str, job_description: str, company_info: str = "") -> str:
        """
        Generate a tailored cover letter
        
        Args:
            resume_text: Resume content
            job_description: Job description
            company_info: Additional company information
            
        Returns:
            Generated cover letter
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert cover letter writer. Create a compelling, 
            professional cover letter that matches the candidate's experience to the job requirements.
            
            Make it:
            - Professional yet personable
            - Specific to the role
            - Highlighting relevant achievements
            - 3-4 paragraphs maximum"""),
            ("user", """Resume:
{resume_text}

Job Description:
{job_description}

Company Info:
{company_info}

Write a tailored cover letter:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description,
            "company_info": company_info or "No additional company information provided."
        })
        
        return result
    
    def extract_keywords(self, job_description: str) -> List[str]:
        """
        Extract important keywords from job description
        
        Args:
            job_description: Job description text
            
        Returns:
            List of important keywords
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the most important technical skills, qualifications, 
            and keywords from the job description. Return them as a comma-separated list.
            Focus on: technical skills, tools, certifications, and key qualifications."""),
            ("user", "{job_description}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({"job_description": job_description})
        
        # Parse the result into a list
        keywords = [kw.strip() for kw in result.split(',')]
        return keywords


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = ResumeTailorAgent()
    
    sample_resume = """
    John Doe
    Software Engineer
    
    Experience:
    - 3 years Python development
    - Built web applications with Django
    - Worked with PostgreSQL databases
    """
    
    sample_job = """
    Senior Python Developer
    
    Requirements:
    - 5+ years Python experience
    - Experience with FastAPI and LangChain
    - Strong database skills (PostgreSQL)
    - AI/ML project experience
    """
    
    print("🎯 Tailoring Resume...\n")
    suggestions = agent.tailor_resume(sample_resume, sample_job)
    print(suggestions)
    
    print("\n" + "="*60)
    print("📝 Extracting Keywords...\n")
    keywords = agent.extract_keywords(sample_job)
    print("Keywords:", keywords)
