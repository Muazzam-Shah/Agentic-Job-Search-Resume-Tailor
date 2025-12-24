"""
Simplified Conversational Job Hunter Agent

A conversational AI that orchestrates all job hunting tools.
Simplified version without LangGraph StateGraph for Python 3.12 compatibility.
"""

from typing import TypedDict, List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import json
import os
from datetime import datetime

# Import all our existing tools
from tools.job_fetcher import JobFetcher
from parsers.resume_parser import ResumeParser
from tools.keyword_extractor import KeywordExtractor
from tools.semantic_matcher import SemanticMatcher
from generators.resume_generator import ResumeGenerator
from generators.simple_pdf_generator import generate_simple_pdf
from generators.cover_letter_generator import CoverLetterGenerator
from tools.company_researcher import CompanyResearcher


class ConversationState(TypedDict):
    """State maintained throughout the conversation"""
    messages: List
    user_input: str
    agent_response: str
    current_intent: str
    workflow_stage: str
    resume_data: Optional[Dict]
    resume_filepath: Optional[str]
    current_jobs: List[Dict]
    saved_jobs: List[Dict]
    selected_job: Optional[Dict]
    generated_resumes: List[Dict]
    generated_cover_letters: List[Dict]
    uploaded_files: Dict[str, str]
    last_tool_output: Optional[Dict]
    tools_used: List[str]
    context: Dict


class SimpleConversationalAgent:
    """
    Simplified conversational agent for job hunting.
    Compatible with Python 3.12.
    """
    
    def __init__(self):
        """Initialize the agent with LLM and tools"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # Initialize all tools
        self.job_fetcher = JobFetcher()
        self.resume_parser = ResumeParser()
        self.keyword_extractor = KeywordExtractor()
        self.semantic_matcher = SemanticMatcher()
        self.resume_generator = ResumeGenerator()
        self.company_researcher = CompanyResearcher()
        self.cover_letter_generator = CoverLetterGenerator()
    
    def _initialize_state(self) -> ConversationState:
        """Initialize conversation state"""
        return {
            "messages": [],
            "user_input": "",
            "agent_response": "",
            "current_intent": "",
            "workflow_stage": "initial",
            "resume_data": None,
            "resume_filepath": None,
            "current_jobs": [],
            "saved_jobs": [],
            "selected_job": None,
            "generated_resumes": [],
            "generated_cover_letters": [],
            "uploaded_files": {},
            "last_tool_output": None,
            "tools_used": [],
            "context": {}
        }
    
    def _classify_intent(self, user_input: str, state: ConversationState) -> str:
        """Classify user's intent"""
        context_info = []
        if state.get("resume_data"):
            context_info.append("User has uploaded a resume")
        if state.get("current_jobs"):
            context_info.append(f"User has {len(state['current_jobs'])} job results")
        if state.get("selected_job"):
            context_info.append(f"User selected job: {state['selected_job'].get('title', 'Unknown')}")
        
        context_str = " | ".join(context_info) if context_info else "No prior context"
        
        # Quick pattern matching for common intents
        user_lower = user_input.lower().strip()
        
        # Check for job selection patterns
        if state.get("current_jobs") and not state.get("selected_job"):
            # User has jobs but hasn't selected one
            selection_patterns = ["first", "second", "third", "fourth", "fifth", 
                                "#1", "#2", "#3", "#4", "#5",
                                "1st", "2nd", "3rd", "4th", "5th"]
            
            # Check if user input is just a number or ordinal
            if user_lower in ["1", "2", "3", "4", "5"] or any(p in user_lower for p in selection_patterns):
                return "select_job"
            
            # Check if user mentions a job title or company from current jobs
            for job in state['current_jobs']:
                job_title = job.get('title', '').lower()
                company = job.get('company', '').lower()
                if (len(job_title) > 5 and job_title in user_lower) or \
                   (len(company) > 3 and company in user_lower):
                    return "select_job"
        
        # Check for resume generation when job is selected but no request to generate
        if state.get("selected_job") and state.get("resume_data") and \
           any(word in user_lower for word in ["resume", "tailor", "generate", "create"]):
            return "generate_resume"
        
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a job hunting assistant.
Classify the user's intent into ONE of these categories:

1. search_jobs - User wants to search for jobs
2. upload_resume - User mentions uploading/sharing resume  
3. generate_resume - User wants to create tailored resume
4. generate_cover_letter - User wants cover letter
5. get_feedback - User wants resume critique
6. select_job - User is selecting a specific job (numbers, ordinals, job titles)
7. general_question - User asking questions
8. continue_workflow - User confirming action (yes, ok, sure)

Context: {context}

Respond with ONLY the intent category name."""),
            ("user", "{user_input}")
        ])
        
        chain = intent_prompt | self.llm
        result = chain.invoke({
            "user_input": user_input,
            "context": context_str
        })
        
        intent = result.content.strip().lower().replace(" ", "_")
        return intent
    
    def _execute_action(self, intent: str, user_input: str, state: ConversationState) -> ConversationState:
        """Execute action based on intent"""
        try:
            if intent == "search_jobs":
                state = self._handle_job_search(state, user_input)
            elif intent == "select_job":
                state = self._handle_job_selection(state, user_input)
            elif intent == "generate_resume":
                # If user says "generate resume for job X", first select the job
                if not state.get("selected_job") and state.get("current_jobs"):
                    state = self._handle_job_selection(state, user_input)
                    # Only proceed if selection was successful
                    if not state.get("selected_job"):
                        return state
                state = self._handle_resume_generation(state)
            elif intent == "generate_cover_letter":
                state = self._handle_cover_letter_generation(state)
            elif intent == "continue_workflow":
                state = self._handle_workflow_continuation(state)
            else:
                state["last_tool_output"] = None
        except Exception as e:
            state["last_tool_output"] = {
                "error": str(e),
                "message": f"I encountered an error: {str(e)}"
            }
        
        return state
    
    def _handle_job_search(self, state: ConversationState, user_input: str) -> ConversationState:
        """Search for jobs"""
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the job search query from the user's message.
Respond in JSON format: {{"query": "job title", "location": "location or Remote"}}

Examples:
- "Find Python developer jobs in New York" → {{"query": "Python developer", "location": "New York"}}
- "Search for remote software engineer positions" → {{"query": "software engineer", "location": "Remote"}}
"""),
            ("user", "{input}")
        ])
        
        chain = extract_prompt | self.llm
        result = chain.invoke({"input": user_input})
        
        try:
            search_params = json.loads(result.content)
        except:
            search_params = {"query": user_input, "location": "Remote"}
        
        jobs = self.job_fetcher.search_jobs(
            query=search_params["query"],
            location=search_params.get("location", "Remote"),
            max_results=5
        )
        
        state["current_jobs"] = jobs
        state["last_tool_output"] = {
            "type": "job_results",
            "jobs": jobs,
            "count": len(jobs)
        }
        state["tools_used"].append("job_search")
        state["workflow_stage"] = "jobs_searched"
        
        return state
    
    def _handle_job_selection(self, state: ConversationState, user_input: str) -> ConversationState:
        """Handle user selecting a specific job"""
        jobs = state.get("current_jobs", [])
        
        if not jobs:
            state["last_tool_output"] = {"error": "No jobs available. Please search first."}
            return state
        
        user_lower = user_input.lower()
        job_index = None
        
        # Try to match ordinal keywords
        ordinals = [
            (["first", "1st", "#1", " 1 ", "one"], 0),
            (["second", "2nd", "#2", " 2 ", "two"], 1),
            (["third", "3rd", "#3", " 3 ", "three"], 2),
            (["fourth", "4th", "#4", " 4 ", "four"], 3),
            (["fifth", "5th", "#5", " 5 ", "five"], 4),
        ]
        
        for keywords, idx in ordinals:
            for keyword in keywords:
                if keyword in user_lower:
                    job_index = idx
                    break
            if job_index is not None:
                break
        
        # If no ordinal match, try to match job title or company name
        if job_index is None:
            for i, job in enumerate(jobs):
                job_title = job.get('title', '').lower()
                company = job.get('company', '').lower()
                
                # Check if user input contains significant parts of job title or company
                if (len(job_title) > 5 and job_title in user_lower) or \
                   (len(company) > 3 and company in user_lower):
                    job_index = i
                    break
        
        if job_index is not None and job_index < len(jobs):
            state["selected_job"] = jobs[job_index]
            state["last_tool_output"] = {
                "type": "job_selected",
                "job": jobs[job_index]
            }
            state["workflow_stage"] = "job_selected"
        else:
            state["last_tool_output"] = {"error": "Please specify which job by number (e.g., '2') or mention the job title"}
        
        return state
    
    def _handle_resume_generation(self, state: ConversationState) -> ConversationState:
        """Generate tailored resume"""
        resume_data = state.get("resume_data")
        selected_job = state.get("selected_job")
        
        if not resume_data:
            state["last_tool_output"] = {"error": "Please upload your resume first."}
            return state
        
        if not selected_job:
            state["last_tool_output"] = {"error": "Please select a job first."}
            return state
        
        job_desc = selected_job.get("description", "")
        if not job_desc:
            job_desc = selected_job.get("job_description", "")
        
        keywords = self.keyword_extractor.extract_keywords(job_desc)
        
        # Get raw text from resume data for semantic matching
        resume_text = resume_data.raw_text if hasattr(resume_data, 'raw_text') else ""
        
        match_analysis = self.semantic_matcher.analyze_match(
            resume_text=resume_text,
            parsed_resume=resume_data,
            job_description=job_desc
        )
        
        docx_path = self.resume_generator.generate_tailored_resume(
            parsed_resume=resume_data,
            job_description=job_desc,
            job_title=selected_job["title"],
            company_name=selected_job["company"]
        )
        
        pdf_path = generate_simple_pdf(
            parsed_resume=resume_data,
            job_description=job_desc,
            job_title=selected_job["title"],
            company_name=selected_job["company"]
        )
        
        doc_info = {
            "docx_path": docx_path,
            "pdf_path": pdf_path,
            "job_title": selected_job["title"],
            "company": selected_job["company"],
            "match_score": match_analysis["composite_score"],
            "generated_at": datetime.now().isoformat()
        }
        
        state["generated_resumes"].append(doc_info)
        state["last_tool_output"] = {
            "type": "resume_generated",
            "documents": doc_info,
            "match_analysis": match_analysis
        }
        state["tools_used"].extend(["keyword_extraction", "semantic_matching", "resume_generation"])
        state["workflow_stage"] = "resume_generated"
        
        return state
    
    def _handle_cover_letter_generation(self, state: ConversationState) -> ConversationState:
        """Generate cover letter"""
        resume_data = state.get("resume_data")
        selected_job = state.get("selected_job")
        
        if not resume_data or not selected_job:
            state["last_tool_output"] = {"error": "Please upload resume and select a job first."}
            return state
        
        cover_letter_content = self.cover_letter_generator.generate_cover_letter(
            resume_data=resume_data,
            company_name=selected_job["company"],
            job_title=selected_job["title"],
            job_description=selected_job.get("description", selected_job.get("job_description", "")),
            style="professional"
        )
        
        from generators.cover_letter_pdf import generate_cover_letter_pdf
        
        # Access Pydantic model field
        candidate_name = resume_data.contact_info.name if hasattr(resume_data, 'contact_info') else "Candidate"
        
        pdf_path = generate_cover_letter_pdf(
            cover_letter_content=cover_letter_content,
            candidate_name=candidate_name,
            job_title=selected_job["title"],
            company_name=selected_job["company"]
        )
        
        doc_info = {
            "pdf_path": pdf_path,
            "job_title": selected_job["title"],
            "company": selected_job["company"],
            "generated_at": datetime.now().isoformat()
        }
        
        state["generated_cover_letters"].append(doc_info)
        state["last_tool_output"] = {
            "type": "cover_letter_generated",
            "document": doc_info
        }
        state["tools_used"].append("cover_letter_generation")
        state["workflow_stage"] = "cover_letter_generated"
        
        return state
    
    def _handle_workflow_continuation(self, state: ConversationState) -> ConversationState:
        """Handle user confirming to continue workflow"""
        stage = state.get("workflow_stage", "initial")
        
        if stage == "job_selected" and state.get("resume_data"):
            return self._handle_resume_generation(state)
        elif stage == "resume_generated":
            return self._handle_cover_letter_generation(state)
        else:
            state["last_tool_output"] = {"message": "What would you like to do?"}
        
        return state
    
    def _generate_response(self, intent: str, user_input: str, state: ConversationState) -> str:
        """Generate natural language response"""
        tool_output = state.get("last_tool_output")
        
        if tool_output:
            if "error" in tool_output:
                return f"⚠️ {tool_output['error']}"
            
            if tool_output.get("type") == "job_results":
                jobs = tool_output["jobs"]
                if not jobs:
                    return "I couldn't find any matching jobs. Try different keywords?"
                
                response = f"I found {len(jobs)} positions!\n\n"
                for i, job in enumerate(jobs[:5], 1):
                    response += f"{i}. {job['title']} at {job['company']}\n"
                response += "\nWhich position interests you?"
                return response
            
            elif tool_output.get("type") == "job_selected":
                job = tool_output["job"]
                response = f"Great! You selected: {job['title']} at {job['company']}\n\n"
                if state.get("resume_data"):
                    response += "I can generate a tailored resume. Should I proceed?"
                else:
                    response += "Please upload your resume to create a tailored version."
                return response
            
            elif tool_output.get("type") == "resume_generated":
                docs = tool_output["documents"]
                return f"✅ Your tailored resume is ready!\n\nMatch Score: {docs['match_score']:.0f}/100\n\nDocuments generated. Would you like a cover letter?"
            
            elif tool_output.get("type") == "cover_letter_generated":
                return "✅ Your cover letter is ready! Documents are available for download."
        
        # Default conversational response
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful job hunting assistant. Respond naturally and helpfully."),
            ("user", "{user_input}")
        ])
        
        chain = response_prompt | self.llm
        result = chain.invoke({"user_input": user_input})
        return result.content
    
    def chat(self, user_input: str, state: Optional[ConversationState] = None) -> tuple[str, ConversationState]:
        """
        Process user input and return response.
        """
        if state is None or not state:
            state = self._initialize_state()
            greeting = """Hi! I'm your Job Hunter AI assistant.

I can help you with:
- Searching for jobs across multiple platforms
- Creating tailored resumes (DOCX and PDF)
- Writing personalized cover letters
- Reviewing and improving your resume

What would you like to work on today?"""
            state["agent_response"] = greeting
            return greeting, state
        
        # Process user input
        state["user_input"] = user_input
        
        # Classify intent
        intent = self._classify_intent(user_input, state)
        state["current_intent"] = intent
        
        # Execute action
        state = self._execute_action(intent, user_input, state)
        
        # Generate response
        response = self._generate_response(intent, user_input, state)
        state["agent_response"] = response
        
        return response, state
    
    def handle_file_upload(self, filepath: str, state: ConversationState) -> tuple[str, ConversationState]:
        """Handle resume file upload"""
        try:
            resume_data = self.resume_parser.parse_file(filepath)
            
            # Keep as Pydantic model for compatibility with other tools
            state["resume_data"] = resume_data
            state["resume_filepath"] = filepath
            state["uploaded_files"]["resume"] = filepath
            state["workflow_stage"] = "resume_uploaded"
            state["tools_used"].append("resume_parser")
            
            # Access Pydantic model fields
            name = resume_data.contact_info.name
            skills_count = len(resume_data.skills) if resume_data.skills else 0
            
            response = f"✅ Resume analyzed!\n\nCandidate: {name}\nSkills: {skills_count} identified\n\n"
            
            if state.get("selected_job"):
                job = state["selected_job"]
                response += f"Ready to generate resume for {job['title']}. Should I proceed?"
            elif state.get("current_jobs"):
                response += f"You have {len(state['current_jobs'])} jobs. Which one interests you?"
            else:
                response += "What would you like to do?\n1. Search for jobs\n2. Get resume feedback"
            
            state["agent_response"] = response
            return response, state
            
        except Exception as e:
            error_msg = f"Error processing resume: {str(e)}"
            state["agent_response"] = error_msg
            return error_msg, state


# Global instance
_agent_instance = None

def get_agent() -> SimpleConversationalAgent:
    """Get or create the global agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SimpleConversationalAgent()
    return _agent_instance
