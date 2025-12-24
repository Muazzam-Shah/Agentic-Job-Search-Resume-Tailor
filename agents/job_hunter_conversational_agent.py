"""
LangGraph-based Conversational Job Hunter Agent

A truly agentic AI assistant that orchestrates all job hunting tools through
natural conversation. Uses LangGraph for state management and intelligent routing.
"""

from typing import TypedDict, List, Optional, Dict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
from agents.feedback_chatbot import FeedbackChatbot


class ConversationState(TypedDict):
    """State maintained throughout the conversation"""
    
    # Conversation
    messages: Annotated[List[BaseMessage], "Full conversation history"]
    user_input: str
    agent_response: str
    
    # Intent & Workflow
    current_intent: str  # search_jobs, upload_resume, generate_resume, etc.
    workflow_stage: str  # initial, jobs_searched, resume_uploaded, docs_generated
    
    # User Data
    resume_data: Optional[Dict]
    resume_filepath: Optional[str]
    
    # Jobs
    current_jobs: List[Dict]
    saved_jobs: List[Dict]
    selected_job: Optional[Dict]
    
    # Generated Documents
    generated_resumes: List[Dict]  # [{path, job_title, company, score}]
    generated_cover_letters: List[Dict]  # [{path, job_title, company, style}]
    
    # Files
    uploaded_files: Dict[str, str]
    
    # Tool Outputs
    last_tool_output: Optional[Dict]
    tools_used: List[str]
    
    # Context
    context: Dict


class ConversationalJobHunterAgent:
    """
    LangGraph-powered conversational agent that orchestrates all job hunting tools.
    
    Features:
    - Natural language understanding
    - Autonomous tool selection and execution
    - Multi-turn conversation with memory
    - Proactive guidance through workflow
    - Rich responses (job cards, download links, scores)
    """
    
    def __init__(self):
        """Initialize the agent with LLM and tools"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=True
        )
        
        # Initialize all tools
        self.job_fetcher = JobFetcher()
        self.resume_parser = ResumeParser()
        self.keyword_extractor = KeywordExtractor()
        self.semantic_matcher = SemanticMatcher()
        self.resume_generator = ResumeGenerator()
        self.company_researcher = CompanyResearcher()
        self.cover_letter_generator = CoverLetterGenerator()
        self.feedback_chatbot = None  # Initialized when needed
        
        # Build the LangGraph
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("greet", self._greet_user)
        workflow.add_node("understand_intent", self._classify_intent)
        workflow.add_node("execute_action", self._execute_action)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("check_continuation", self._check_continuation)
        
        # Set entry point
        workflow.set_entry_point("greet")
        
        # Add edges
        workflow.add_edge("greet", "understand_intent")
        workflow.add_edge("understand_intent", "execute_action")
        workflow.add_edge("execute_action", "generate_response")
        workflow.add_edge("generate_response", "check_continuation")
        
        # Conditional edge from check_continuation
        workflow.add_conditional_edges(
            "check_continuation",
            self._should_continue,
            {
                "continue": "understand_intent",
                "end": END
            }
        )
        
        return workflow
    
    def _greet_user(self, state: ConversationState) -> ConversationState:
        """Initial greeting"""
        if not state.get("messages"):
            greeting = """Hi! I'm your Job Hunter AI assistant. 👋

I can help you with:
🔍 Searching for jobs across multiple platforms
📄 Creating tailored resumes (DOCX and FAANG-style PDFs)
✉️ Writing personalized cover letters with company research
📊 Reviewing and improving your resume
💡 Job matching and keyword analysis

What would you like to work on today?"""
            
            state["messages"] = [AIMessage(content=greeting)]
            state["agent_response"] = greeting
            state["workflow_stage"] = "initial"
            state["context"] = {}
            state["tools_used"] = []
            state["saved_jobs"] = []
            state["generated_resumes"] = []
            state["generated_cover_letters"] = []
            state["uploaded_files"] = {}
            
        return state
    
    def _classify_intent(self, state: ConversationState) -> ConversationState:
        """Classify user's intent using LLM"""
        user_input = state.get("user_input", "")
        
        if not user_input:
            state["current_intent"] = "greeting"
            return state
        
        # Build context from conversation
        context_info = []
        if state.get("resume_data"):
            context_info.append("User has uploaded a resume")
        if state.get("current_jobs"):
            context_info.append(f"User has {len(state['current_jobs'])} job results")
        if state.get("selected_job"):
            context_info.append(f"User selected job: {state['selected_job']['job_title']}")
        
        context_str = " | ".join(context_info) if context_info else "No prior context"
        
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a job hunting assistant.
Classify the user's intent into ONE of these categories:

1. search_jobs - User wants to search for jobs (keywords: "find jobs", "search", "looking for", job titles)
2. upload_resume - User mentions uploading/sharing resume (keywords: "upload", "my resume", "here's my resume")
3. generate_resume - User wants to create tailored resume (keywords: "generate", "create resume", "tailor my resume")
4. generate_cover_letter - User wants cover letter (keywords: "cover letter", "write a letter")
5. get_feedback - User wants resume critique (keywords: "review", "feedback", "improve", "critique")
6. select_job - User is selecting a specific job (keywords: "first one", "that one", "job #2", "the Netflix position")
7. general_question - User asking questions (keywords: "how", "what", "can you", "tell me")
8. continue_workflow - User confirming action (keywords: "yes", "ok", "sure", "go ahead", "do it")

Context: {context}

Respond with ONLY the intent category name, nothing else."""),
            ("user", "{user_input}")
        ])
        
        intent_chain = intent_prompt | self.llm
        result = intent_chain.invoke({
            "user_input": user_input,
            "context": context_str
        })
        
        intent = result.content.strip().lower().replace(" ", "_")
        state["current_intent"] = intent
        
        return state
    
    def _execute_action(self, state: ConversationState) -> ConversationState:
        """Execute action based on intent"""
        intent = state["current_intent"]
        user_input = state.get("user_input", "")
        
        try:
            if intent == "search_jobs":
                state = self._handle_job_search(state, user_input)
                
            elif intent == "upload_resume":
                # This will be handled by file upload endpoint
                state["last_tool_output"] = {
                    "message": "Please upload your resume file (PDF or DOCX)"
                }
                
            elif intent == "generate_resume":
                state = self._handle_resume_generation(state)
                
            elif intent == "generate_cover_letter":
                state = self._handle_cover_letter_generation(state)
                
            elif intent == "get_feedback":
                state = self._handle_feedback(state)
                
            elif intent == "select_job":
                state = self._handle_job_selection(state, user_input)
                
            elif intent == "continue_workflow":
                state = self._handle_workflow_continuation(state)
                
            else:  # general_question or greeting
                state["last_tool_output"] = None
                
        except Exception as e:
            state["last_tool_output"] = {
                "error": str(e),
                "message": f"I encountered an error: {str(e)}. Let me try to help you differently."
            }
        
        return state
    
    def _handle_job_search(self, state: ConversationState, user_input: str) -> ConversationState:
        """Search for jobs"""
        # Extract job query from user input using LLM
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the job search query from the user's message.
Respond in JSON format:
{
    "query": "job title",
    "location": "location or 'Remote' or 'Anywhere'"
}

Examples:
- "Find me Python developer jobs in New York" → {"query": "Python developer", "location": "New York"}
- "Search for remote software engineer positions" → {"query": "software engineer", "location": "Remote"}
- "Looking for data scientist roles" → {"query": "data scientist", "location": "Anywhere"}
"""),
            ("user", "{input}")
        ])
        
        chain = extract_prompt | self.llm
        result = chain.invoke({"input": user_input})
        
        try:
            search_params = json.loads(result.content)
        except:
            # Fallback parsing
            search_params = {
                "query": user_input,
                "location": "Remote"
            }
        
        # Search jobs
        jobs = self.job_fetcher.search_jobs(
            query=search_params["query"],
            location=search_params.get("location", "Remote"),
            num_results=5
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
            state["last_tool_output"] = {
                "error": "No jobs available. Please search for jobs first."
            }
            return state
        
        # Parse which job they selected
        user_lower = user_input.lower()
        
        job_index = None
        if "first" in user_lower or "#1" in user_lower or "1" in user_lower:
            job_index = 0
        elif "second" in user_lower or "#2" in user_lower or "2" in user_lower:
            job_index = 1
        elif "third" in user_lower or "#3" in user_lower or "3" in user_lower:
            job_index = 2
        elif "fourth" in user_lower or "#4" in user_lower or "4" in user_lower:
            job_index = 3
        elif "fifth" in user_lower or "#5" in user_lower or "5" in user_lower:
            job_index = 4
        
        if job_index is not None and job_index < len(jobs):
            state["selected_job"] = jobs[job_index]
            state["last_tool_output"] = {
                "type": "job_selected",
                "job": jobs[job_index]
            }
            state["workflow_stage"] = "job_selected"
        else:
            state["last_tool_output"] = {
                "error": "I couldn't determine which job you selected. Please specify (e.g., 'the first one')"
            }
        
        return state
    
    def _handle_resume_generation(self, state: ConversationState) -> ConversationState:
        """Generate tailored resume"""
        resume_data = state.get("resume_data")
        selected_job = state.get("selected_job")
        
        if not resume_data:
            state["last_tool_output"] = {
                "error": "Please upload your resume first so I can tailor it."
            }
            return state
        
        if not selected_job:
            state["last_tool_output"] = {
                "error": "Please select a job first so I know what to tailor your resume for."
            }
            return state
        
        # Extract keywords
        job_desc = selected_job.get("job_description", "")
        keywords = self.keyword_extractor.extract_keywords(job_desc)
        
        # Calculate match score
        match_analysis = self.semantic_matcher.analyze_match(
            resume_data=resume_data,
            job_description=job_desc
        )
        
        # Generate tailored resume (DOCX)
        docx_path = self.resume_generator.generate_tailored_resume(
            parsed_resume=resume_data,
            job_description=job_desc,
            job_title=selected_job["job_title"],
            company_name=selected_job["company_name"]
        )
        
        # Generate PDF version
        pdf_path = generate_simple_pdf(
            parsed_resume=resume_data,
            job_description=job_desc,
            job_title=selected_job["job_title"],
            company_name=selected_job["company_name"],
            output_dir="output/resumes/pdf"
        )
        
        # Store generated documents
        doc_info = {
            "docx_path": docx_path,
            "pdf_path": pdf_path,
            "job_title": selected_job["job_title"],
            "company": selected_job["company_name"],
            "match_score": match_analysis["composite_score"],
            "generated_at": datetime.now().isoformat()
        }
        
        state["generated_resumes"].append(doc_info)
        state["last_tool_output"] = {
            "type": "resume_generated",
            "documents": doc_info,
            "match_analysis": match_analysis
        }
        state["tools_used"].extend(["keyword_extraction", "semantic_matching", "resume_generation", "pdf_generation"])
        state["workflow_stage"] = "resume_generated"
        
        return state
    
    def _handle_cover_letter_generation(self, state: ConversationState) -> ConversationState:
        """Generate cover letter"""
        resume_data = state.get("resume_data")
        selected_job = state.get("selected_job")
        
        if not resume_data:
            state["last_tool_output"] = {
                "error": "Please upload your resume first."
            }
            return state
        
        if not selected_job:
            state["last_tool_output"] = {
                "error": "Please select a job first."
            }
            return state
        
        # Generate cover letter
        cover_letter_content = self.cover_letter_generator.generate_cover_letter(
            resume_data=resume_data,
            company_name=selected_job["company_name"],
            job_title=selected_job["job_title"],
            job_description=selected_job.get("job_description", ""),
            style="professional"
        )
        
        # Save as PDF
        from generators.cover_letter_pdf import generate_cover_letter_pdf
        
        pdf_path = generate_cover_letter_pdf(
            cover_letter_content=cover_letter_content,
            candidate_name=resume_data["contact_info"]["name"],
            job_title=selected_job["job_title"],
            company_name=selected_job["company_name"]
        )
        
        doc_info = {
            "pdf_path": pdf_path,
            "job_title": selected_job["job_title"],
            "company": selected_job["company_name"],
            "style": "professional",
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
    
    def _handle_feedback(self, state: ConversationState) -> ConversationState:
        """Provide resume feedback"""
        resume_data = state.get("resume_data")
        
        if not resume_data:
            state["last_tool_output"] = {
                "error": "Please upload your resume first so I can review it."
            }
            return state
        
        # Initialize feedback chatbot if not done
        if not self.feedback_chatbot:
            self.feedback_chatbot = FeedbackChatbot()
            self.feedback_chatbot.start_session(resume_data)
        
        # Get critique
        from tools.resume_critic import ResumeCritic
        
        critic = ResumeCritic()
        feedback = critic.critique_resume(resume_data)
        
        state["last_tool_output"] = {
            "type": "resume_feedback",
            "feedback": feedback
        }
        state["tools_used"].append("resume_critique")
        
        return state
    
    def _handle_workflow_continuation(self, state: ConversationState) -> ConversationState:
        """Handle user saying 'yes', 'ok', 'sure' to continue workflow"""
        stage = state.get("workflow_stage", "initial")
        
        # Determine what the user is confirming
        if stage == "job_selected":
            # They selected a job, probably want to generate resume
            if state.get("resume_data"):
                return self._handle_resume_generation(state)
            else:
                state["last_tool_output"] = {
                    "message": "Great! Please upload your resume so I can tailor it for this position."
                }
        
        elif stage == "resume_generated":
            # Resume is done, offer cover letter
            return self._handle_cover_letter_generation(state)
        
        else:
            state["last_tool_output"] = {
                "message": "What would you like to do?"
            }
        
        return state
    
    def _generate_response(self, state: ConversationState) -> ConversationState:
        """Generate natural language response based on tool outputs"""
        intent = state["current_intent"]
        tool_output = state.get("last_tool_output")
        user_input = state.get("user_input", "")
        
        # Build context for response generation
        context_parts = []
        
        if tool_output:
            if "error" in tool_output:
                response = f"⚠️ {tool_output['error']}\n\n{tool_output.get('message', '')}"
                
            elif tool_output.get("type") == "job_results":
                jobs = tool_output["jobs"]
                response = self._format_job_results(jobs)
                
            elif tool_output.get("type") == "job_selected":
                job = tool_output["job"]
                response = f"""Great choice! You selected:

📋 **{job['job_title']}**
🏢 {job['company_name']}
📍 {job.get('location', 'Location not specified')}

"""
                if state.get("resume_data"):
                    response += "I can now generate a tailored resume for this position. Should I proceed? 🚀"
                else:
                    response += "To create a tailored resume for this role, please upload your master resume (PDF or DOCX)."
                    
            elif tool_output.get("type") == "resume_generated":
                docs = tool_output["documents"]
                analysis = tool_output["match_analysis"]
                response = f"""🎉 **Your tailored resume is ready!**

📊 **Match Analysis:**
- Overall Score: {analysis['composite_score']:.0f}/100
- Match Strength: {analysis['match_strength']}

📄 **Generated Documents:**
- DOCX: {os.path.basename(docs['docx_path'])}
- PDF: {os.path.basename(docs['pdf_path'])}

✨ **Keywords Matched:** {analysis.get('keywords_matched', 'N/A')}

Would you like me to:
1. Generate a cover letter for this position?
2. Provide detailed feedback on your resume?
3. Search for more jobs?"""
                
            elif tool_output.get("type") == "cover_letter_generated":
                doc = tool_output["document"]
                response = f"""✉️ **Your personalized cover letter is ready!**

📄 {os.path.basename(doc['pdf_path'])}
🏢 For: {doc['company']} - {doc['job_title']}
🎨 Style: {doc['style'].title()}

Your cover letter includes:
✅ Company-specific research and insights
✅ Your achievements aligned with job requirements  
✅ Professional formatting matching your resume

Anything else you need help with?"""
                
            elif tool_output.get("type") == "resume_feedback":
                feedback = tool_output["feedback"]
                response = self._format_feedback(feedback)
                
            else:
                response = tool_output.get("message", "I'm ready to help! What would you like to do?")
        else:
            # No tool output, generate conversational response
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful job hunting assistant. Respond naturally and helpfully to the user.
Be conversational, encouraging, and proactive in suggesting next steps.

Current context:
- User has resume: {has_resume}
- Jobs found: {job_count}
- Selected job: {has_selected_job}
- Workflow stage: {stage}
"""),
                ("user", "{user_input}")
            ])
            
            chain = response_prompt | self.llm
            result = chain.invoke({
                "user_input": user_input,
                "has_resume": "Yes" if state.get("resume_data") else "No",
                "job_count": len(state.get("current_jobs", [])),
                "has_selected_job": "Yes" if state.get("selected_job") else "No",
                "stage": state.get("workflow_stage", "initial")
            })
            
            response = result.content
        
        # Add response to messages
        state["messages"].append(HumanMessage(content=user_input))
        state["messages"].append(AIMessage(content=response))
        state["agent_response"] = response
        
        return state
    
    def _format_job_results(self, jobs: List[Dict]) -> str:
        """Format job search results as text"""
        if not jobs:
            return "I couldn't find any jobs matching your criteria. Try a different search query or location?"
        
        response = f"I found {len(jobs)} positions! Here are the top matches:\n\n"
        
        for i, job in enumerate(jobs[:5], 1):
            response += f"""📋 **{i}. {job['job_title']}**
🏢 {job['company_name']}
📍 {job.get('location', 'Location not specified')}
💰 {job.get('salary', 'Salary not specified')}

"""
        
        response += "\nWhich position interests you? (e.g., 'the first one' or 'job #2')"
        
        return response
    
    def _format_feedback(self, feedback: Dict) -> str:
        """Format resume feedback"""
        overall = feedback.get("overall_score", 0)
        
        response = f"""📊 **Resume Analysis Complete!**

**Overall Score: {overall}/100**

"""
        
        categories = ["content", "format", "keywords", "ats"]
        for cat in categories:
            if cat in feedback:
                cat_data = feedback[cat]
                score = cat_data.get("score", 0)
                response += f"\n**{cat.title()}: {score}/100**\n"
                
                if cat_data.get("strengths"):
                    response += "✅ Strengths:\n"
                    for strength in cat_data["strengths"][:2]:
                        response += f"  • {strength}\n"
                
                if cat_data.get("issues"):
                    response += "⚠️ Areas to improve:\n"
                    for issue in cat_data["issues"][:2]:
                        response += f"  • {issue}\n"
        
        if feedback.get("top_priorities"):
            response += "\n💡 **Top 3 Priorities:**\n"
            for i, priority in enumerate(feedback["top_priorities"][:3], 1):
                response += f"{i}. {priority}\n"
        
        response += "\nWould you like me to help improve your resume?"
        
        return response
    
    def _check_continuation(self, state: ConversationState) -> ConversationState:
        """Check if conversation should continue"""
        # Mark as ready to end after processing
        state['ready_to_end'] = True
        return state
    
    def _should_continue(self, state: ConversationState) -> Literal["continue", "end"]:
        """Determine if conversation should continue or end"""
        # End after each response (single turn, wait for next user input)
        return "end"
    
    def chat(self, user_input: str, state: Optional[ConversationState] = None) -> tuple[str, ConversationState]:
        """
        Process user input and return response.
        
        Args:
            user_input: User's message
            state: Optional existing conversation state
            
        Returns:
            Tuple of (agent_response, updated_state)
        """
        if state is None:
            # Initialize new conversation
            state = {
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
            # Run greeting first
            state = self.app.invoke(state)
        
        # Add user input
        state["user_input"] = user_input
        
        # Run the graph
        state = self.app.invoke(state)
        
        return state["agent_response"], state
    
    def handle_file_upload(self, filepath: str, state: ConversationState) -> tuple[str, ConversationState]:
        """
        Handle resume file upload.
        
        Args:
            filepath: Path to uploaded resume file
            state: Current conversation state
            
        Returns:
            Tuple of (response, updated_state)
        """
        try:
            # Parse resume
            resume_data = self.resume_parser.parse_resume(filepath)
            
            state["resume_data"] = resume_data
            state["resume_filepath"] = filepath
            state["uploaded_files"]["resume"] = filepath
            state["workflow_stage"] = "resume_uploaded"
            state["tools_used"].append("resume_parser")
            
            # Generate response
            name = resume_data["contact_info"]["name"]
            skills_count = len(resume_data.get("skills", []))
            exp_count = len(resume_data.get("experience", []))
            
            response = f"""✅ **Resume uploaded and analyzed!**

👤 {name}
📧 {resume_data['contact_info']['email']}

**Profile Summary:**
- {skills_count} skills identified
- {exp_count} work experiences
- {resume_data.get('summary', 'No summary found')[:100]}...

"""
            
            if state.get("selected_job"):
                job = state["selected_job"]
                response += f"\nI can now generate a tailored resume for **{job['job_title']}** at **{job['company_name']}**. Should I proceed?"
            elif state.get("current_jobs"):
                response += f"\nYou have {len(state['current_jobs'])} jobs in your search results. Which one interests you?"
            else:
                response += "\nWhat would you like to do?\n1. Search for jobs\n2. Get resume feedback\n3. Generate tailored resume for a specific job"
            
            # Add to conversation
            state["messages"].append(AIMessage(content=response))
            state["agent_response"] = response
            
            return response, state
            
        except Exception as e:
            error_msg = f"Error processing resume: {str(e)}"
            state["messages"].append(AIMessage(content=error_msg))
            state["agent_response"] = error_msg
            return error_msg, state


# Global instance for Flask app
conversational_agent = None

def get_agent() -> ConversationalJobHunterAgent:
    """Get or create the global agent instance"""
    global conversational_agent
    if conversational_agent is None:
        conversational_agent = ConversationalJobHunterAgent()
    return conversational_agent
