"""
Feedback Chatbot - Conversational Resume Improvement System

This module provides an interactive chatbot for resume feedback and
iterative improvement using LangChain conversational agent.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.resume_parser import ParsedResume, parse_resume
from tools.resume_critic import ResumeCritic, ComprehensiveFeedback
from generators.resume_generator import ResumeGenerator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Conversation State Models
# ============================================================================

class ConversationState(BaseModel):
    """Track conversation state and improvement progress."""
    session_id: str
    current_resume: Optional[Dict] = None
    original_resume: Optional[Dict] = None
    feedback_history: List[Dict] = Field(default_factory=list)
    improvements_made: List[str] = Field(default_factory=list)
    iteration_count: int = 0
    focus_area: Optional[str] = None
    job_description: Optional[str] = None


class ImprovementSuggestion(BaseModel):
    """Single improvement suggestion."""
    category: str
    section: str
    current_text: str
    suggested_text: str
    rationale: str
    priority: str  # high, medium, low


# ============================================================================
# Feedback Chatbot Class
# ============================================================================

class FeedbackChatbot:
    """
    Conversational chatbot for resume feedback and improvement.
    
    Provides multi-turn conversations with memory, intelligent suggestions,
    and iterative improvement workflow.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        session_id: Optional[str] = None
    ):
        """
        Initialize the feedback chatbot.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for conversational responses
            session_id: Optional session ID for tracking
        """
        self.model = model
        self.temperature = temperature
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        # Initialize memory for multi-turn conversations
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Initialize tools
        self.critic = ResumeCritic()
        self.generator = ResumeGenerator()
        
        # Initialize state
        self.state = ConversationState(session_id=self.session_id)
        
        # Conversation statistics
        self.stats = {
            "messages_exchanged": 0,
            "suggestions_provided": 0,
            "improvements_applied": 0,
            "session_start": datetime.now().isoformat()
        }
        
        logger.info(f"FeedbackChatbot initialized - Session: {self.session_id}")
    
    
    def start_session(self, resume_path: str, job_description: Optional[str] = None) -> str:
        """
        Start a feedback session with a resume.
        
        Args:
            resume_path: Path to resume file (PDF/DOCX)
            job_description: Optional job description for targeted feedback
        
        Returns:
            Welcome message with initial assessment
        """
        logger.info(f"Starting feedback session for: {resume_path}")
        
        # Parse resume
        parsed_resume = parse_resume(resume_path)
        self.state.current_resume = parsed_resume.dict()
        self.state.original_resume = parsed_resume.dict()
        self.state.job_description = job_description
        
        # Get initial critique
        feedback = self.critic.critique_resume(parsed_resume, job_description)
        self.state.feedback_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "initial_critique",
            "feedback": feedback.dict()
        })
        
        # Create welcome message
        welcome = self._format_initial_feedback(feedback)
        
        # Save to memory
        self.memory.save_context(
            {"input": "Please review my resume"},
            {"output": welcome}
        )
        
        self.stats["messages_exchanged"] += 1
        
        return welcome
    
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return response.
        
        Args:
            user_message: User's message/question
        
        Returns:
            Chatbot's response
        """
        logger.info(f"Processing message: {user_message[:100]}...")
        
        # Analyze intent
        intent = self._analyze_intent(user_message)
        
        # Route to appropriate handler
        if intent == "request_feedback":
            response = self._handle_feedback_request(user_message)
        elif intent == "ask_about_section":
            response = self._handle_section_question(user_message)
        elif intent == "request_improvement":
            response = self._handle_improvement_request(user_message)
        elif intent == "compare_versions":
            response = self._handle_comparison_request()
        elif intent == "general_question":
            response = self._handle_general_question(user_message)
        else:
            response = self._handle_general_conversation(user_message)
        
        # Save to memory
        self.memory.save_context(
            {"input": user_message},
            {"output": response}
        )
        
        self.stats["messages_exchanged"] += 1
        
        return response
    
    
    def improve_section(
        self,
        section: str,
        specific_improvement: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Generate improved version of a resume section.
        
        Args:
            section: Section to improve (summary, experience, skills, etc.)
            specific_improvement: Optional specific improvement to make
        
        Returns:
            Tuple of (improved_text, explanation)
        """
        logger.info(f"Generating improvement for section: {section}")
        
        if not self.state.current_resume:
            return "Please upload a resume first.", None
        
        parsed_resume = ParsedResume(**self.state.current_resume)
        
        # Get current section text
        current_text = self._get_section_text(parsed_resume, section)
        
        # Generate improvement
        improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume writer. Improve the given resume section 
            while maintaining the candidate's authentic voice. Focus on:
            - Action verbs and quantifiable achievements
            - Clear, concise language
            - ATS-friendly keywords
            - Professional tone"""),
            ("user", """Improve this resume section:
            
            SECTION: {section}
            CURRENT TEXT:
            {current_text}
            
            {specific_instruction}
            
            Provide:
            1. Improved version
            2. Brief explanation of changes (2-3 sentences)
            
            Return as JSON: {{"improved": "...", "explanation": "..."}}""")
        ])
        
        specific_instruction = ""
        if specific_improvement:
            specific_instruction = f"SPECIFIC FOCUS: {specific_improvement}"
        
        try:
            formatted = improvement_prompt.format_messages(
                section=section,
                current_text=current_text,
                specific_instruction=specific_instruction
            )
            
            response = self.llm.invoke(formatted)
            content = response.content.strip()
            
            # Parse response
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            
            result = json.loads(content)
            improved_text = result.get("improved", current_text)
            explanation = result.get("explanation", "Section improved")
            
            # Track improvement
            self.state.improvements_made.append({
                "section": section,
                "timestamp": datetime.now().isoformat(),
                "improvement": specific_improvement or "general improvement"
            })
            self.stats["improvements_applied"] += 1
            
            return improved_text, explanation
            
        except Exception as e:
            logger.error(f"Error improving section: {e}")
            return current_text, "Unable to generate improvement"
    
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history.
        
        Returns:
            List of message dictionaries
        """
        messages = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages
    
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session.
        
        Returns:
            Dictionary with session statistics and progress
        """
        summary = {
            "session_id": self.session_id,
            "statistics": self.stats,
            "state": {
                "iterations": self.state.iteration_count,
                "improvements_made": len(self.state.improvements_made),
                "focus_area": self.state.focus_area
            },
            "progress": []
        }
        
        # Calculate progress if we have feedback history
        if len(self.state.feedback_history) > 1:
            initial = self.state.feedback_history[0]["feedback"]
            latest = self.state.feedback_history[-1]["feedback"]
            
            summary["progress"] = {
                "initial_score": initial.get("overall_score", 0),
                "current_score": latest.get("overall_score", 0),
                "improvement": latest.get("overall_score", 0) - initial.get("overall_score", 0)
            }
        
        return summary
    
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze user message intent."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["feedback", "review", "analyze", "critique"]):
            return "request_feedback"
        elif any(word in message_lower for word in ["summary", "experience", "skills", "education"]):
            return "ask_about_section"
        elif any(word in message_lower for word in ["improve", "rewrite", "fix", "better"]):
            return "request_improvement"
        elif any(word in message_lower for word in ["compare", "before", "after", "progress"]):
            return "compare_versions"
        elif any(word in message_lower for word in ["what", "how", "why", "should", "can"]):
            return "general_question"
        else:
            return "general_conversation"
    
    
    def _handle_feedback_request(self, message: str) -> str:
        """Handle request for feedback on specific area."""
        if not self.state.current_resume:
            return "Please upload a resume first to get feedback."
        
        parsed_resume = ParsedResume(**self.state.current_resume)
        
        # Determine focus area from message
        focus_areas = []
        if "content" in message.lower():
            focus_areas.append("content")
        if "format" in message.lower():
            focus_areas.append("format")
        if "keyword" in message.lower():
            focus_areas.append("keywords")
        if "ats" in message.lower():
            focus_areas.append("ats")
        
        # Get targeted feedback
        feedback = self.critic.critique_resume(
            parsed_resume,
            self.state.job_description,
            focus_areas if focus_areas else None
        )
        
        # Format response
        response_parts = [f"Here's my feedback on your resume:"]
        
        if not focus_areas or "content" in focus_areas:
            response_parts.append(f"\n**Content Quality (Score: {feedback.content.score}/100)**")
            if feedback.content.issues:
                response_parts.append("Issues found:")
                for issue in feedback.content.issues[:3]:
                    response_parts.append(f"- {issue}")
            if feedback.content.suggestions:
                response_parts.append("\nSuggestions:")
                for suggestion in feedback.content.suggestions[:3]:
                    response_parts.append(f"- {suggestion}")
        
        if not focus_areas or "keywords" in focus_areas:
            response_parts.append(f"\n**Keywords (Score: {feedback.keywords.score}/100)**")
            if feedback.keywords.missing_keywords:
                response_parts.append(f"Missing keywords: {', '.join(feedback.keywords.missing_keywords[:5])}")
        
        response_parts.append(f"\n**Overall Score: {feedback.overall_score}/100**")
        response_parts.append(f"\nTop priorities:")
        for priority in feedback.top_priorities[:3]:
            response_parts.append(f"1. {priority}")
        
        self.stats["suggestions_provided"] += 1
        
        return "\n".join(response_parts)
    
    
    def _handle_section_question(self, message: str) -> str:
        """Handle questions about specific resume sections."""
        if not self.state.current_resume:
            return "Please upload a resume first."
        
        # Extract section from message
        section = None
        if "summary" in message.lower():
            section = "summary"
        elif "experience" in message.lower():
            section = "experience"
        elif "skills" in message.lower():
            section = "skills"
        elif "education" in message.lower():
            section = "education"
        
        if not section:
            return "Which section would you like to discuss? (summary, experience, skills, or education)"
        
        # Get suggestions for that section
        parsed_resume = ParsedResume(**self.state.current_resume)
        suggestions = self.critic.suggest_improvements(parsed_resume, "content", section)
        
        response = f"Here are specific improvements for your {section} section:\n\n"
        for i, suggestion in enumerate(suggestions[:5], 1):
            response += f"{i}. {suggestion}\n"
        
        response += f"\nWould you like me to rewrite your {section} section for you?"
        
        return response
    
    
    def _handle_improvement_request(self, message: str) -> str:
        """Handle request to improve a section."""
        # Extract section
        section = None
        if "summary" in message.lower():
            section = "summary"
        elif "experience" in message.lower():
            section = "experience"
        
        if not section:
            return "Which section would you like me to improve? Please specify (summary, experience, etc.)"
        
        # Improve section
        improved, explanation = self.improve_section(section)
        
        response = f"Here's an improved version of your {section}:\n\n"
        response += f"**Improved {section.title()}:**\n{improved}\n\n"
        response += f"**Changes Made:**\n{explanation}\n\n"
        response += "Would you like to apply this change or request further modifications?"
        
        return response
    
    
    def _handle_comparison_request(self) -> str:
        """Handle request to compare resume versions."""
        if not self.state.original_resume or not self.state.current_resume:
            return "I need both an original and modified resume to compare."
        
        if len(self.state.improvements_made) == 0:
            return "No improvements have been made yet to compare."
        
        response = f"**Progress Summary:**\n\n"
        response += f"Improvements made: {len(self.state.improvements_made)}\n"
        response += f"Iterations: {self.state.iteration_count}\n\n"
        response += "Changes applied:\n"
        
        for improvement in self.state.improvements_made:
            response += f"- {improvement.get('section', 'Unknown')}: {improvement.get('improvement', 'Modified')}\n"
        
        return response
    
    
    def _handle_general_question(self, message: str) -> str:
        """Handle general resume-related questions."""
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", """You are a helpful resume advisor. Answer questions about resume best practices,
            formatting, content, and job search strategies. Be concise but informative."""),
            ("user", "{input}")
        ])
        
        try:
            # Get chat history
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            formatted = prompt.format_messages(
                chat_history=chat_history[-10:],  # Last 10 messages
                input=message
            )
            
            response = self.llm.invoke(formatted)
            return response.content
            
        except Exception as e:
            logger.error(f"Error handling question: {e}")
            return "I'm here to help with your resume. Could you please rephrase your question?"
    
    
    def _handle_general_conversation(self, message: str) -> str:
        """Handle general conversational messages."""
        return self._handle_general_question(message)
    
    
    def _format_initial_feedback(self, feedback: ComprehensiveFeedback) -> str:
        """Format initial feedback into welcome message."""
        message = f"""Hello! I've reviewed your resume. Here's my initial assessment:

**Overall Score: {feedback.overall_score}/100**

**Strengths:**
"""
        for strength in feedback.strengths[:3]:
            message += f"✓ {strength}\n"
        
        message += "\n**Top Priorities for Improvement:**\n"
        for i, priority in enumerate(feedback.top_priorities[:3], 1):
            message += f"{i}. {priority}\n"
        
        message += f"""
**Detailed Scores:**
- Content: {feedback.content.score}/100
- Format: {feedback.format.score}/100
- Keywords: {feedback.keywords.score}/100
- ATS Compatibility: {feedback.ats.compatibility_score}/100

I'm here to help you improve! You can ask me to:
- Review a specific section (e.g., "How's my summary?")
- Suggest improvements (e.g., "Improve my experience section")
- Answer questions (e.g., "What are action verbs?")
- Compare versions

What would you like to work on first?
"""
        return message
    
    
    def _get_section_text(self, parsed_resume: ParsedResume, section: str) -> str:
        """Extract text from a specific resume section."""
        if section.lower() == "summary":
            return parsed_resume.summary or "No summary found"
        elif section.lower() == "experience":
            text = []
            for exp in parsed_resume.experience[:3]:
                text.append(f"{exp.title} at {exp.company}")
                text.extend(exp.description[:3])
            return "\n".join(text)
        elif section.lower() == "skills":
            return ", ".join(parsed_resume.skills[:20])
        elif section.lower() == "education":
            text = []
            for edu in parsed_resume.education:
                text.append(f"{edu.degree} - {edu.institution}")
            return "\n".join(text)
        else:
            return "Section not found"


# ============================================================================
# Convenience Functions
# ============================================================================

def start_feedback_session(resume_path: str, job_description: Optional[str] = None) -> FeedbackChatbot:
    """
    Quick function to start a feedback session.
    
    Args:
        resume_path: Path to resume file
        job_description: Optional job description
    
    Returns:
        Initialized FeedbackChatbot
    """
    chatbot = FeedbackChatbot()
    welcome = chatbot.start_session(resume_path, job_description)
    print(welcome)
    return chatbot
