"""
Job Hunter CLI - Interactive Command-Line Interface

This module provides a user-friendly command-line interface for the Job Hunter
agentic AI system. Users can interact with the agent through natural language
or use specific commands for common workflows.

Features:
- Natural language interaction with the agent
- Quick commands for common tasks (search, tailor, analyze)
- Session management and history
- Rich terminal output with colors and formatting
- Configuration management

Author: Job Hunter Team
Date: December 21, 2025
"""

import os
import sys
import json
from typing import Optional, Dict, Any
from pathlib import Path

from agents.job_hunter_agent import create_agent
from utils.logger import logger



class JobHunterCLI:
    """
    Interactive CLI for Job Hunter Agent.
    
    Provides a command-line interface with:
    - Natural language interaction
    - Quick commands for common tasks
    - Session persistence
    - Rich output formatting
    """
    
    COMMANDS = {
        "help": "Show available commands",
        "search": "Search for jobs (usage: search <query> [location])",
        "tailor": "Tailor resume for a job (usage: tailor <resume_path>)",
        "analyze": "Analyze resume-job match (usage: analyze <resume> <job_desc>)",
        "history": "Show conversation history",
        "stats": "Show session statistics",
        "clear": "Clear conversation memory",
        "exit": "Exit the application",
        "quit": "Exit the application"
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize CLI.
        
        Args:
            verbose: Enable verbose agent logging
        """
        self.verbose = verbose
        self.agent = None
        self.running = False
        
        print(self._get_banner())
    
    def _get_banner(self) -> str:
        """Get ASCII art banner."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         🎯 JOB HUNTER AGENT 🎯                               ║
║                                                                              ║
║                  AI-Powered Resume Tailoring Assistant                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Welcome! I'm your AI assistant for creating perfectly tailored resumes.

Type 'help' for available commands, or just chat with me naturally!
Example: "Help me tailor my resume for a Senior Python Developer role"

"""
    
    def start(self):
        """Start the interactive CLI session."""
        self.running = True
        
        try:
            # Initialize agent
            print("Initializing Job Hunter Agent...")
            self.agent = create_agent(verbose=self.verbose)
            print(f"✅ Agent ready! (Session: {self.agent.session_id})\n")
            
            # Main interaction loop
            while self.running:
                try:
                    # Get user input
                    user_input = input("You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Process input
                    self._process_input(user_input)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted. Type 'exit' to quit or continue chatting.")
                    continue
                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break
        
        except Exception as e:
            logger.error(f"CLI error: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            print("Please check your API keys and configuration.")
        
        finally:
            self._cleanup()
    
    def _process_input(self, user_input: str):
        """
        Process user input.
        
        Args:
            user_input: User's text input
        """
        # Check for commands
        if user_input.lower().startswith(tuple(self.COMMANDS.keys())):
            self._handle_command(user_input)
        else:
            # Natural language query - send to agent
            self._handle_query(user_input)
    
    def _handle_command(self, command_input: str):
        """
        Handle specific commands.
        
        Args:
            command_input: Command string
        """
        parts = command_input.lower().split(maxsplit=1)
        command = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "help":
            self._show_help()
        
        elif command == "search":
            if not args:
                print("Usage: search <query> [location]")
                print("Example: search Senior Python Developer Remote")
            else:
                self._handle_search(args)
        
        elif command == "tailor":
            if not args:
                print("Usage: tailor <resume_path>")
                print("Example: tailor ./resume.pdf")
                print("\nI'll then ask you for job details.")
            else:
                self._handle_tailor(args)
        
        elif command == "analyze":
            print("Usage: analyze <resume_path> <job_description_text>")
            print("Or just ask me naturally: 'Analyze my resume for this job...'")
        
        elif command == "history":
            self._show_history()
        
        elif command == "stats":
            self._show_stats()
        
        elif command == "clear":
            self.agent.reset_memory()
            print("✅ Conversation memory cleared.")
        
        elif command in ["exit", "quit"]:
            print("\n👋 Thanks for using Job Hunter! Goodbye!")
            self.running = False
        
        else:
            print(f"Unknown command: {command}")
            print("Type 'help' for available commands.")
    
    def _handle_query(self, query: str):
        """
        Handle natural language query.
        
        Args:
            query: User's natural language query
        """
        print("\n🤖 Agent: Thinking...\n")
        
        # Execute agent
        result = self.agent.run(query)
        
        if result["success"]:
            print(f"🤖 Agent:\n{result['output']}\n")
            
            # Show execution stats if verbose
            if self.verbose:
                print(f"⏱️  Execution time: {result['execution_time']:.2f}s")
                print(f"🔢 Estimated tokens: {result['tokens_used']}")
                print(f"📊 Steps taken: {len(result['intermediate_steps'])}\n")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}\n")
            print("Please try rephrasing your request or use 'help' for guidance.")
    
    def _handle_search(self, query: str):
        """
        Handle job search command.
        
        Args:
            query: Search query
        """
        # Parse query and location
        parts = query.rsplit(maxsplit=1)
        if len(parts) == 2 and len(parts[1]) <= 50:  # Assume last part might be location
            job_query = parts[0]
            location = parts[1]
        else:
            job_query = query
            location = "Remote"
        
        task = f"Search for '{job_query}' jobs in {location}. Show me the top 5 results with company names, job titles, and brief descriptions."
        
        self._handle_query(task)
    
    def _handle_tailor(self, resume_path: str):
        """
        Handle resume tailoring command.
        
        Args:
            resume_path: Path to master resume
        """
        # Validate resume exists
        if not os.path.exists(resume_path):
            print(f"❌ Resume file not found: {resume_path}")
            return
        
        print(f"\n📄 Master resume: {resume_path}")
        print("\nPlease provide job details:")
        print("  Option 1: Paste the job description (type 'desc' then paste)")
        print("  Option 2: Provide job title and location to search (e.g., 'Senior Python Developer, Remote')")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == "desc":
            print("\nPaste job description (press Ctrl+D when done, or Ctrl+Z on Windows):")
            job_desc_lines = []
            try:
                while True:
                    line = input()
                    job_desc_lines.append(line)
            except EOFError:
                pass
            
            job_description = "\n".join(job_desc_lines)
            
            if not job_description.strip():
                print("❌ No job description provided.")
                return
            
            company = input("\nCompany name: ").strip()
            job_title = input("Job title: ").strip()
            
            if not company or not job_title:
                print("❌ Company name and job title are required.")
                return
            
            task = f"""
Tailor my resume for this position:

Company: {company}
Job Title: {job_title}
Master Resume: {resume_path}

Job Description:
{job_description}

Please analyze the job requirements, extract keywords, and generate a tailored resume.
Save it to ./output directory and provide me with the file path.
"""
            self._handle_query(task)
        
        else:
            # Assume it's a job search query
            task = f"""
I want to tailor my resume for: {choice}

My master resume is at: {resume_path}

Please:
1. Search for this type of job
2. Find the best match
3. Analyze the requirements
4. Generate a tailored resume

Save to ./output directory.
"""
            self._handle_query(task)
    
    def _show_help(self):
        """Show available commands."""
        print("\n" + "=" * 80)
        print("AVAILABLE COMMANDS:")
        print("=" * 80)
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:12} - {desc}")
        print("=" * 80)
        print("\nNATURAL LANGUAGE EXAMPLES:")
        print("  - 'Search for Senior Python Developer jobs in San Francisco'")
        print("  - 'Tailor my resume for a Data Science role at Google'")
        print("  - 'Analyze how well my resume matches this job description...'")
        print("  - 'Show me the top 5 remote software engineer positions'")
        print("=" * 80 + "\n")
    
    def _show_history(self):
        """Show conversation history."""
        history = self.agent.get_conversation_history()
        
        if not history:
            print("\n📭 No conversation history yet.\n")
            return
        
        print("\n" + "=" * 80)
        print("CONVERSATION HISTORY:")
        print("=" * 80)
        for i, msg in enumerate(history, 1):
            role = "You" if msg["role"] == "human" else "Agent"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            print(f"\n{i}. {role}:")
            print(f"   {content}")
        print("=" * 80 + "\n")
    
    def _show_stats(self):
        """Show session statistics."""
        stats = self.agent.get_session_stats()
        
        print("\n" + "=" * 80)
        print("SESSION STATISTICS:")
        print("=" * 80)
        print(f"  Session ID:          {stats['session_id']}")
        print(f"  Model:               {stats['model']}")
        print(f"  Executions:          {stats['executions']}")
        print(f"  Total Tokens Used:   ~{stats['total_tokens_used']:,}")
        print(f"  Conversation Length: {stats['conversation_length']} messages")
        print(f"  Tools Available:     {stats['tools_available']}")
        print("=" * 80 + "\n")
    
    def _cleanup(self):
        """Cleanup before exit."""
        if self.agent:
            stats = self.agent.get_session_stats()
            logger.info(f"Session ended: {stats['executions']} executions, ~{stats['total_tokens_used']} tokens")


def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Job Hunter - AI-Powered Resume Tailoring Assistant"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose agent logging"
    )
    
    args = parser.parse_args()
    
    # Start CLI
    cli = JobHunterCLI(verbose=args.verbose)
    cli.start()


if __name__ == "__main__":
    main()
