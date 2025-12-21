# Job Hunter - AI-Powered Resume & Job Search Platform

An intelligent agentic AI system that automates resume tailoring for job applications using LangChain, GPT-4, and RAG technology. Now with a professional web interface!

## 🚀 Quick Start (Web Interface)

**The fastest way to get started:**

### Windows
```bash
start.bat
```

### Linux/Mac
```bash
chmod +x start.sh
./start.sh
```

Then open your browser to **http://localhost:5000**

---

**For CLI usage or detailed setup:** [QUICKSTART.md](QUICKSTART.md)  
**For RAG features:** [RAG_QUICKSTART.md](RAG_QUICKSTART.md)  
**For web frontend details:** [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)  
**For project progress:** [PROGRESS.md](PROGRESS.md)

## 🎯 Project Overview

Job Hunter is an autonomous agent system with a professional web interface that:
- 🔍 **Fetches job data** using multiple APIs (JSearch, Adzuna)
- 🤖 **Parses resumes** using GPT-4 powered extraction
- 📝 **Generates tailored resumes** optimized for ATS systems
- 🧠 **Uses RAG** for intelligent job matching and recommendations
- 📊 **Provides analytics** on skills, companies, and success rates
- 🌐 **Web interface** for easy interaction (no command line needed!)

## ✨ Key Features

### Web Interface (NEW!)
- 📤 **Resume Upload**: Drag-and-drop PDF/DOCX upload with automatic parsing
- 🔎 **Job Search**: Search across multiple job boards from one interface
- ⚡ **One-Click Generation**: Generate tailored resumes with a single click
- 📈 **Analytics Dashboard**: View skill trends, company stats, and success metrics
- 💾 **Easy Export**: Download resumes as professionally formatted DOCX files

### AI-Powered Backend
- 🎯 **Smart Matching**: RAG-based job-resume matching
- 🔧 **Multi-Tool Agent**: LangChain ReAct agent with 7+ specialized tools
- 💾 **Vector Storage**: ChromaDB for semantic search
- 📊 **Success Tracking**: Monitor application outcomes and improve strategies

## 🏗️ Architecture

### Multi-Layer System:
1. **Web Frontend** (Phase 9 ✅)
   - HTML5 + Tailwind CSS
   - Vanilla JavaScript
   - Responsive design

2. **Flask Backend** (Phase 9 ✅)
   - RESTful API
   - File upload/download
   - Integration layer

3. **Agent Orchestrator** (Phase 7 ✅)
   - LangChain ReAct Agent
   - 7+ specialized tools
   - Conversational interface

4. **RAG Engine** (Phase 8 ✅)
   - ChromaDB vector store
   - 4 search strategies
   - Job corpus analytics
   - Success tracking

5. **Core Services** (Phases 1-6 ✅)
   - Job API integration
   - Resume parsing (GPT-4)
   - Resume generation
   - Keyword extraction

## 🚀 Quick Start (Detailed)

### Prerequisites
- Python 3.12+
- OpenAI API key
- Adzuna API credentials (free tier)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file:
```env
# OpenAI API (Required)
OPENAI_API_KEY=sk-your-key-here

# Job APIs (At least one required)
ADZUNA_APP_ID=your_app_id
ADZUNA_API_KEY=your_api_key

# Optional
GITHUB_TOKEN=your_github_token
GOOGLE_API_KEY=your_google_key
GOOGLE_CSE_ID=your_cse_id
```

5. **Run the web application**
```bash
python app.py
```

Or use the quick start scripts:
- Windows: `start.bat`
- Linux/Mac: `./start.sh`

6. **Access the web interface**

Open your browser to: **http://localhost:5000**

## 📁 Project Structure

```
Project/
├── app.py                    # Flask web application ⭐ NEW
├── templates/                # HTML templates ⭐ NEW
│   ├── index.html           # Landing page
│   └── dashboard.html       # Main application
├── uploads/                  # Uploaded resumes ⭐ NEW
├── output/                   # Generated resumes
├── agents/                   # LangChain agent implementations
│   └── job_hunter_agent.py  # Main ReAct agent
├── tools/                    # Custom tools
│   ├── job_fetcher.py       # Multi-source job search
│   └── langchain_tools.py   # 7 LangChain tools
├── rag/                      # RAG system
│   ├── vector_store.py      # ChromaDB integration
│   ├── rag_retriever.py     # 4 search strategies
│   ├── job_corpus.py        # Job analytics
│   └── resume_history.py    # Success tracking
├── parsers/                  # Resume parsers
├── generators/               # Resume generators
├── utils/                    # Utilities
├── tests/                    # 100+ tests
├── data/                     # Sample data & ChromaDB
├── examples/                 # Usage examples
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
├── FRONTEND_GUIDE.md         # Web interface guide ⭐ NEW
├── RAG_QUICKSTART.md         # RAG features guide
├── QUICKSTART.md             # CLI quick start
├── PROGRESS.md               # Development progress
└── README.md                 # This file
```

## 🔧 Configuration

### Getting API Keys:

1. **OpenAI API Key**: 
   - Sign up at [platform.openai.com](https://platform.openai.com)
   - Go to API keys section
   - Create new secret key

2. **Adzuna API**: 
   - Sign up at [developer.adzuna.com](https://developer.adzuna.com)
   - Free tier: 250 requests/month
   - Get App ID and API Key

3. **Optional APIs**:
   - GitHub: For additional job sources
   - Google: For web search integration

## 💻 Usage

### Web Interface (Recommended)

1. **Start the server**: `python app.py` or `start.bat`
2. **Upload Resume**: Drag-and-drop your resume (PDF/DOCX)
3. **Search Jobs**: Enter job title and location
4. **Generate Resume**: Click "Use for Resume" on any job listing
5. **Download**: Get your tailored DOCX resume

### Command Line Interface

```bash
# Interactive agent
python examples/cli_example.py

# Direct generation
python main.py --job "Software Engineer at Google" --resume resume.pdf
```

### Python API

```python
from agents.job_hunter_agent import create_job_hunter_agent

agent = create_job_hunter_agent()
result = agent.run("Find software engineering jobs in San Francisco and tailor my resume for the best match")
```

## 📊 Features in Detail

### 1. Resume Upload & Parsing
- Supports PDF and DOCX formats
- GPT-4 powered extraction
- Extracts: name, email, phone, skills, experience, education
- Handles various resume formats

### 2. Job Search
- **Multi-source**: Adzuna, JSearch, and more
- **Smart filtering**: By location, salary, company
- **Corpus building**: Automatically builds searchable job database

### 3. Resume Generation
- **ATS-optimized**: Passes applicant tracking systems
- **Tailored content**: Matches job requirements
- **Professional format**: Clean DOCX output
- **Smart bullet points**: Uses proven successful phrases

### 4. RAG System
- **Semantic search**: Find relevant jobs by meaning, not just keywords
- **4 strategies**: Hybrid, semantic, keyword, filtered
- **Analytics**: Skill trends, company insights
- **Success tracking**: Learn from past applications

### 5. Analytics Dashboard
- **Top skills**: Most in-demand skills
- **Company stats**: Who's hiring most
- **Success rates**: Interview/offer/rejection metrics
- **Quick stats**: Overview of your job search

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=. tests/

# Specific test suite
pytest tests/test_job_fetcher.py
pytest tests/test_rag.py
pytest tests/test_agent.py
```

## 📚 Documentation

- **[FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)**: Complete web interface guide
- **[RAG_QUICKSTART.md](RAG_QUICKSTART.md)**: RAG system quick start
- **[QUICKSTART.md](QUICKSTART.md)**: CLI and setup guide
- **[PROGRESS.md](PROGRESS.md)**: Development progress (95% complete!)
- **Phase Summaries**: Detailed docs for each phase
  - [Phase6_Summary.md](Phase6_Summary.md): Agent orchestration
  - [Phase8_Summary.md](Phase8_Summary.md): RAG integration
  - [Phase9_Summary.md](Phase9_Summary.md): Web frontend

## 🎓 Academic Context

This is a practical agent system project for:
- **Course**: Agentic AI
- **Track**: Track A - Practical Agent System
- **Semester**: 7
- **Instructor**: Engr. Asima Sarwar

### Project Meets Requirements:
✅ Real actions (file creation, API calls, document generation)  
✅ Autonomous decisions (keyword matching, content selection)  
✅ External tools (job APIs, vector DB, LLM APIs)  
✅ Reasoning capability (ReAct pattern)  
✅ Tangible outputs (DOCX files, analytics)  
✅ 5+ evaluation methods implemented

## 🤝 Contributing

This is an academic project, but feedback and suggestions are welcome!

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

**[Your Name]**  
Semester 7 - Agentic AI Course  
Instructor: Engr. Asima Sarwar

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- LangChain for agent framework
- Adzuna for job API
- Tailwind CSS for styling
- ChromaDB for vector storage

---

**Last Updated:** January 7, 2025  
**Project Status:** 95% Complete (Phase 9/12)  
**Next Phase:** Evaluation Framework
