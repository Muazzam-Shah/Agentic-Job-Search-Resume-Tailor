"""
Flask Web Application for Job Hunter AI

Provides a web interface for the Job Hunter agentic AI system.
Allows users to upload resumes, search jobs, and generate tailored resumes.

Author: Job Hunter Team
Date: December 21, 2025
"""

import os
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import json

from agents.job_hunter_agent import JobHunterAgent
from tools.job_fetcher import JobFetcher
from parsers.resume_parser import ResumeParser
from generators.resume_generator import ResumeGenerator
from rag.job_corpus import JobCorpusManager
from rag.resume_history import ResumeHistoryTracker
from utils.logger import logger

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Initialize components
agent = JobHunterAgent()
job_fetcher = JobFetcher()
resume_parser = ResumeParser()
resume_generator = ResumeGenerator()
job_corpus = JobCorpusManager()
resume_tracker = ResumeHistoryTracker()


# ============================================================================
# ROUTES - HTML PAGES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')


# ============================================================================
# API ENDPOINTS - JOB SEARCH
# ============================================================================

@app.route('/api/jobs/search', methods=['POST'])
def search_jobs():
    """Search for jobs"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        location = data.get('location', 'Remote')
        max_results = data.get('max_results', 20)
        
        logger.info(f"Job search: query='{query}', location='{location}', max_results={max_results}")
        
        # Search jobs
        jobs = job_fetcher.search_jobs(
            query=query,
            location=location,
            max_results=max_results
        )
        
        # Add to corpus
        for job in jobs:
            job_corpus.add_job(
                job_id=job.get('id', f"job_{hash(job.get('title', ''))}"),
                title=job.get('title', 'Unknown Position'),
                company=job.get('company', 'Unknown Company'),
                description=job.get('description', ''),
                location=job.get('location', location),
                salary=f"{job.get('salary_min', 'N/A')} - {job.get('salary_max', 'N/A')}" if job.get('salary_min') or job.get('salary_max') else 'Not specified',
                employment_type=job.get('employment_type', ''),
                apply_link=job.get('apply_link', ''),
                source=job.get('source', '')
            )
        
        return jsonify({
            'success': True,
            'count': len(jobs),
            'jobs': jobs
        })
        
    except Exception as e:
        logger.error(f"Job search failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/jobs/<job_id>')
def get_job(job_id):
    """Get job details"""
    try:
        job = job_corpus.get_job_by_id(job_id)
        
        if job:
            return jsonify({
                'success': True,
                'job': job
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API ENDPOINTS - RESUME OPERATIONS
# ============================================================================

@app.route('/api/resume/upload', methods=['POST'])
def upload_resume():
    """Upload and parse resume"""
    try:
        if 'resume' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No resume file provided'
            }), 400
        
        file = request.files['resume']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Resume uploaded: {filename}")
        
        # Parse resume
        parsed_resume = resume_parser.parse_file(filepath)
        
        # Convert Pydantic model to dict for JSON serialization
        parsed_data = parsed_resume.model_dump()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'parsed_data': parsed_data
        })
        
    except Exception as e:
        logger.error(f"Resume upload failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/resume/generate', methods=['POST'])
def generate_resume():
    """Generate tailored resume"""
    try:
        data = request.get_json()
        resume_file = data.get('resume_file')
        job_description = data.get('job_description')
        company_name = data.get('company_name', 'Company')
        job_title = data.get('job_title', 'Position')
        
        if not resume_file or not job_description:
            return jsonify({
                'success': False,
                'error': 'Resume file and job description required'
            }), 400
        
        logger.info(f"Generating resume for {company_name} - {job_title}")
        
        # Parse the resume file first
        parsed_resume = resume_parser.parse_file(resume_file)
        
        # Generate resume
        output_path = resume_generator.generate_tailored_resume(
            parsed_resume=parsed_resume,
            job_description=job_description,
            company_name=company_name,
            job_title=job_title
        )
        
        # Track application
        resume_tracker.track_application(
            application_id=f"app_{Path(output_path).stem}",
            job_id=data.get('job_id', 'unknown'),
            job_title=job_title,
            company=company_name,
            resume_content=job_description,  # Simplified
            bullet_points=[],
            keywords_used=[]
        )
        
        return jsonify({
            'success': True,
            'output_path': output_path,
            'filename': Path(output_path).name
        })
        
    except Exception as e:
        logger.error(f"Resume generation failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/resume/download/<filename>')
def download_resume(filename):
    """Download generated resume"""
    try:
        # Check in output/resumes subfolder first (where ResumeGenerator saves)
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], 'resumes', secure_filename(filename))
        
        # Fallback to root output folder
        if not os.path.exists(filepath):
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            logger.error(f"Resume file not found: {filename}")
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        logger.info(f"Downloading resume: {filename}")
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Resume download failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API ENDPOINTS - RAG & ANALYTICS
# ============================================================================

@app.route('/api/rag/search', methods=['POST'])
def rag_search():
    """RAG-powered search"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        mode = data.get('mode', 'jobs')
        k = data.get('k', 5)
        
        # Use agent's RAG search tool
        result = agent.run(f"Search the knowledge base for: {query} (mode: {mode}, k: {k})")
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analytics/skills')
def get_skill_trends():
    """Get skill trends"""
    try:
        trends = job_corpus.get_skill_trends(top_n=20)
        
        return jsonify({
            'success': True,
            'trends': [{'skill': skill, 'count': count} for skill, count in trends]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analytics/companies')
def get_company_stats():
    """Get company statistics"""
    try:
        stats = job_corpus.get_company_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analytics/success-rate')
def get_success_rate():
    """Get application success rate"""
    try:
        metrics = resume_tracker.get_success_rate()
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API ENDPOINTS - AGENT
# ============================================================================

@app.route('/api/agent/query', methods=['POST'])
def agent_query():
    """Query the agent"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        logger.info(f"Agent query: {query}")
        
        # Run agent
        result = agent.run(query)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("JOB HUNTER - WEB APPLICATION")
    print("=" * 80)
    print("\n🚀 Starting server...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"\n🌐 Access the app at: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
