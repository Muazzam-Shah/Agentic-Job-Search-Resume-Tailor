"""
Job Corpus Manager

Manages historical job descriptions and provides analytics.
Features job indexing, trend analysis, and company insights.

Author: AI Job Hunter
Date: 2024
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

from rag.vector_store import JobVectorStore
from utils.logger import logger


# ============================================================================
# JOB CORPUS MANAGER
# ============================================================================

class JobCorpusManager:
    """
    Manages a corpus of historical job descriptions.
    
    Features:
    - Index and store job descriptions
    - Track job trends over time
    - Company-specific insights
    - Skill demand analysis
    - Salary range tracking
    """
    
    def __init__(
        self,
        corpus_path: str = "data/job_corpus.json",
        vector_store: Optional[JobVectorStore] = None,
        collection_name: str = "job_corpus"
    ):
        """
        Initialize job corpus manager.
        
        Args:
            corpus_path: Path to save corpus metadata
            vector_store: Vector store for semantic search
            collection_name: ChromaDB collection name
        """
        self.corpus_path = Path(corpus_path)
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Vector store for semantic search
        self.vector_store = vector_store or JobVectorStore(
            collection_name=collection_name,
            use_chromadb=True
        )
        
        # Metadata storage
        self.jobs: Dict[str, Dict] = {}  # job_id -> job data
        self.load_corpus()
        
        logger.info(f"Job Corpus Manager initialized with {len(self.jobs)} jobs")
    
    # ========================================================================
    # ADDING & INDEXING
    # ========================================================================
    
    def add_job(
        self,
        job_id: str,
        title: str,
        company: str,
        description: str,
        **metadata
    ) -> bool:
        """
        Add a job to the corpus.
        
        Args:
            job_id: Unique job identifier
            title: Job title
            company: Company name
            description: Job description
            **metadata: Additional metadata (location, salary, etc.)
            
        Returns:
            True if added successfully
        """
        try:
            # Create job record
            job_data = {
                "job_id": job_id,
                "title": title,
                "company": company,
                "description": description,
                "added_at": datetime.now().isoformat(),
                **metadata
            }
            
            # Add to metadata storage
            self.jobs[job_id] = job_data
            
            # Add to vector store
            self.vector_store.add_documents([{
                "content": f"{title}\n\n{description}",
                "metadata": {
                    "job_id": job_id,
                    "title": title,
                    "company": company,
                    **metadata
                }
            }])
            
            # Sanitize for logging (remove emojis and special unicode)
            safe_title = title.encode('ascii', 'ignore').decode('ascii') if title else 'Unknown'
            safe_company = company.encode('ascii', 'ignore').decode('ascii') if company else 'Unknown'
            logger.info(f"Added job: {job_id} - {safe_title} at {safe_company}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add job {job_id}: {e}")
            return False
    
    def add_jobs_batch(self, jobs: List[Dict[str, Any]]) -> int:
        """
        Add multiple jobs to corpus.
        
        Args:
            jobs: List of job dictionaries
            
        Returns:
            Number of jobs added
        """
        count = 0
        for job in jobs:
            if self.add_job(**job):
                count += 1
        
        logger.info(f"Added {count}/{len(jobs)} jobs to corpus")
        return count
    
    # ========================================================================
    # SEARCHING & RETRIEVAL
    # ========================================================================
    
    def search_jobs(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar jobs.
        
        Args:
            query: Search query
            k: Number of results
            filter_metadata: Metadata filters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching jobs
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata,
            similarity_threshold=similarity_threshold
        )
        
        # Enrich with full job data
        enriched_results = []
        for result in results:
            job_id = result["metadata"].get("job_id")
            if job_id and job_id in self.jobs:
                enriched_results.append({
                    **self.jobs[job_id],
                    "similarity": result["similarity"]
                })
        
        return enriched_results
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def get_jobs_by_company(self, company: str) -> List[Dict]:
        """Get all jobs from a company"""
        return [
            job for job in self.jobs.values()
            if job["company"].lower() == company.lower()
        ]
    
    # ========================================================================
    # ANALYTICS
    # ========================================================================
    
    def get_skill_trends(self, top_n: int = 20) -> List[tuple]:
        """
        Analyze skill demand across all jobs.
        
        Returns:
            List of (skill, count) tuples, sorted by frequency
        """
        skill_counter = Counter()
        
        for job in self.jobs.values():
            # Extract skills from description (simple approach)
            description = job["description"].lower()
            
            # Common tech skills to look for
            skills = [
                'python', 'java', 'javascript', 'typescript', 'react', 'node.js',
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'sql', 'nosql',
                'mongodb', 'postgresql', 'redis', 'kafka', 'spark', 'hadoop',
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
                'langchain', 'openai', 'gpt', 'llm', 'machine learning', 'ai',
                'deep learning', 'nlp', 'computer vision', 'data science',
                'rest api', 'graphql', 'microservices', 'ci/cd', 'git',
                'linux', 'bash', 'terraform', 'ansible'
            ]
            
            for skill in skills:
                if skill in description:
                    skill_counter[skill] += 1
        
        return skill_counter.most_common(top_n)
    
    def get_company_stats(self) -> Dict[str, int]:
        """
        Get job counts by company.
        
        Returns:
            Dictionary of company -> job count
        """
        company_counter = Counter()
        for job in self.jobs.values():
            company_counter[job["company"]] += 1
        
        return dict(company_counter.most_common())
    
    def get_salary_ranges(self) -> Dict[str, List[str]]:
        """
        Get salary ranges by job category.
        
        Returns:
            Dictionary of category -> list of salaries
        """
        salary_by_category = defaultdict(list)
        
        for job in self.jobs.values():
            if "salary" in job and "category" in job:
                salary_by_category[job["category"]].append(job["salary"])
        
        return dict(salary_by_category)
    
    def get_location_stats(self) -> Dict[str, int]:
        """
        Get job counts by location.
        
        Returns:
            Dictionary of location -> job count
        """
        location_counter = Counter()
        
        for job in self.jobs.values():
            if "location" in job:
                location_counter[job["location"]] += 1
        
        return dict(location_counter.most_common())
    
    def get_temporal_trends(self) -> Dict[str, int]:
        """
        Analyze when jobs were added (by month).
        
        Returns:
            Dictionary of month -> job count
        """
        month_counter = Counter()
        
        for job in self.jobs.values():
            added_at = datetime.fromisoformat(job["added_at"])
            month_key = added_at.strftime("%Y-%m")
            month_counter[month_key] += 1
        
        return dict(sorted(month_counter.items()))
    
    def analyze_company(self, company: str) -> Dict[str, Any]:
        """
        Deep analysis of a specific company.
        
        Args:
            company: Company name
            
        Returns:
            Dictionary with company insights
        """
        company_jobs = self.get_jobs_by_company(company)
        
        if not company_jobs:
            return {"error": f"No jobs found for {company}"}
        
        # Extract skills
        skills = Counter()
        locations = Counter()
        salaries = []
        
        for job in company_jobs:
            # Skills (simplified)
            description = job["description"].lower()
            for skill in ['python', 'java', 'javascript', 'aws', 'docker', 'kubernetes']:
                if skill in description:
                    skills[skill] += 1
            
            # Locations
            if "location" in job:
                locations[job["location"]] += 1
            
            # Salaries
            if "salary" in job:
                salaries.append(job["salary"])
        
        return {
            "company": company,
            "total_jobs": len(company_jobs),
            "top_skills": dict(skills.most_common(10)),
            "locations": dict(locations),
            "salaries": salaries,
            "sample_titles": [job["title"] for job in company_jobs[:5]]
        }
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save_corpus(self) -> bool:
        """Save corpus metadata to disk"""
        try:
            with open(self.corpus_path, 'w') as f:
                json.dump(self.jobs, f, indent=2)
            
            logger.info(f"Saved {len(self.jobs)} jobs to {self.corpus_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save corpus: {e}")
            return False
    
    def load_corpus(self) -> bool:
        """Load corpus metadata from disk"""
        try:
            if self.corpus_path.exists():
                with open(self.corpus_path, 'r') as f:
                    self.jobs = json.load(f)
                
                logger.info(f"Loaded {len(self.jobs)} jobs from {self.corpus_path}")
                return True
            else:
                logger.info("No existing corpus found, starting fresh")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load corpus: {e}")
            self.jobs = {}
            return False
    
    def clear_corpus(self):
        """Clear all jobs from corpus"""
        self.jobs.clear()
        self.vector_store.clear()
        logger.info("Corpus cleared")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive corpus statistics.
        
        Returns:
            Dictionary with various statistics
        """
        return {
            "total_jobs": len(self.jobs),
            "total_companies": len(set(job["company"] for job in self.jobs.values())),
            "top_skills": self.get_skill_trends(10),
            "top_companies": dict(Counter(job["company"] for job in self.jobs.values()).most_common(10)),
            "locations": self.get_location_stats(),
            "temporal_trends": self.get_temporal_trends()
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 80)
    print("JOB CORPUS MANAGER TEST")
    print("=" * 80)
    
    # Initialize manager
    print("\n1️⃣  Initializing corpus manager...")
    manager = JobCorpusManager(
        corpus_path="data/test_corpus.json",
        collection_name="test_corpus"
    )
    print("   ✅ Manager initialized")
    
    # Add sample jobs
    print("\n2️⃣  Adding sample jobs...")
    sample_jobs = [
        {
            "job_id": "job_001",
            "title": "Senior Python Developer",
            "company": "AI Corp",
            "description": "Build AI applications with Python, LangChain, OpenAI API",
            "location": "San Francisco, CA",
            "salary": "$180k-$250k",
            "category": "engineering"
        },
        {
            "job_id": "job_002",
            "title": "Data Scientist",
            "company": "DataTech",
            "description": "Machine learning with Python, TensorFlow, PyTorch",
            "location": "Remote",
            "salary": "$160k-$220k",
            "category": "data-science"
        },
        {
            "job_id": "job_003",
            "title": "Full Stack Engineer",
            "company": "AI Corp",
            "description": "React, Node.js, TypeScript, AWS deployment",
            "location": "New York, NY",
            "salary": "$140k-$190k",
            "category": "engineering"
        }
    ]
    
    count = manager.add_jobs_batch(sample_jobs)
    print(f"   ✅ Added {count} jobs")
    
    # Search
    print("\n3️⃣  Testing semantic search...")
    results = manager.search_jobs("Python AI developer", k=2)
    print(f"\n   Found {len(results)} results:")
    for job in results:
        print(f"   - {job['title']} at {job['company']} (similarity: {job['similarity']:.3f})")
    
    # Company analysis
    print("\n4️⃣  Analyzing company...")
    analysis = manager.analyze_company("AI Corp")
    print(f"\n   Company: {analysis['company']}")
    print(f"   Total jobs: {analysis['total_jobs']}")
    print(f"   Top skills: {analysis['top_skills']}")
    
    # Skill trends
    print("\n5️⃣  Skill trends...")
    trends = manager.get_skill_trends(5)
    print("\n   Top 5 skills:")
    for skill, count in trends:
        print(f"   - {skill}: {count}")
    
    # Statistics
    print("\n6️⃣  Corpus statistics...")
    stats = manager.get_stats()
    print(f"\n   Total jobs: {stats['total_jobs']}")
    print(f"   Total companies: {stats['total_companies']}")
    
    # Save
    print("\n7️⃣  Saving corpus...")
    manager.save_corpus()
    print("   ✅ Saved")
    
    # Cleanup
    manager.clear_corpus()
    
    print("\n" + "=" * 80)
    print("✅ Job Corpus Manager tests complete!")
    print("=" * 80)
