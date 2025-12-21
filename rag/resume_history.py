"""
Resume History Tracker

Tracks successful resume applications and extracts proven patterns.
Features application tracking, successful phrase extraction, and A/B testing.

Author: AI Job Hunter
Date: 2024
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter

from rag.vector_store import JobVectorStore
from utils.logger import logger


# ============================================================================
# RESUME HISTORY TRACKER
# ============================================================================

class ResumeHistoryTracker:
    """
    Tracks resume application history and success patterns.
    
    Features:
    - Store resume versions
    - Track application outcomes
    - Extract successful bullet points
    - Analyze which keywords worked
    - A/B testing data collection
    """
    
    def __init__(
        self,
        history_path: str = "data/resume_history.json",
        bullet_store: Optional[JobVectorStore] = None,
        resume_store: Optional[JobVectorStore] = None
    ):
        """
        Initialize resume history tracker.
        
        Args:
            history_path: Path to save history metadata
            bullet_store: Vector store for successful bullet points
            resume_store: Vector store for resume versions
        """
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Vector stores
        self.bullet_store = bullet_store or JobVectorStore(
            collection_name="successful_bullets",
            use_chromadb=True
        )
        
        self.resume_store = resume_store or JobVectorStore(
            collection_name="resume_history",
            use_chromadb=True
        )
        
        # Metadata storage
        self.applications: Dict[str, Dict] = {}  # application_id -> data
        self.load_history()
        
        logger.info(f"Resume History Tracker initialized with {len(self.applications)} applications")
    
    # ========================================================================
    # TRACKING APPLICATIONS
    # ========================================================================
    
    def track_application(
        self,
        application_id: str,
        job_id: str,
        job_title: str,
        company: str,
        resume_content: str,
        bullet_points: List[str],
        keywords_used: List[str],
        **metadata
    ) -> bool:
        """
        Track a resume application.
        
        Args:
            application_id: Unique application identifier
            job_id: Job posting ID
            job_title: Job title
            company: Company name
            resume_content: Full resume content
            bullet_points: List of bullet points used
            keywords_used: List of keywords/skills mentioned
            **metadata: Additional metadata
            
        Returns:
            True if tracked successfully
        """
        try:
            # Create application record
            application_data = {
                "application_id": application_id,
                "job_id": job_id,
                "job_title": job_title,
                "company": company,
                "resume_content": resume_content,
                "bullet_points": bullet_points,
                "keywords_used": keywords_used,
                "applied_at": datetime.now().isoformat(),
                "outcome": "pending",  # pending, interview, rejected, offer
                **metadata
            }
            
            # Add to metadata storage
            self.applications[application_id] = application_data
            
            # Add resume to vector store
            self.resume_store.add_documents([{
                "content": resume_content,
                "metadata": {
                    "application_id": application_id,
                    "job_id": job_id,
                    "job_title": job_title,
                    "company": company,
                    "applied_at": application_data["applied_at"]
                }
            }])
            
            logger.info(f"Tracked application: {application_id} for {job_title} at {company}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track application {application_id}: {e}")
            return False
    
    def update_outcome(
        self,
        application_id: str,
        outcome: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Update application outcome.
        
        Args:
            application_id: Application ID
            outcome: Outcome status (interview, rejected, offer)
            notes: Optional notes
            
        Returns:
            True if updated successfully
        """
        if application_id not in self.applications:
            logger.error(f"Application {application_id} not found")
            return False
        
        try:
            self.applications[application_id]["outcome"] = outcome
            self.applications[application_id]["outcome_updated_at"] = datetime.now().isoformat()
            
            if notes:
                self.applications[application_id]["outcome_notes"] = notes
            
            # If successful, add bullets to successful bullet store
            if outcome in ["interview", "offer"]:
                self._add_successful_bullets(application_id)
            
            logger.info(f"Updated application {application_id} outcome to: {outcome}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update outcome: {e}")
            return False
    
    def _add_successful_bullets(self, application_id: str):
        """Add bullet points from successful application to store"""
        try:
            app = self.applications[application_id]
            
            # Create documents for each bullet point
            bullet_docs = []
            for i, bullet in enumerate(app["bullet_points"]):
                bullet_docs.append({
                    "content": bullet,
                    "metadata": {
                        "application_id": application_id,
                        "job_id": app["job_id"],
                        "job_title": app["job_title"],
                        "company": app["company"],
                        "outcome": app["outcome"],
                        "bullet_index": i
                    }
                })
            
            self.bullet_store.add_documents(bullet_docs)
            logger.info(f"Added {len(bullet_docs)} successful bullets from {application_id}")
            
        except Exception as e:
            logger.error(f"Failed to add successful bullets: {e}")
    
    # ========================================================================
    # RETRIEVING SUCCESSFUL PATTERNS
    # ========================================================================
    
    def get_successful_bullets(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.7,
        outcome_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Retrieve successful bullet points similar to query.
        
        Args:
            query: Search query
            k: Number of results
            min_similarity: Minimum similarity threshold
            outcome_filter: Filter by outcome (e.g., ["interview", "offer"])
            
        Returns:
            List of similar successful bullets
        """
        # Build metadata filter
        filter_metadata = {}
        if outcome_filter:
            filter_metadata["outcome"] = outcome_filter[0]  # Simple filter for now
        
        results = self.bullet_store.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata if filter_metadata else None,
            similarity_threshold=min_similarity
        )
        
        return results
    
    def get_successful_resumes(
        self,
        query: str,
        k: int = 3,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """
        Retrieve successful resume versions similar to query.
        
        Args:
            query: Search query (job description or keywords)
            k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar successful resumes
        """
        results = self.resume_store.similarity_search(
            query=query,
            k=k,
            similarity_threshold=min_similarity
        )
        
        # Enrich with application data
        enriched = []
        for result in results:
            app_id = result["metadata"].get("application_id")
            if app_id and app_id in self.applications:
                app = self.applications[app_id]
                if app["outcome"] in ["interview", "offer"]:
                    enriched.append({
                        **result,
                        "outcome": app["outcome"],
                        "keywords_used": app.get("keywords_used", [])
                    })
        
        return enriched
    
    def get_top_keywords(
        self,
        outcome: Optional[str] = None,
        top_n: int = 20
    ) -> List[tuple]:
        """
        Get most frequently used keywords in applications.
        
        Args:
            outcome: Filter by outcome (interview, offer, rejected, pending)
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, count) tuples
        """
        keyword_counter = Counter()
        
        for app in self.applications.values():
            # Filter by outcome if specified
            if outcome and app["outcome"] != outcome:
                continue
            
            # Count keywords
            for keyword in app.get("keywords_used", []):
                keyword_counter[keyword] += 1
        
        return keyword_counter.most_common(top_n)
    
    # ========================================================================
    # ANALYTICS
    # ========================================================================
    
    def get_success_rate(self) -> Dict[str, float]:
        """
        Calculate success rates.
        
        Returns:
            Dictionary with success metrics
        """
        total = len(self.applications)
        if total == 0:
            return {"error": "No applications tracked"}
        
        outcomes = Counter(app["outcome"] for app in self.applications.values())
        
        return {
            "total_applications": total,
            "interview_rate": outcomes["interview"] / total,
            "offer_rate": outcomes["offer"] / total,
            "rejection_rate": outcomes["rejected"] / total,
            "pending_rate": outcomes["pending"] / total,
            "outcomes_breakdown": dict(outcomes)
        }
    
    def get_company_success_rate(self, company: str) -> Dict[str, Any]:
        """
        Get success rate for a specific company.
        
        Args:
            company: Company name
            
        Returns:
            Success metrics for the company
        """
        company_apps = [
            app for app in self.applications.values()
            if app["company"].lower() == company.lower()
        ]
        
        if not company_apps:
            return {"error": f"No applications for {company}"}
        
        total = len(company_apps)
        outcomes = Counter(app["outcome"] for app in company_apps)
        
        return {
            "company": company,
            "total_applications": total,
            "interview_count": outcomes["interview"],
            "offer_count": outcomes["offer"],
            "rejection_count": outcomes["rejected"],
            "interview_rate": outcomes["interview"] / total,
            "offer_rate": outcomes["offer"] / total
        }
    
    def get_keyword_effectiveness(self) -> Dict[str, Dict]:
        """
        Analyze which keywords lead to success.
        
        Returns:
            Dictionary of keyword -> success metrics
        """
        keyword_stats = {}
        
        # Collect all keywords
        all_keywords = set()
        for app in self.applications.values():
            all_keywords.update(app.get("keywords_used", []))
        
        # Analyze each keyword
        for keyword in all_keywords:
            apps_with_keyword = [
                app for app in self.applications.values()
                if keyword in app.get("keywords_used", [])
            ]
            
            if apps_with_keyword:
                total = len(apps_with_keyword)
                outcomes = Counter(app["outcome"] for app in apps_with_keyword)
                
                keyword_stats[keyword] = {
                    "total_uses": total,
                    "interview_count": outcomes["interview"],
                    "offer_count": outcomes["offer"],
                    "success_rate": (outcomes["interview"] + outcomes["offer"]) / total
                }
        
        # Sort by success rate
        sorted_keywords = sorted(
            keyword_stats.items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True
        )
        
        return dict(sorted_keywords[:20])  # Top 20
    
    def get_application_timeline(self) -> List[Dict]:
        """
        Get applications sorted by date.
        
        Returns:
            List of applications with dates
        """
        apps = [
            {
                "application_id": app_id,
                "applied_at": app["applied_at"],
                "job_title": app["job_title"],
                "company": app["company"],
                "outcome": app["outcome"]
            }
            for app_id, app in self.applications.items()
        ]
        
        return sorted(apps, key=lambda x: x["applied_at"], reverse=True)
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save_history(self) -> bool:
        """Save history to disk"""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.applications, f, indent=2)
            
            logger.info(f"Saved {len(self.applications)} applications to {self.history_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            return False
    
    def load_history(self) -> bool:
        """Load history from disk"""
        try:
            if self.history_path.exists():
                with open(self.history_path, 'r') as f:
                    self.applications = json.load(f)
                
                logger.info(f"Loaded {len(self.applications)} applications from {self.history_path}")
                return True
            else:
                logger.info("No existing history found, starting fresh")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            self.applications = {}
            return False
    
    def clear_history(self):
        """Clear all history"""
        self.applications.clear()
        self.bullet_store.clear()
        self.resume_store.clear()
        logger.info("History cleared")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 80)
    print("RESUME HISTORY TRACKER TEST")
    print("=" * 80)
    
    # Initialize tracker
    print("\n1️⃣  Initializing tracker...")
    tracker = ResumeHistoryTracker(history_path="data/test_resume_history.json")
    print("   ✅ Tracker initialized")
    
    # Track applications
    print("\n2️⃣  Tracking sample applications...")
    
    tracker.track_application(
        application_id="app_001",
        job_id="job_001",
        job_title="Senior Python Developer",
        company="AI Corp",
        resume_content="Senior Python Developer with 5 years experience in AI/ML...",
        bullet_points=[
            "Developed AI chatbot using LangChain and GPT-4, reducing response time by 50%",
            "Built RAG system with ChromaDB, improving retrieval accuracy by 30%",
            "Led team of 3 engineers in microservices migration to AWS"
        ],
        keywords_used=["Python", "LangChain", "GPT-4", "ChromaDB", "AWS"]
    )
    
    tracker.track_application(
        application_id="app_002",
        job_id="job_002",
        job_title="Data Scientist",
        company="DataTech",
        resume_content="Data Scientist with expertise in machine learning...",
        bullet_points=[
            "Implemented ML pipeline reducing model training time by 60%",
            "Deployed 10+ models to production using MLOps best practices"
        ],
        keywords_used=["Python", "TensorFlow", "MLOps", "AWS"]
    )
    
    print("   ✅ Tracked 2 applications")
    
    # Update outcomes
    print("\n3️⃣  Updating outcomes...")
    tracker.update_outcome("app_001", "interview", "Phone screen scheduled")
    tracker.update_outcome("app_002", "rejected", "Not enough ML experience")
    print("   ✅ Outcomes updated")
    
    # Get successful bullets
    print("\n4️⃣  Retrieving successful bullets...")
    bullets = tracker.get_successful_bullets("AI chatbot development", k=2)
    print(f"\n   Found {len(bullets)} similar bullets:")
    for bullet in bullets:
        print(f"   - {bullet['content'][:80]}...")
    
    # Success rate
    print("\n5️⃣  Success metrics...")
    metrics = tracker.get_success_rate()
    print(f"\n   Total applications: {metrics['total_applications']}")
    print(f"   Interview rate: {metrics['interview_rate']:.1%}")
    
    # Top keywords
    print("\n6️⃣  Top keywords...")
    keywords = tracker.get_top_keywords(top_n=5)
    print("\n   Most used keywords:")
    for kw, count in keywords:
        print(f"   - {kw}: {count}")
    
    # Save
    print("\n7️⃣  Saving history...")
    tracker.save_history()
    print("   ✅ Saved")
    
    # Cleanup
    tracker.clear_history()
    
    print("\n" + "=" * 80)
    print("✅ Resume History Tracker tests complete!")
    print("=" * 80)
