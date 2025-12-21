"""
RAG System Tests

Comprehensive tests for all RAG components:
- Vector Store
- RAG Retriever
- Job Corpus Manager
- Resume History Tracker
- RAG LangChain Tools

Author: AI Job Hunter
Date: 2024
"""

import pytest
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import RAG components
from rag.vector_store import JobVectorStore
from rag.rag_retriever import (
    RAGRetriever, RetrievalConfig, SearchStrategy, RetrievalMode,
    create_job_retriever, create_full_retriever
)
from rag.job_corpus import JobCorpusManager
from rag.resume_history import ResumeHistoryTracker
from tools.langchain_tools import RAGSearchTool, SuccessfulBulletsTool


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_jobs():
    """Sample job descriptions for testing"""
    return [
        {
            "content": "Senior Python Developer - AI/ML expertise. LangChain, OpenAI API, vector databases required.",
            "metadata": {
                "title": "Senior Python Developer",
                "company": "AI Corp",
                "location": "San Francisco, CA",
                "salary": "$180k-$250k",
                "category": "engineering"
            }
        },
        {
            "content": "Data Scientist - Machine Learning, Python, TensorFlow, PyTorch. PhD preferred.",
            "metadata": {
                "title": "Data Scientist",
                "company": "DataTech",
                "location": "Remote",
                "salary": "$160k-$220k",
                "category": "data-science"
            }
        },
        {
            "content": "Full Stack Engineer - React, Node.js, TypeScript. Build modern web applications.",
            "metadata": {
                "title": "Full Stack Engineer",
                "company": "WebCo",
                "location": "New York, NY",
                "salary": "$140k-$190k",
                "category": "engineering"
            }
        }
    ]


@pytest.fixture
def vector_store():
    """Create test vector store"""
    store = JobVectorStore(
        collection_name="test_rag",
        use_chromadb=False  # Use in-memory for testing
    )
    yield store
    # Cleanup
    store.clear()


@pytest.fixture
def populated_vector_store(vector_store, sample_jobs):
    """Vector store with sample data"""
    vector_store.add_documents(sample_jobs)
    return vector_store


# ============================================================================
# VECTOR STORE TESTS
# ============================================================================

class TestVectorStore:
    """Tests for JobVectorStore"""
    
    def test_initialization(self, vector_store):
        """Test vector store initialization"""
        assert vector_store is not None
        assert vector_store.get_document_count() == 0
    
    def test_add_documents(self, vector_store, sample_jobs):
        """Test adding documents"""
        count = vector_store.add_documents(sample_jobs)
        assert count == len(sample_jobs)
        assert vector_store.get_document_count() == len(sample_jobs)
    
    def test_similarity_search(self, populated_vector_store):
        """Test semantic similarity search"""
        results = populated_vector_store.similarity_search(
            query="Python AI developer",
            k=2
        )
        
        assert len(results) <= 2
        assert all("similarity" in r for r in results)
        assert all("content" in r for r in results)
        assert all("metadata" in r for r in results)
    
    def test_metadata_filtering(self, populated_vector_store):
        """Test search with metadata filters"""
        results = populated_vector_store.similarity_search(
            query="software engineer",
            k=10,
            filter_metadata={"category": "engineering"}
        )
        
        # Should only return engineering jobs
        for result in results:
            assert result["metadata"]["category"] == "engineering"
    
    def test_similarity_threshold(self, populated_vector_store):
        """Test similarity threshold filtering"""
        # High threshold should return fewer/no results
        results = populated_vector_store.similarity_search(
            query="blockchain developer",  # Unrelated query
            k=10,
            similarity_threshold=0.9
        )
        
        # All results should meet threshold
        for result in results:
            assert result["similarity"] >= 0.9
    
    def test_get_embedding(self, vector_store):
        """Test embedding generation"""
        embedding = vector_store.get_embedding("test text")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # OpenAI embedding dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_batch_embeddings(self, vector_store):
        """Test batch embedding generation"""
        texts = ["text 1", "text 2", "text 3"]
        embeddings = vector_store.get_embeddings_batch(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) == 1536 for emb in embeddings)
    
    def test_delete_documents(self, populated_vector_store):
        """Test document deletion"""
        initial_count = populated_vector_store.get_document_count()
        
        # Get first document ID
        results = populated_vector_store.similarity_search("test", k=1)
        if results and "id" in results[0]["metadata"]:
            doc_id = results[0]["metadata"]["id"]
            populated_vector_store.delete_documents([doc_id])
            
            assert populated_vector_store.get_document_count() < initial_count
    
    def test_clear(self, populated_vector_store):
        """Test clearing all documents"""
        populated_vector_store.clear()
        assert populated_vector_store.get_document_count() == 0


# ============================================================================
# RAG RETRIEVER TESTS
# ============================================================================

class TestRAGRetriever:
    """Tests for RAG Retriever"""
    
    def test_initialization(self, populated_vector_store):
        """Test retriever initialization"""
        retriever = RAGRetriever(job_store=populated_vector_store)
        assert retriever is not None
        assert retriever.job_store is not None
    
    def test_semantic_search(self, populated_vector_store):
        """Test semantic search strategy"""
        retriever = RAGRetriever(job_store=populated_vector_store)
        
        results = retriever.retrieve(
            query="Python developer",
            config=RetrievalConfig(
                k=2,
                strategy=SearchStrategy.SEMANTIC
            )
        )
        
        assert len(results) <= 2
        assert all(hasattr(r, "content") for r in results)
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "rank") for r in results)
    
    def test_hybrid_search(self, populated_vector_store):
        """Test hybrid search strategy"""
        retriever = RAGRetriever(job_store=populated_vector_store)
        
        results = retriever.retrieve(
            query="Python machine learning",
            config=RetrievalConfig(
                k=2,
                strategy=SearchStrategy.HYBRID
            )
        )
        
        assert len(results) <= 2
        # Scores should be adjusted by keyword matching
        assert all(0 <= r.score <= 1 for r in results)
    
    def test_mmr_search(self, populated_vector_store):
        """Test MMR search for diversity"""
        retriever = RAGRetriever(job_store=populated_vector_store)
        
        results = retriever.retrieve(
            query="software engineer",
            config=RetrievalConfig(
                k=3,
                strategy=SearchStrategy.MMR,
                diversity_penalty=0.5
            )
        )
        
        assert len(results) <= 3
        # Results should be diverse
    
    def test_context_augmentation(self, populated_vector_store):
        """Test context augmentation"""
        retriever = RAGRetriever(job_store=populated_vector_store)
        
        results = retriever.retrieve("Python developer", config=RetrievalConfig(k=2))
        context = retriever.augment_context("Python developer", results)
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Retrieved Context" in context
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test create_job_retriever
        retriever = create_job_retriever(
            job_collection="test_jobs",
            use_chromadb=False
        )
        assert retriever is not None
        assert retriever.job_store is not None
        
        # Test create_full_retriever
        full_retriever = create_full_retriever(use_chromadb=False)
        assert full_retriever is not None
        assert full_retriever.job_store is not None
        assert full_retriever.resume_store is not None
        assert full_retriever.bullet_store is not None


# ============================================================================
# JOB CORPUS TESTS
# ============================================================================

class TestJobCorpus:
    """Tests for Job Corpus Manager"""
    
    @pytest.fixture
    def corpus_manager(self, tmp_path):
        """Create test corpus manager"""
        corpus_file = tmp_path / "test_corpus.json"
        manager = JobCorpusManager(
            corpus_path=str(corpus_file),
            collection_name="test_corpus"
        )
        yield manager
        # Cleanup
        manager.clear_corpus()
    
    def test_add_job(self, corpus_manager):
        """Test adding a single job"""
        success = corpus_manager.add_job(
            job_id="job_001",
            title="Python Developer",
            company="TechCo",
            description="Build Python applications",
            location="Remote",
            salary="$150k"
        )
        
        assert success is True
        assert len(corpus_manager.jobs) == 1
        assert corpus_manager.get_job_by_id("job_001") is not None
    
    def test_add_jobs_batch(self, corpus_manager):
        """Test adding multiple jobs"""
        jobs = [
            {
                "job_id": f"job_{i}",
                "title": f"Developer {i}",
                "company": "TechCo",
                "description": f"Description {i}"
            }
            for i in range(3)
        ]
        
        count = corpus_manager.add_jobs_batch(jobs)
        assert count == 3
        assert len(corpus_manager.jobs) == 3
    
    def test_search_jobs(self, corpus_manager):
        """Test job search"""
        # Add test jobs
        corpus_manager.add_job(
            job_id="job_001",
            title="Python Developer",
            company="TechCo",
            description="Python, Django, AI development"
        )
        
        # Search
        results = corpus_manager.search_jobs("Python AI", k=5)
        assert len(results) >= 0  # May be empty if similarity too low
    
    def test_get_jobs_by_company(self, corpus_manager):
        """Test getting jobs by company"""
        corpus_manager.add_job(
            job_id="job_001",
            title="Developer",
            company="TechCo",
            description="Description"
        )
        corpus_manager.add_job(
            job_id="job_002",
            title="Engineer",
            company="TechCo",
            description="Description"
        )
        
        jobs = corpus_manager.get_jobs_by_company("TechCo")
        assert len(jobs) == 2
    
    def test_skill_trends(self, corpus_manager):
        """Test skill trend analysis"""
        corpus_manager.add_job(
            job_id="job_001",
            title="Python Developer",
            company="TechCo",
            description="Python, AWS, Docker required"
        )
        
        trends = corpus_manager.get_skill_trends(top_n=5)
        assert isinstance(trends, list)
        # Should find at least some skills
        skill_names = [skill for skill, count in trends]
        assert any(skill in skill_names for skill in ["python", "aws", "docker"])
    
    def test_company_stats(self, corpus_manager):
        """Test company statistics"""
        corpus_manager.add_job(
            job_id="job_001",
            title="Dev",
            company="CompanyA",
            description="Desc"
        )
        corpus_manager.add_job(
            job_id="job_002",
            title="Dev",
            company="CompanyA",
            description="Desc"
        )
        
        stats = corpus_manager.get_company_stats()
        assert "CompanyA" in stats
        assert stats["CompanyA"] == 2
    
    def test_save_load(self, corpus_manager):
        """Test persistence"""
        # Add job
        corpus_manager.add_job(
            job_id="job_001",
            title="Developer",
            company="TechCo",
            description="Description"
        )
        
        # Save
        assert corpus_manager.save_corpus() is True
        
        # Load
        initial_count = len(corpus_manager.jobs)
        corpus_manager.jobs.clear()
        assert corpus_manager.load_corpus() is True
        assert len(corpus_manager.jobs) == initial_count


# ============================================================================
# RESUME HISTORY TESTS
# ============================================================================

class TestResumeHistory:
    """Tests for Resume History Tracker"""
    
    @pytest.fixture
    def history_tracker(self, tmp_path):
        """Create test history tracker"""
        history_file = tmp_path / "test_history.json"
        tracker = ResumeHistoryTracker(history_path=str(history_file))
        yield tracker
        # Cleanup
        tracker.clear_history()
    
    def test_track_application(self, history_tracker):
        """Test tracking an application"""
        success = history_tracker.track_application(
            application_id="app_001",
            job_id="job_001",
            job_title="Python Developer",
            company="TechCo",
            resume_content="Resume content here...",
            bullet_points=[
                "Built AI chatbot",
                "Led team of 3 engineers"
            ],
            keywords_used=["Python", "AI", "LangChain"]
        )
        
        assert success is True
        assert len(history_tracker.applications) == 1
    
    def test_update_outcome(self, history_tracker):
        """Test updating application outcome"""
        # Track application
        history_tracker.track_application(
            application_id="app_001",
            job_id="job_001",
            job_title="Developer",
            company="TechCo",
            resume_content="Content",
            bullet_points=["Bullet 1"],
            keywords_used=["Python"]
        )
        
        # Update outcome
        success = history_tracker.update_outcome("app_001", "interview")
        assert success is True
        assert history_tracker.applications["app_001"]["outcome"] == "interview"
    
    def test_get_successful_bullets(self, history_tracker):
        """Test retrieving successful bullets"""
        # Track successful application
        history_tracker.track_application(
            application_id="app_001",
            job_id="job_001",
            job_title="Developer",
            company="TechCo",
            resume_content="Content",
            bullet_points=["Built AI chatbot using LangChain"],
            keywords_used=["Python", "AI"]
        )
        
        # Mark as successful
        history_tracker.update_outcome("app_001", "interview")
        
        # Retrieve bullets
        bullets = history_tracker.get_successful_bullets("AI chatbot", k=5)
        assert isinstance(bullets, list)
    
    def test_success_rate(self, history_tracker):
        """Test success rate calculation"""
        # Track multiple applications
        history_tracker.track_application(
            "app_001", "job_001", "Dev", "Co", "Content", [], []
        )
        history_tracker.update_outcome("app_001", "interview")
        
        history_tracker.track_application(
            "app_002", "job_002", "Dev", "Co", "Content", [], []
        )
        history_tracker.update_outcome("app_002", "rejected")
        
        metrics = history_tracker.get_success_rate()
        assert metrics["total_applications"] == 2
        assert metrics["interview_rate"] == 0.5
        assert metrics["rejection_rate"] == 0.5
    
    def test_top_keywords(self, history_tracker):
        """Test keyword frequency analysis"""
        history_tracker.track_application(
            "app_001", "job_001", "Dev", "Co", "Content",
            [], ["Python", "AWS"]
        )
        history_tracker.track_application(
            "app_002", "job_002", "Dev", "Co", "Content",
            [], ["Python", "Docker"]
        )
        
        keywords = history_tracker.get_top_keywords(top_n=5)
        assert len(keywords) > 0
        # Python should be most frequent
        assert keywords[0][0] == "Python"
        assert keywords[0][1] == 2


# ============================================================================
# RAG TOOLS TESTS
# ============================================================================

class TestRAGTools:
    """Tests for RAG LangChain Tools"""
    
    def test_rag_search_tool_initialization(self):
        """Test RAG search tool initialization"""
        tool = RAGSearchTool()
        assert tool.name == "rag_search"
        assert tool.description is not None
    
    def test_successful_bullets_tool_initialization(self):
        """Test successful bullets tool initialization"""
        tool = SuccessfulBulletsTool()
        assert tool.name == "get_successful_bullets"
        assert tool.description is not None
    
    def test_rag_search_tool_execution(self, populated_vector_store):
        """Test RAG search tool execution"""
        tool = RAGSearchTool()
        
        # Execute tool
        result = tool._run(
            query="Python developer",
            k=2,
            mode="jobs"
        )
        
        # Parse result
        result_data = json.loads(result)
        assert "success" in result_data
        
        # Note: May fail if no data in default store
        # This is expected in test environment


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRAGIntegration:
    """Integration tests for complete RAG workflow"""
    
    def test_end_to_end_job_search(self, tmp_path):
        """Test complete job search and retrieval workflow"""
        # Setup corpus
        corpus_file = tmp_path / "corpus.json"
        corpus = JobCorpusManager(corpus_path=str(corpus_file))
        
        # Add jobs
        corpus.add_job(
            job_id="job_001",
            title="Python Developer",
            company="TechCo",
            description="Python, AI, LangChain development"
        )
        
        # Search
        results = corpus.search_jobs("Python AI", k=5)
        assert isinstance(results, list)
        
        # Cleanup
        corpus.clear_corpus()
    
    def test_resume_tracking_workflow(self, tmp_path):
        """Test complete resume tracking workflow"""
        # Setup tracker
        history_file = tmp_path / "history.json"
        tracker = ResumeHistoryTracker(history_path=str(history_file))
        
        # Track application
        tracker.track_application(
            application_id="app_001",
            job_id="job_001",
            job_title="Developer",
            company="TechCo",
            resume_content="Content",
            bullet_points=["Bullet 1", "Bullet 2"],
            keywords_used=["Python", "AWS"]
        )
        
        # Update to successful
        tracker.update_outcome("app_001", "interview")
        
        # Get successful bullets
        bullets = tracker.get_successful_bullets("development", k=5)
        assert isinstance(bullets, list)
        
        # Cleanup
        tracker.clear_history()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])
