"""
RAG Retriever Module

Advanced retrieval system for job descriptions and resume data.
Features context augmentation, re-ranking, and hybrid search.

Author: AI Job Hunter
Date: 2024
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import JobVectorStore
from utils.logger import logger


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class SearchStrategy(Enum):
    """Search strategy types"""
    SEMANTIC = "semantic"  # Pure vector similarity
    HYBRID = "hybrid"      # Keyword + semantic
    MMR = "mmr"           # Maximal Marginal Relevance (diversity)
    RERANK = "rerank"     # LLM-based re-ranking


class RetrievalMode(Enum):
    """Retrieval modes"""
    JOBS = "jobs"                    # Job descriptions
    RESUMES = "resumes"              # Resume history
    SUCCESSFUL_BULLETS = "bullets"   # Proven bullet points
    MIXED = "mixed"                  # All sources


@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""
    k: int = 5                          # Number of results
    similarity_threshold: float = 0.7    # Minimum similarity
    strategy: SearchStrategy = SearchStrategy.SEMANTIC
    rerank: bool = False                 # Enable LLM re-ranking
    diversity_penalty: float = 0.5       # For MMR (0=relevance, 1=diversity)
    expand_query: bool = False           # Query expansion
    filter_metadata: Optional[Dict] = None  # Metadata filtering


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int
    source: str  # 'job', 'resume', 'bullet'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "rank": self.rank,
            "source": self.source
        }


# ============================================================================
# RAG RETRIEVER
# ============================================================================

class RAGRetriever:
    """
    Advanced RAG retriever with multiple search strategies.
    
    Features:
    - Semantic similarity search
    - Hybrid keyword + vector search
    - MMR for diverse results
    - LLM-based re-ranking
    - Query expansion
    - Multi-source retrieval
    """
    
    def __init__(
        self,
        job_store: Optional[JobVectorStore] = None,
        resume_store: Optional[JobVectorStore] = None,
        bullet_store: Optional[JobVectorStore] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize RAG retriever.
        
        Args:
            job_store: Vector store for job descriptions
            resume_store: Vector store for resume history
            bullet_store: Vector store for successful bullet points
            model: LLM model for re-ranking and query expansion
        """
        self.job_store = job_store
        self.resume_store = resume_store
        self.bullet_store = bullet_store
        
        # LLM for re-ranking and query expansion
        self.llm = ChatOpenAI(model=model, temperature=0.0)
        
        # Embeddings for keyword matching
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        logger.info("RAG Retriever initialized")
    
    # ========================================================================
    # MAIN RETRIEVAL
    # ========================================================================
    
    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        mode: RetrievalMode = RetrievalMode.JOBS
    ) -> List[RetrievalResult]:
        """
        Main retrieval method.
        
        Args:
            query: Search query
            config: Retrieval configuration
            mode: Retrieval mode (jobs, resumes, bullets, mixed)
            
        Returns:
            List of retrieval results, ranked by relevance
        """
        config = config or RetrievalConfig()
        
        logger.info(f"Retrieving with query: '{query}' | Mode: {mode.value} | Strategy: {config.strategy.value}")
        
        # Query expansion
        if config.expand_query:
            query = self._expand_query(query)
            logger.info(f"Expanded query: '{query}'")
        
        # Multi-source retrieval
        if mode == RetrievalMode.MIXED:
            results = self._retrieve_mixed(query, config)
        else:
            results = self._retrieve_single_source(query, config, mode)
        
        # Re-ranking
        if config.rerank and len(results) > 1:
            results = self._rerank_results(query, results)
        
        # Add rank numbers
        for i, result in enumerate(results, 1):
            result.rank = i
        
        logger.info(f"Retrieved {len(results)} results")
        return results
    
    def _retrieve_single_source(
        self,
        query: str,
        config: RetrievalConfig,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Retrieve from single source"""
        # Select store
        if mode == RetrievalMode.JOBS:
            store = self.job_store
            source = "job"
        elif mode == RetrievalMode.RESUMES:
            store = self.resume_store
            source = "resume"
        elif mode == RetrievalMode.SUCCESSFUL_BULLETS:
            store = self.bullet_store
            source = "bullet"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        if not store:
            logger.warning(f"No store available for mode: {mode.value}")
            return []
        
        # Search strategy
        if config.strategy == SearchStrategy.SEMANTIC:
            docs = self._semantic_search(store, query, config)
        elif config.strategy == SearchStrategy.HYBRID:
            docs = self._hybrid_search(store, query, config)
        elif config.strategy == SearchStrategy.MMR:
            docs = self._mmr_search(store, query, config)
        else:
            docs = self._semantic_search(store, query, config)
        
        # Convert to retrieval results
        results = [
            RetrievalResult(
                content=doc["content"],
                metadata=doc["metadata"],
                score=doc["similarity"],
                rank=0,  # Will be set later
                source=source
            )
            for doc in docs
        ]
        
        return results
    
    def _retrieve_mixed(
        self,
        query: str,
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """Retrieve from multiple sources and merge"""
        all_results = []
        
        # Retrieve from each source
        modes = [RetrievalMode.JOBS, RetrievalMode.RESUMES, RetrievalMode.SUCCESSFUL_BULLETS]
        for mode in modes:
            try:
                results = self._retrieve_single_source(query, config, mode)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to retrieve from {mode.value}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to k results
        return all_results[:config.k]
    
    # ========================================================================
    # SEARCH STRATEGIES
    # ========================================================================
    
    def _semantic_search(
        self,
        store: JobVectorStore,
        query: str,
        config: RetrievalConfig
    ) -> List[Dict]:
        """Pure semantic similarity search"""
        return store.similarity_search(
            query=query,
            k=config.k,
            filter_metadata=config.filter_metadata,
            similarity_threshold=config.similarity_threshold
        )
    
    def _hybrid_search(
        self,
        store: JobVectorStore,
        query: str,
        config: RetrievalConfig
    ) -> List[Dict]:
        """
        Hybrid keyword + semantic search.
        
        Combines BM25-style keyword matching with vector similarity.
        """
        # Get semantic results
        semantic_results = self._semantic_search(store, query, config)
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Re-score based on keyword overlap
        for doc in semantic_results:
            keyword_score = self._keyword_match_score(doc["content"], keywords)
            # Weighted combination: 70% semantic, 30% keyword
            doc["similarity"] = 0.7 * doc["similarity"] + 0.3 * keyword_score
        
        # Re-sort by new score
        semantic_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return semantic_results
    
    def _mmr_search(
        self,
        store: JobVectorStore,
        query: str,
        config: RetrievalConfig
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance search.
        
        Balances relevance with diversity to avoid redundant results.
        """
        # Get more candidates than needed
        candidates = store.similarity_search(
            query=query,
            k=config.k * 3,
            filter_metadata=config.filter_metadata,
            similarity_threshold=config.similarity_threshold
        )
        
        if not candidates:
            return []
        
        # MMR algorithm
        selected = []
        selected_embeddings = []
        query_embedding = store.get_embedding(query)
        
        while len(selected) < config.k and len(candidates) > 0:
            best_score = -1
            best_idx = -1
            
            for i, doc in enumerate(candidates):
                # Relevance to query
                relevance = doc["similarity"]
                
                # Diversity penalty (similarity to already selected)
                if selected_embeddings:
                    doc_embedding = store.get_embedding(doc["content"])
                    max_similarity = max(
                        store.cosine_similarity(doc_embedding, sel_emb)
                        for sel_emb in selected_embeddings
                    )
                    diversity = 1 - max_similarity
                else:
                    diversity = 1.0
                
                # MMR score
                mmr_score = (
                    config.diversity_penalty * relevance +
                    (1 - config.diversity_penalty) * diversity
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            # Add best candidate
            if best_idx >= 0:
                best_doc = candidates.pop(best_idx)
                selected.append(best_doc)
                selected_embeddings.append(store.get_embedding(best_doc["content"]))
        
        return selected
    
    # ========================================================================
    # RE-RANKING
    # ========================================================================
    
    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        LLM-based re-ranking of results.
        
        Uses GPT to assess true relevance beyond vector similarity.
        """
        logger.info(f"Re-ranking {len(results)} results...")
        
        # Create re-ranking prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at ranking job search results.
Given a query and a list of job descriptions, rank them by relevance.

Return ONLY a JSON list of indices (0-indexed) in order of relevance.
Example: [2, 0, 4, 1, 3]"""),
            ("user", """Query: {query}

Results:
{results}

Ranking (JSON list of indices):""")
        ])
        
        # Format results
        results_text = "\n\n".join([
            f"[{i}] {result.metadata.get('title', 'N/A')}\n{result.content[:300]}..."
            for i, result in enumerate(results)
        ])
        
        # Get re-ranking
        chain = prompt | self.llm | StrOutputParser()
        try:
            ranking_str = chain.invoke({
                "query": query,
                "results": results_text
            })
            
            # Parse ranking
            ranking = json.loads(ranking_str.strip())
            
            # Re-order results
            reranked = [results[i] for i in ranking if i < len(results)]
            
            # Update scores based on new ranking
            for i, result in enumerate(reranked):
                result.score = 1.0 - (i / len(reranked))
            
            logger.info(f"Re-ranking successful")
            return reranked
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}, returning original order")
            return results
    
    # ========================================================================
    # QUERY EXPANSION
    # ========================================================================
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms using LLM.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query expansion expert.
Given a job search query, expand it with relevant synonyms and related terms.

Keep the expansion concise and focused. Return only the expanded query text."""),
            ("user", "Original query: {query}\n\nExpanded query:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            expanded = chain.invoke({"query": query})
            return expanded.strip()
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can'
        }
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords
    
    def _keyword_match_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword match score"""
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches / len(keywords) if keywords else 0.0
    
    # ========================================================================
    # CONTEXT AUGMENTATION
    # ========================================================================
    
    def augment_context(
        self,
        query: str,
        results: List[RetrievalResult],
        max_tokens: int = 2000
    ) -> str:
        """
        Augment prompt with retrieved context.
        
        Formats results into a context string for LLM consumption.
        
        Args:
            query: Original query
            results: Retrieved results
            max_tokens: Maximum context length (approximate)
            
        Returns:
            Formatted context string
        """
        context_parts = [
            "# Retrieved Context",
            f"\nQuery: {query}",
            f"\nFound {len(results)} relevant results:\n"
        ]
        
        current_length = len(" ".join(context_parts))
        
        for i, result in enumerate(results, 1):
            # Format result
            result_text = f"\n## Result {i} (Score: {result.score:.3f})\n"
            result_text += f"**Source:** {result.source}\n"
            
            # Add metadata
            if result.metadata:
                result_text += f"**Metadata:** {json.dumps(result.metadata, indent=2)}\n"
            
            result_text += f"\n{result.content}\n"
            result_text += "-" * 80 + "\n"
            
            # Check length
            if current_length + len(result_text) > max_tokens * 4:  # ~4 chars per token
                break
            
            context_parts.append(result_text)
            current_length += len(result_text)
        
        return "\n".join(context_parts)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_job_retriever(
    job_collection: str = "jobs",
    use_chromadb: bool = True
) -> RAGRetriever:
    """
    Create a retriever for job descriptions.
    
    Args:
        job_collection: ChromaDB collection name
        use_chromadb: Use ChromaDB or in-memory
        
    Returns:
        Configured RAG retriever
    """
    job_store = JobVectorStore(
        collection_name=job_collection,
        use_chromadb=use_chromadb
    )
    
    return RAGRetriever(job_store=job_store)


def create_full_retriever(
    job_collection: str = "jobs",
    resume_collection: str = "resumes",
    bullet_collection: str = "bullets",
    use_chromadb: bool = True
) -> RAGRetriever:
    """
    Create a retriever with all sources.
    
    Args:
        job_collection: Job descriptions collection
        resume_collection: Resume history collection
        bullet_collection: Successful bullets collection
        use_chromadb: Use ChromaDB or in-memory
        
    Returns:
        Configured RAG retriever with all sources
    """
    job_store = JobVectorStore(collection_name=job_collection, use_chromadb=use_chromadb)
    resume_store = JobVectorStore(collection_name=resume_collection, use_chromadb=use_chromadb)
    bullet_store = JobVectorStore(collection_name=bullet_collection, use_chromadb=use_chromadb)
    
    return RAGRetriever(
        job_store=job_store,
        resume_store=resume_store,
        bullet_store=bullet_store
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 80)
    print("RAG RETRIEVER TEST")
    print("=" * 80)
    
    # Create sample job store
    print("\n1️⃣  Setting up test data...")
    job_store = JobVectorStore(collection_name="test_rag_jobs", use_chromadb=False)
    
    sample_jobs = [
        {
            "content": "Senior Python Developer with AI/ML expertise. LangChain, OpenAI API, vector databases.",
            "metadata": {"title": "Senior Python Developer", "company": "AI Corp"}
        },
        {
            "content": "Full Stack Engineer - React, Node.js, TypeScript. Build modern web apps.",
            "metadata": {"title": "Full Stack Engineer", "company": "WebCo"}
        },
        {
            "content": "Data Scientist - Machine Learning, Python, TensorFlow, PyTorch.",
            "metadata": {"title": "Data Scientist", "company": "DataTech"}
        }
    ]
    
    job_store.add_documents(sample_jobs)
    print(f"   ✅ Added {len(sample_jobs)} jobs to store")
    
    # Create retriever
    print("\n2️⃣  Creating retriever...")
    retriever = RAGRetriever(job_store=job_store)
    print("   ✅ Retriever initialized")
    
    # Test semantic search
    print("\n3️⃣  Testing semantic search...")
    results = retriever.retrieve(
        query="Python AI developer",
        config=RetrievalConfig(k=2, strategy=SearchStrategy.SEMANTIC)
    )
    print(f"\n   Found {len(results)} results:")
    for r in results:
        print(f"   {r.rank}. {r.metadata['title']} (score: {r.score:.3f})")
    
    # Test with re-ranking
    print("\n4️⃣  Testing with LLM re-ranking...")
    results_reranked = retriever.retrieve(
        query="Python AI developer",
        config=RetrievalConfig(k=3, rerank=True)
    )
    print(f"\n   Re-ranked {len(results_reranked)} results:")
    for r in results_reranked:
        print(f"   {r.rank}. {r.metadata['title']} (score: {r.score:.3f})")
    
    # Test context augmentation
    print("\n5️⃣  Testing context augmentation...")
    context = retriever.augment_context("Python developer", results)
    print(f"\n   Context preview (first 300 chars):")
    print(f"   {context[:300]}...")
    
    print("\n" + "=" * 80)
    print("✅ RAG Retriever tests complete!")
    print("=" * 80)
