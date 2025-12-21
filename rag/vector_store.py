"""
Vector Store for Job Hunter RAG System

This module provides a hybrid vector store implementation supporting both:
1. ChromaDB - Persistent vector database (recommended for production)
2. In-memory - Simple NumPy-based store (for testing/development)

Features:
- OpenAI embeddings (text-embedding-3-small)
- Metadata filtering
- Batch operations
- Similarity thresholds
- Persistence to disk
- Hybrid search (coming soon)

Author: Job Hunter Team
Date: December 21, 2025
"""

from openai import OpenAI
from typing import List, Dict, Tuple, Optional, Any
import os
import json
import numpy as np
from datetime import datetime
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Using in-memory vector store.")
    print("   Install with: pip install chromadb")

from utils.logger import logger


class JobVectorStore:
    """
    Vector store for job descriptions and resumes using OpenAI embeddings.
    
    Supports both ChromaDB (persistent) and in-memory storage.
    """
    
    def __init__(
        self,
        collection_name: str = "job_hunter",
        persist_directory: str = "./data/vectorstore",
        use_chromadb: bool = True,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            use_chromadb: Use ChromaDB if available (falls back to in-memory)
            embedding_model: OpenAI embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize vector store
        self.use_chromadb = use_chromadb and CHROMADB_AVAILABLE
        
        if self.use_chromadb:
            self._init_chromadb()
        else:
            self._init_memory_store()
        
        logger.info(f"Vector store initialized: {'ChromaDB' if self.use_chromadb else 'In-Memory'}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Job descriptions and resume data"}
            )
            
            logger.info(f"ChromaDB collection '{self.collection_name}' ready")
            logger.info(f"Current document count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            logger.warning("Falling back to in-memory store")
            self.use_chromadb = False
            self._init_memory_store()
    
    def _init_memory_store(self):
        """Initialize in-memory vector store."""
        self.documents = []  # List of dicts: {"id": str, "content": str, "metadata": dict, "embedding": list}
        self.persist_file = os.path.join(self.persist_directory, "jobs.json")
        logger.info("In-memory vector store initialized")
        
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (1536 dimensions for text-embedding-3-small)
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise
        
        return all_embeddings
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to vector store with batch embedding generation.
        
        Args:
            documents: List of dicts with 'content' and optional 'metadata' keys
                      e.g., [{"content": "text", "metadata": {"title": "..."}, "id": "optional"}]
            batch_size: Batch size for embedding generation
            
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Extract texts and generate embeddings in batches
        texts = [doc["content"] for doc in documents]
        embeddings = self.get_embeddings_batch(texts, batch_size=batch_size)
        
        # Add to appropriate store
        if self.use_chromadb:
            return self._add_to_chromadb(documents, embeddings)
        else:
            return self._add_to_memory(documents, embeddings)
    
    def _add_to_chromadb(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """Add documents to ChromaDB collection."""
        ids = []
        contents = []
        metadatas = []
        
        for doc, embedding in zip(documents, embeddings):
            doc_id = doc.get("id", str(uuid.uuid4()))
            ids.append(doc_id)
            contents.append(doc["content"])
            
            # Prepare metadata (must be simple types for ChromaDB)
            metadata = doc.get("metadata", {}).copy()
            metadata["added_at"] = datetime.now().isoformat()
            metadatas.append(metadata)
        
        try:
            self.collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} documents to ChromaDB")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    def _add_to_memory(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """Add documents to in-memory store."""
        for doc, embedding in zip(documents, embeddings):
            doc_id = doc.get("id", str(uuid.uuid4()))
            self.documents.append({
                "id": doc_id,
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "embedding": embedding,
                "added_at": datetime.now().isoformat()
            })
        
        logger.info(f"Added {len(documents)} documents to memory store")
        return len(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"company": "Google"})
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of documents with similarity scores, sorted by relevance
        """
        if self.use_chromadb:
            return self._search_chromadb(query, k, filter_metadata, similarity_threshold)
        else:
            return self._search_memory(query, k, filter_metadata, similarity_threshold)
    
    def _search_chromadb(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search using ChromaDB."""
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Prepare where clause for metadata filtering
        where = filter_metadata if filter_metadata else None
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where
            )
            
            # Format results
            documents = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # ChromaDB returns distance, convert to similarity
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Assuming cosine distance
                    
                    if similarity >= similarity_threshold:
                        documents.append({
                            "id": results['ids'][0][i],
                            "content": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "similarity": float(similarity)
                        })
            
            logger.info(f"Found {len(documents)} documents (threshold: {similarity_threshold})")
            return documents
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    def _search_memory(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search using in-memory store."""
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc in self.documents:
            # Apply metadata filter if provided
            if filter_metadata:
                match = all(
                    doc["metadata"].get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue
            
            similarity = self.cosine_similarity(query_embedding, doc["embedding"])
            
            if similarity >= similarity_threshold:
                similarities.append((doc, similarity))
        
        # Sort by similarity (highest first) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, similarity in similarities[:k]:
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity": float(similarity)
            })
        
        logger.info(f"Found {len(results)} documents (threshold: {similarity_threshold})")
        return results
    
    def delete_documents(self, ids: List[str]) -> int:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if self.use_chromadb:
            try:
                self.collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} documents from ChromaDB")
                return len(ids)
            except Exception as e:
                logger.error(f"Failed to delete documents: {e}")
                return 0
        else:
            initial_count = len(self.documents)
            self.documents = [doc for doc in self.documents if doc["id"] not in ids]
            deleted_count = initial_count - len(self.documents)
            logger.info(f"Deleted {deleted_count} documents from memory store")
            return deleted_count
    
    def get_document_count(self) -> int:
        """Get total number of documents in store."""
        if self.use_chromadb:
            return self.collection.count()
        else:
            return len(self.documents)
    
    def clear(self):
        """Clear all documents from vector store."""
        if self.use_chromadb:
            # Delete collection and recreate
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Job descriptions and resume data"}
            )
            logger.info("ChromaDB collection cleared")
        else:
            self.documents = []
            logger.info("Memory store cleared")
    
    
    def save(self):
        """
        Save vector store to disk.
        
        ChromaDB automatically persists, this is for in-memory store.
        """
        if not self.use_chromadb:
            os.makedirs(os.path.dirname(self.persist_file), exist_ok=True)
            
            # Don't save embeddings (too large), they'll be regenerated on load
            save_data = [
                {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "added_at": doc.get("added_at", "")
                }
                for doc in self.documents
            ]
            
            with open(self.persist_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"Saved {len(save_data)} documents to {self.persist_file}")
    
    def load(self) -> bool:
        """
        Load vector store from disk.
        
        ChromaDB auto-loads, this is for in-memory store.
        
        Returns:
            True if loaded successfully
        """
        if self.use_chromadb:
            # ChromaDB persists automatically
            count = self.collection.count()
            logger.info(f"Loaded ChromaDB collection with {count} documents")
            return count > 0
        else:
            if os.path.exists(self.persist_file):
                with open(self.persist_file, 'r') as f:
                    data = json.load(f)
                
                # Regenerate embeddings
                logger.info(f"Loading {len(data)} documents and regenerating embeddings...")
                texts = [doc["content"] for doc in data]
                embeddings = self.get_embeddings_batch(texts)
                
                self.documents = []
                for doc, embedding in zip(data, embeddings):
                    self.documents.append({
                        "id": doc["id"],
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "embedding": embedding,
                        "added_at": doc.get("added_at", "")
                    })
                
                logger.info(f"Loaded {len(self.documents)} documents from {self.persist_file}")
                return True
            
            logger.warning(f"No saved data found at {self.persist_file}")
            return False


# ============================================================================
# TESTING & EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 80)
    print("JOB VECTOR STORE TEST")
    print("=" * 80)
    
    # Sample job descriptions
    sample_jobs = [
        {
            "content": """Senior Python Developer - AI/ML Focus
            
            We're seeking an experienced Python developer to join our AI team.
            
            Required Skills:
            - 5+ years Python experience
            - LangChain, LangGraph frameworks
            - OpenAI API, GPT models
            - Vector databases (Pinecone, ChromaDB)
            - REST API development
            
            Nice to have:
            - AWS deployment experience
            - Docker, Kubernetes
            - React frontend skills
            """,
            "metadata": {
                "title": "Senior Python Developer",
                "company": "AI Innovations Inc",
                "location": "San Francisco, CA",
                "salary": "$180k-$250k",
                "job_type": "full-time",
                "category": "engineering"
            }
        },
        {
            "content": """Data Scientist - Machine Learning
            
            Join our data science team to build cutting-edge ML models.
            
            Requirements:
            - PhD or Master's in Computer Science/Statistics
            - Python, R, SQL
            - TensorFlow, PyTorch, Scikit-learn
            - Experience with A/B testing
            - Statistical modeling expertise
            
            Preferred:
            - NLP experience
            - Cloud platforms (AWS/GCP)
            - Spark, Hadoop
            """,
            "metadata": {
                "title": "Data Scientist",
                "company": "DataCorp",
                "location": "Remote",
                "salary": "$160k-$220k",
                "job_type": "full-time",
                "category": "data-science"
            }
        },
        {
            "content": """Full Stack Engineer - React/Node.js
            
            Build modern web applications with cutting-edge technology.
            
            Tech Stack:
            - React, TypeScript, Next.js
            - Node.js, Express
            - PostgreSQL, MongoDB
            - AWS (Lambda, S3, CloudFront)
            - Docker, CI/CD
            
            Requirements:
            - 3+ years full stack development
            - Strong JavaScript/TypeScript skills
            - RESTful API design
            - Agile methodology
            """,
            "metadata": {
                "title": "Full Stack Engineer",
                "company": "StartupXYZ",
                "location": "New York, NY",
                "salary": "$140k-$190k",
                "job_type": "full-time",
                "category": "engineering"
            }
        },
        {
            "content": """DevOps Engineer - Cloud Infrastructure
            
            Manage and optimize our cloud infrastructure.
            
            Core Responsibilities:
            - AWS infrastructure management
            - Kubernetes cluster administration
            - CI/CD pipeline development
            - Infrastructure as Code (Terraform)
            - Monitoring and alerting
            
            Required:
            - 4+ years DevOps experience
            - AWS certification preferred
            - Strong Linux skills
            - Python/Bash scripting
            """,
            "metadata": {
                "title": "DevOps Engineer",
                "company": "CloudTech Solutions",
                "location": "Austin, TX",
                "salary": "$150k-$200k",
                "job_type": "full-time",
                "category": "devops"
            }
        }
    ]
    
    # Test 1: Initialize vector store (ChromaDB)
    print("\n1️⃣  Initializing ChromaDB vector store...")
    store = JobVectorStore(
        collection_name="test_jobs",
        use_chromadb=True
    )
    
    # Test 2: Add documents
    print("\n2️⃣  Adding sample job descriptions...")
    count = store.add_documents(sample_jobs)
    print(f"   ✅ Added {count} documents")
    print(f"   Total documents: {store.get_document_count()}")
    
    # Test 3: Basic similarity search
    print("\n3️⃣  Testing similarity search...")
    print("   Query: 'Python AI developer with LangChain'")
    results = store.similarity_search(
        query="Python AI developer with LangChain",
        k=3
    )
    
    print(f"\n   Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n   {i}. {doc['metadata']['title']} at {doc['metadata']['company']}")
        print(f"      Location: {doc['metadata']['location']}")
        print(f"      Similarity: {doc['similarity']:.3f}")
    
    # Test 4: Search with metadata filter
    print("\n4️⃣  Testing metadata filtering...")
    print("   Filter: category='engineering'")
    filtered_results = store.similarity_search(
        query="software engineer",
        k=5,
        filter_metadata={"category": "engineering"}
    )
    
    print(f"\n   Found {len(filtered_results)} engineering positions:")
    for doc in filtered_results:
        print(f"   - {doc['metadata']['title']} at {doc['metadata']['company']}")
    
    # Test 5: Search with similarity threshold
    print("\n5️⃣  Testing similarity threshold...")
    print("   Threshold: 0.7 (only highly relevant results)")
    high_quality_results = store.similarity_search(
        query="machine learning python",
        k=10,
        similarity_threshold=0.7
    )
    
    print(f"\n   Found {len(high_quality_results)} high-quality matches:")
    for doc in high_quality_results:
        print(f"   - {doc['metadata']['title']} (similarity: {doc['similarity']:.3f})")
    
    # Test 6: Get embedding
    print("\n6️⃣  Testing embedding generation...")
    sample_text = "I am an experienced Python developer"
    embedding = store.get_embedding(sample_text)
    print(f"   Text: '{sample_text}'")
    print(f"   Embedding dimensions: {len(embedding)}")
    print(f"   First 5 values: {[f'{x:.4f}' for x in embedding[:5]]}")
    
    # Test 7: Document count
    print("\n7️⃣  Document statistics...")
    total_docs = store.get_document_count()
    print(f"   Total documents in store: {total_docs}")
    
    # Test 8: In-memory fallback test
    print("\n8️⃣  Testing in-memory store...")
    memory_store = JobVectorStore(
        collection_name="test_memory",
        use_chromadb=False
    )
    memory_store.add_documents(sample_jobs[:2])
    mem_results = memory_store.similarity_search("Python developer", k=2)
    print(f"   ✅ In-memory store working ({len(mem_results)} results)")
    
    # Cleanup
    print("\n9️⃣  Cleanup...")
    if CHROMADB_AVAILABLE:
        store.clear()
        print("   ✅ ChromaDB collection cleared")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
