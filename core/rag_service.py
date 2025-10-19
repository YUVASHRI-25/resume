"""
RAG (Retrieval-Augmented Generation) service using Chroma DB for resume analysis.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG service for finding similar resumes and providing contextual recommendations.
    Uses Chroma DB for vector storage and retrieval.
    """
    
    def __init__(self):
        """Initialize the RAG service."""
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.is_initialized = False
        
        # Configuration
        self.persist_directory = settings.rag.chroma_persist_directory
        self.embedding_model_name = settings.rag.embedding_model
        self.similarity_threshold = settings.rag.similarity_threshold
        self.max_results = settings.rag.max_results
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def initialize(self):
        """Initialize Chroma DB client and embedding model."""
        try:
            # Initialize Chroma DB client
            if CHROMADB_AVAILABLE:
                self.chroma_client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info("Chroma DB client initialized successfully")
            
            # Initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Embedding model {self.embedding_model_name} loaded successfully")
            
            # Get or create collection
            self._setup_collection()
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            self.is_initialized = False
    
    def _setup_collection(self):
        """Set up or get existing Chroma DB collection."""
        try:
            collection_name = "resume_embeddings"
            
            # Try to get existing collection
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                logger.info(f"Retrieved existing collection: {collection_name}")
            except:
                # Create new collection if it doesn't exist
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Resume embeddings for similarity search"}
                )
                logger.info(f"Created new collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            self.collection = None
    
    def add_resume_to_index(self, resume_id: str, resume_text: str, 
                           metadata: Dict[str, Any] = None) -> bool:
        """
        Add a resume to the vector index.
        
        Args:
            resume_id: Unique identifier for the resume
            resume_text: Resume content
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        if not self.is_initialized:
            logger.error("RAG service not initialized")
            return False
        
        try:
            # Generate embedding
            embedding = self._generate_embedding(resume_text)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "resume_id": resume_id,
                "text_length": len(resume_text),
                "added_at": datetime.utcnow().isoformat()
            })
            
            # Add to collection
            self.collection.add(
                ids=[resume_id],
                embeddings=[embedding],
                documents=[resume_text],
                metadatas=[metadata]
            )
            
            logger.info(f"Added resume {resume_id} to vector index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding resume to index: {e}")
            return False
    
    def get_similar_resumes(self, resume_text: str, job_role: str = None, 
                          limit: int = None) -> Dict[str, Any]:
        """
        Find similar resumes based on text content.
        
        Args:
            resume_text: Resume content to compare
            job_role: Optional job role filter
            limit: Maximum number of results
            
        Returns:
            Similar resumes with metadata
        """
        if not self.is_initialized:
            logger.error("RAG service not initialized")
            return {"error": "RAG service not initialized"}
        
        try:
            # Generate embedding for query
            query_embedding = self._generate_embedding(resume_text)
            
            # Prepare query metadata filter
            where_clause = None
            if job_role:
                where_clause = {"job_role": job_role}
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit or self.max_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            similar_resumes = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score
                similarity_score = 1 - distance
                
                if similarity_score >= self.similarity_threshold:
                    similar_resumes.append({
                        "resume_id": metadata.get("resume_id"),
                        "similarity_score": similarity_score,
                        "content_preview": doc[:500] + "..." if len(doc) > 500 else doc,
                        "metadata": metadata
                    })
            
            return {
                "similar_resumes": similar_resumes,
                "total_found": len(similar_resumes),
                "query_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error finding similar resumes: {e}")
            return {"error": f"Search failed: {str(e)}"}
    
    def get_contextual_recommendations(self, resume_text: str, job_role: str) -> Dict[str, Any]:
        """
        Get contextual recommendations based on similar resumes.
        
        Args:
            resume_text: Resume content
            job_role: Target job role
            
        Returns:
            Contextual recommendations
        """
        try:
            # Find similar resumes
            similar_resumes = self.get_similar_resumes(resume_text, job_role)
            
            if "error" in similar_resumes:
                return similar_resumes
            
            # Analyze patterns from similar resumes
            recommendations = self._analyze_patterns(similar_resumes["similar_resumes"])
            
            return {
                "recommendations": recommendations,
                "based_on_resumes": len(similar_resumes["similar_resumes"]),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting contextual recommendations: {e}")
            return {"error": f"Recommendation generation failed: {str(e)}"}
    
    def update_resume_in_index(self, resume_id: str, resume_text: str, 
                             metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing resume in the index.
        
        Args:
            resume_id: Resume identifier
            resume_text: Updated resume content
            metadata: Updated metadata
            
        Returns:
            Success status
        """
        try:
            # First, try to delete existing entry
            try:
                self.collection.delete(ids=[resume_id])
            except:
                pass  # Resume might not exist
            
            # Add updated resume
            return self.add_resume_to_index(resume_id, resume_text, metadata)
            
        except Exception as e:
            logger.error(f"Error updating resume in index: {e}")
            return False
    
    def remove_resume_from_index(self, resume_id: str) -> bool:
        """
        Remove a resume from the index.
        
        Args:
            resume_id: Resume identifier
            
        Returns:
            Success status
        """
        try:
            self.collection.delete(ids=[resume_id])
            logger.info(f"Removed resume {resume_id} from index")
            return True
            
        except Exception as e:
            logger.error(f"Error removing resume from index: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample metadata
            sample_results = self.collection.get(limit=10, include=["metadatas"])
            
            job_roles = set()
            for metadata in sample_results["metadatas"]:
                if "job_role" in metadata:
                    job_roles.add(metadata["job_role"])
            
            return {
                "total_resumes": count,
                "unique_job_roles": len(job_roles),
                "job_roles": list(job_roles),
                "collection_name": self.collection.name,
                "embedding_model": self.embedding_model_name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": f"Failed to get stats: {str(e)}"}
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured model."""
        try:
            if self.embedding_model:
                # Truncate text if too long
                max_length = 512
                if len(text) > max_length:
                    text = text[:max_length]
                
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            else:
                logger.error("Embedding model not available")
                return []
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def _analyze_patterns(self, similar_resumes: List[Dict[str, Any]]) -> List[str]:
        """
        Analyze patterns from similar resumes to generate recommendations.
        
        Args:
            similar_resumes: List of similar resume data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not similar_resumes:
            return ["No similar resumes found for comparison"]
        
        # Analyze common patterns
        high_scoring_resumes = [r for r in similar_resumes if r["similarity_score"] > 0.8]
        
        if high_scoring_resumes:
            recommendations.append(
                f"Found {len(high_scoring_resumes)} highly similar resumes. "
                "Consider reviewing their structure and content for inspiration."
            )
        
        # Analyze content length patterns
        content_lengths = [len(r["content_preview"]) for r in similar_resumes]
        avg_length = sum(content_lengths) / len(content_lengths)
        
        if avg_length > 2000:
            recommendations.append(
                "Similar resumes tend to be more detailed. Consider expanding your content."
            )
        elif avg_length < 1000:
            recommendations.append(
                "Similar resumes are more concise. Consider tightening your content."
            )
        
        # Analyze metadata patterns
        job_roles = [r["metadata"].get("job_role") for r in similar_resumes if r["metadata"].get("job_role")]
        if job_roles:
            most_common_role = max(set(job_roles), key=job_roles.count)
            recommendations.append(
                f"Most similar resumes are for {most_common_role} roles. "
                "Ensure your resume aligns with this role's requirements."
            )
        
        return recommendations
    
    def bulk_index_resumes(self, resumes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bulk index multiple resumes.
        
        Args:
            resumes_data: List of resume data dictionaries
            
        Returns:
            Bulk indexing results
        """
        try:
            successful = 0
            failed = 0
            
            for resume_data in resumes_data:
                resume_id = resume_data.get("id")
                resume_text = resume_data.get("text", "")
                metadata = resume_data.get("metadata", {})
                
                if self.add_resume_to_index(resume_id, resume_text, metadata):
                    successful += 1
                else:
                    failed += 1
            
            return {
                "successful": successful,
                "failed": failed,
                "total": len(resumes_data),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in bulk indexing: {e}")
            return {"error": f"Bulk indexing failed: {str(e)}"}
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get RAG service status and configuration."""
        return {
            "initialized": self.is_initialized,
            "chromadb_available": CHROMADB_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "embedding_model": self.embedding_model_name,
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "persist_directory": self.persist_directory,
            "collection_exists": self.collection is not None
        }
