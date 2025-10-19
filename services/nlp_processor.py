"""
NLP processing utilities for text analysis and language processing.
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
import nltk
from collections import Counter

# NLP imports with fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import language_tool_python
    GRAMMAR_TOOL_AVAILABLE = True
except ImportError:
    GRAMMAR_TOOL_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    Natural Language Processing utilities for resume analysis.
    Handles text cleaning, tokenization, keyword extraction, and language analysis.
    """
    
    def __init__(self):
        """Initialize the NLP processor."""
        self.nlp_model = None
        self.stop_words = set()
        self.lemmatizer = None
        self.grammar_tool = None
        self.is_initialized = False
        
        # Initialize NLTK data
        self._download_nltk_data()
    
    def initialize(self):
        """Initialize NLP models and tools."""
        try:
            # Initialize spaCy model
            if SPACY_AVAILABLE:
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded successfully")
                except OSError:
                    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                    self.nlp_model = None
            
            # Initialize NLTK components
            if NLTK_AVAILABLE:
                try:
                    self.stop_words = set(stopwords.words('english'))
                    self.lemmatizer = WordNetLemmatizer()
                    logger.info("NLTK components initialized successfully")
                except Exception as e:
                    logger.warning(f"NLTK initialization failed: {e}")
                    self.stop_words = self._get_basic_stopwords()
                    self.lemmatizer = None
            
            # Initialize grammar checker
            if GRAMMAR_TOOL_AVAILABLE:
                try:
                    self.grammar_tool = language_tool_python.LanguageTool('en-US')
                    logger.info("Grammar checker initialized successfully")
                except Exception as e:
                    logger.warning(f"Grammar checker initialization failed: {e}")
                    self.grammar_tool = None
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP processor: {e}")
            self.is_initialized = False
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK data: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
            
            # Remove multiple periods
            text = re.sub(r'\.{2,}', '.', text)
            
            # Remove multiple spaces
            text = re.sub(r' {2,}', ' ', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract keywords from text using TF-IDF.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        try:
            if not SKLEARN_AVAILABLE:
                return self._extract_keywords_basic(text, max_keywords)
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Tokenize and filter
            tokens = self._tokenize_text(cleaned_text)
            filtered_tokens = [token for token in tokens if len(token) > 2 and token.lower() not in self.stop_words]
            
            if not filtered_tokens:
                return []
            
            # Use TF-IDF to extract keywords
            vectorizer = TfidfVectorizer(
                max_features=max_keywords,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_tokens)])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top keywords
            scores = tfidf_matrix.toarray()[0]
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            keywords = [keyword for keyword, score in keyword_scores[:max_keywords]]
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return self._extract_keywords_basic(text, max_keywords)
    
    def _extract_keywords_basic(self, text: str, max_keywords: int) -> List[str]:
        """Basic keyword extraction fallback."""
        try:
            # Simple frequency-based keyword extraction
            tokens = self._tokenize_text(text)
            filtered_tokens = [token.lower() for token in tokens 
                            if len(token) > 2 and token.lower() not in self.stop_words]
            
            # Count word frequencies
            word_counts = Counter(filtered_tokens)
            
            # Return most common words
            keywords = [word for word, count in word_counts.most_common(max_keywords)]
            return keywords
            
        except Exception as e:
            logger.error(f"Error in basic keyword extraction: {e}")
            return []
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self._tokenize_text(text)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Internal tokenization method."""
        try:
            if NLTK_AVAILABLE:
                return word_tokenize(text)
            else:
                # Basic tokenization fallback
                return text.split()
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return text.split()
    
    def sent_tokenize(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            if NLTK_AVAILABLE:
                return sent_tokenize(text)
            else:
                # Basic sentence splitting fallback
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Error splitting sentences: {e}")
            return [text]
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        try:
            if not self.nlp_model:
                return {"error": "spaCy model not available"}
            
            doc = self.nlp_model(text)
            entities = {}
            
            for ent in doc.ents:
                entity_type = ent.label_
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(ent.text)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting named entities: {e}")
            return {"error": str(e)}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment analysis results
        """
        try:
            # Basic sentiment analysis using word counting
            positive_words = ['excellent', 'outstanding', 'achieved', 'successful', 'improved', 'increased', 'led', 'managed']
            negative_words = ['failed', 'poor', 'decreased', 'problem', 'issue', 'challenge']
            
            tokens = self._tokenize_text(text.lower())
            
            positive_count = sum(1 for token in tokens if token in positive_words)
            negative_count = sum(1 for token in tokens if token in negative_words)
            
            total_words = len(tokens)
            
            if total_words == 0:
                return {"sentiment": "neutral", "score": 0.0}
            
            sentiment_score = (positive_count - negative_count) / total_words
            
            if sentiment_score > 0.1:
                sentiment = "positive"
            elif sentiment_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "score": sentiment_score,
                "positive_words": positive_count,
                "negative_words": negative_count
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"sentiment": "neutral", "score": 0.0}
    
    def check_grammar(self, text: str) -> List[Dict[str, Any]]:
        """
        Check grammar and language quality.
        
        Args:
            text: Input text
            
        Returns:
            List of grammar issues
        """
        try:
            if self.grammar_tool and GRAMMAR_TOOL_AVAILABLE:
                matches = self.grammar_tool.check(text[:5000])  # Limit text length
                issues = []
                
                for match in matches:
                    issues.append({
                        "message": match.message,
                        "offset": match.offset,
                        "length": match.length,
                        "rule_id": match.ruleId,
                        "suggestions": match.replacements[:3] if match.replacements else []
                    })
                
                return issues
            else:
                return self._basic_grammar_check(text)
                
        except Exception as e:
            logger.error(f"Error checking grammar: {e}")
            return self._basic_grammar_check(text)
    
    def _basic_grammar_check(self, text: str) -> List[Dict[str, Any]]:
        """Basic grammar check fallback."""
        issues = []
        
        try:
            sentences = self.sent_tokenize(text)
            
            for i, sentence in enumerate(sentences):
                # Check for long sentences
                if len(sentence.split()) > 30:
                    issues.append({
                        "message": f"Sentence {i+1} might be too long ({len(sentence.split())} words)",
                        "offset": 0,
                        "length": len(sentence),
                        "rule_id": "SENTENCE_LENGTH",
                        "suggestions": ["Consider breaking into shorter sentences"]
                    })
                
                # Check for repeated words
                words = sentence.lower().split()
                for j in range(len(words) - 1):
                    if words[j] == words[j + 1] and len(words[j]) > 3:
                        issues.append({
                            "message": f"Repeated word '{words[j]}' in sentence {i+1}",
                            "offset": 0,
                            "length": len(words[j]),
                            "rule_id": "REPEATED_WORD",
                            "suggestions": [f"Remove duplicate '{words[j]}'"]
                        })
            
            return issues
            
        except Exception as e:
            logger.error(f"Error in basic grammar check: {e}")
            return []
    
    def calculate_readability(self, text: str) -> Dict[str, Any]:
        """
        Calculate text readability metrics.
        
        Args:
            text: Input text
            
        Returns:
            Readability metrics
        """
        try:
            sentences = self.sent_tokenize(text)
            words = self._tokenize_text(text)
            
            if not sentences or not words:
                return {"error": "No text to analyze"}
            
            # Basic readability metrics
            avg_words_per_sentence = len(words) / len(sentences)
            avg_syllables_per_word = self._count_syllables(text) / len(words)
            
            # Flesch Reading Ease (simplified)
            flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            
            # Determine readability level
            if flesch_score >= 90:
                level = "Very Easy"
            elif flesch_score >= 80:
                level = "Easy"
            elif flesch_score >= 70:
                level = "Fairly Easy"
            elif flesch_score >= 60:
                level = "Standard"
            elif flesch_score >= 50:
                level = "Fairly Difficult"
            elif flesch_score >= 30:
                level = "Difficult"
            else:
                level = "Very Difficult"
            
            return {
                "flesch_score": round(flesch_score, 2),
                "readability_level": level,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2),
                "avg_syllables_per_word": round(avg_syllables_per_word, 2),
                "total_words": len(words),
                "total_sentences": len(sentences)
            }
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return {"error": str(e)}
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified method)."""
        try:
            words = self._tokenize_text(text.lower())
            syllable_count = 0
            
            for word in words:
                # Remove punctuation
                word = re.sub(r'[^a-z]', '', word)
                if not word:
                    continue
                
                # Count vowels
                vowels = 'aeiouy'
                syllable_count += sum(1 for char in word if char in vowels)
                
                # Adjust for silent e
                if word.endswith('e'):
                    syllable_count -= 1
                
                # Ensure at least one syllable per word
                if syllable_count <= 0:
                    syllable_count = 1
            
            return syllable_count
            
        except Exception as e:
            logger.error(f"Error counting syllables: {e}")
            return len(text.split())  # Fallback
    
    def _get_basic_stopwords(self) -> set:
        """Get basic stopwords if NLTK is not available."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Get NLP processor status and capabilities."""
        return {
            "initialized": self.is_initialized,
            "spacy_available": SPACY_AVAILABLE,
            "nltk_available": NLTK_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "fuzzywuzzy_available": FUZZYWUZZY_AVAILABLE,
            "grammar_tool_available": GRAMMAR_TOOL_AVAILABLE,
            "spacy_model_loaded": self.nlp_model is not None,
            "grammar_tool_loaded": self.grammar_tool is not None,
            "stop_words_count": len(self.stop_words),
            "lemmatizer_available": self.lemmatizer is not None
        }
