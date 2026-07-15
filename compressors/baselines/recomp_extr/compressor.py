"""RECOMP Extractive implementation for document compression."""

import torch
from typing import List
from transformers import AutoTokenizer
from ...base import BaseCompressor, SearchResult
from .contriever import Contriever


class RecompExtractiveCompressor(BaseCompressor):
    """RECOMP: Extractive document compression using dense retrieval."""
    
    def __init__(
        self,
        model_name: str = "fangyuan/nq_extractive_compressor",
        batch_size: int = 32,
        cache_dir: str = "./cache",
        device: str = None,
        n_sentences: int = 2
    ):
        """Initialize RECOMP Extractive compressor.
        
        Args:
            model_name: Model identifier from HuggingFace
            batch_size: Batch size for processing
            cache_dir: Directory for model caching
            device: Device to run on (None for auto)
            n_sentences: Number of sentences to select
        """
        self.batch_size = batch_size
        self.n_sentences = n_sentences
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else torch.device(device)
        )
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        self.model = Contriever.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(self.device)
        
        self.model.eval()
    
    def encode_texts(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """Encode texts into embeddings."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize and encode
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.model(**inputs)
                all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents by selecting top n sentences.
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            List containing selected top sentences
        """
        if not documents:
            return []
            
        # Encode query and documents
        texts = [query] + [doc.text for doc in documents]
        embeddings = self.encode_texts(texts)
        
        # Split query and document embeddings
        query_embedding = embeddings[0].unsqueeze(0)
        doc_embeddings = embeddings[1:]
        
        # Compute similarity scores
        if len(doc_embeddings) == 1:
            similarity_scores = (query_embedding @ doc_embeddings.T)
        else:
            similarity_scores = (query_embedding @ doc_embeddings.T).squeeze()
        
        # Convert to CPU and get top k indices
        scores = similarity_scores.cpu()
        n_select = min(self.n_sentences, len(documents))
        top_indices = torch.topk(scores, n_select).indices.tolist()
        
        # If only one document, convert to list
        if isinstance(top_indices, int):
            top_indices = [top_indices]
        
        # Select top documents
        compressed_docs = []
        for idx in top_indices:
            doc = documents[idx]
            compressed_docs.append(SearchResult(
                evi_id=doc.evi_id,
                docid=doc.docid,
                title=doc.title,
                text=doc.text,
                score=float(scores[idx])
            ))
        
        return sorted(
            compressed_docs,
            key=lambda x: x.score,
            reverse=True
        )
