from transformers import BertModel, BertTokenizer
import torch
from typing import List, Dict, Union
import numpy as np

class BERT:
    def __init__(self, model_name: str = "bert-base-uncased", sep: str = " "):
        self.sep = sep
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        embeddings = []
        for start_idx in range(0, len(queries), batch_size):
            batch_queries = queries[start_idx:start_idx + batch_size]
            inputs = self.tokenizer(batch_queries, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        embeddings = []
        for start_idx in range(0, len(sentences), batch_size):
            batch_sentences = sentences[start_idx:start_idx + batch_size]
            inputs = self.tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.concatenate(embeddings, axis=0)