import ast
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from collections import Counter, defaultdict


def ProcessingData(docs_path: str) -> list:
    Docs = pd.read_csv(docs_path)
    Docs['Doc'] = Docs['Doc'].apply(lambda x: ast.literal_eval(x))
    return Docs['Doc']

def BuildVocab(docs: list, min_count: int=1) -> dict:
    word_counts = Counter(word for doc in docs for word in doc)
    vocab = {word: i for i, (word, count) in enumerate(word_counts.items()) if count >= min_count}
    return vocab

def BuildCoMatrix(docs: list, vocab: list, window_size: int=5) -> dict:
    cooc = defaultdict(float)
    for doc in docs:
        indexed_doc = [vocab[w] for w in doc if w in vocab]
        for i, wi in enumerate(indexed_doc):
            for j in range(max(0, i - window_size), min(len(indexed_doc), i + window_size + 1)):
                if i != j:
                    cooc[(wi, indexed_doc[j])] += 1.0
    return cooc

class Word2VecDataset(Dataset):
    def __init__(self, docs, vocab, window_size=2, neg_sample_num=5, model_type="cbow"):
        self.vocab = vocab
        self.word2idx = vocab
        self.idx2word = {i: w for w, i in vocab.items()}
        self.vocab_size = len(vocab)
        self.window_size = window_size
        self.neg_sample_num = neg_sample_num
        self.model_type = model_type.lower()

        tokens = [w for doc in docs for w in doc if w in vocab]

        word_counts = Counter(tokens)
        freqs = np.array([word_counts[self.idx2word[i]] for i in range(len(vocab))])
        unigram_dist = freqs / freqs.sum()
        self.noise_dist = (unigram_dist ** 0.75)
        self.noise_dist /= self.noise_dist.sum()

        self.data = []
        if self.model_type == "cbow":
            for i in range(window_size, len(tokens) - window_size):
                context = tokens[i - window_size:i] + tokens[i + 1:i + window_size + 1]
                target = tokens[i]
                context_ids = [vocab[w] for w in context]
                target_id = vocab[target]
                self.data.append((context_ids, target_id))

        elif self.model_type == "skipgram":
            for i in range(window_size, len(tokens) - window_size):
                target = tokens[i]
                context = tokens[i - window_size:i] + tokens[i + 1:i + window_size + 1]
                target_id = vocab[target]
                for w in context:
                    pos_id = vocab[w]
                    neg_ids = np.random.choice(len(vocab), self.neg_sample_num, p=self.noise_dist)
                    self.data.append((target_id, pos_id, neg_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.model_type == "cbow":
            context_ids, target_id = self.data[idx]
            return torch.tensor(context_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)
        elif self.model_type == "skipgram":
            target_id, pos_id, neg_ids = self.data[idx]
            return (
                torch.tensor(target_id, dtype=torch.long),
                torch.tensor(pos_id, dtype=torch.long),
                torch.tensor(neg_ids, dtype=torch.long)
            )
        

class Doc2VecDataset(Dataset):
    def __init__(self, docs, vocab, window_size=5, model_type="dm"):
        self.data = []
        self.model_type = model_type
        self.vocab = vocab
        self.word2id = {w: i for i, w in enumerate(vocab)}
        
        for doc_id, doc in enumerate(docs):
            indexed = [self.word2id[w] for w in doc if w in self.word2id]
            for i, word in enumerate(indexed):
                if model_type == "dm":
                    start = max(0, i - window_size)
                    end = min(len(indexed), i + window_size + 1)
                    context = [indexed[j] for j in range(start, end) if j != i]
                    if len(context) > 0:
                        self.data.append((doc_id, context, word))
                else:
                    window = np.random.randint(1, window_size + 1)
                    start = max(0, i - window)
                    end = min(len(indexed), i + window + 1)
                    context = [indexed[j] for j in range(start, end) if j != i]
                    for w in context:
                        self.data.append((doc_id, w))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]