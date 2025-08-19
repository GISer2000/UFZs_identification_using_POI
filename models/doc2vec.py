import torch.nn as nn

class DM(nn.Module):
    def __init__(self, vocab_size, doc_size, embedding_dim):
        super(DM, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, embedding_dim)
        self.doc_emb = nn.Embedding(doc_size, embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size)
        nn.init.xavier_uniform_(self.word_emb.weight)
        nn.init.xavier_uniform_(self.doc_emb.weight)

    def forward(self, doc_id, context_ids):
        context_vec = self.word_emb(context_ids).mean(dim=1)
        doc_vec = self.doc_emb(doc_id)
        hidden = context_vec + doc_vec
        logits = self.out(hidden)
        return logits


class DBOW(nn.Module):
    def __init__(self, vocab_size, doc_size, embedding_dim):
        super(DBOW, self).__init__()
        self.doc_emb = nn.Embedding(doc_size, embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size)
        nn.init.xavier_uniform_(self.doc_emb.weight)

    def forward(self, doc_id):
        doc_vec = self.doc_emb(doc_id)
        logits = self.out(doc_vec)
        return logits