import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.in_emb = nn.Embedding(vocab_size, embedding_dim)
        self.word_emb = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids):
        embeds = self.in_emb(context_ids)
        hidden = embeds.mean(dim=1)
        out = self.word_emb(hidden)
        return out


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.in_emb = nn.Embedding(vocab_size, embedding_dim)
        self.word_emb = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_ids, pos_ids, neg_ids):
        v = self.in_emb(target_ids)
        u_pos = self.word_emb(pos_ids)
        u_neg = self.word_emb(neg_ids)

        pos_score = torch.mul(v, u_pos).sum(dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-9)

        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
        neg_loss = torch.log(torch.sigmoid(-neg_score) + 1e-9).sum(dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return loss