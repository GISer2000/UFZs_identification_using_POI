import torch
import torch.nn as nn

class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVe, self).__init__()
        self.target_emb = nn.Embedding(vocab_size, embedding_dim)
        self.context_emb = nn.Embedding(vocab_size, embedding_dim)
        self.target_bias = nn.Embedding(vocab_size, 1)
        self.context_bias = nn.Embedding(vocab_size, 1)
        
        nn.init.xavier_uniform_(self.target_emb.weight)
        nn.init.xavier_uniform_(self.context_emb.weight)
        nn.init.zeros_(self.target_bias.weight)
        nn.init.zeros_(self.context_bias.weight)

    def forward(self, target_idx, context_idx, cooc_val):
        target_vec = self.target_emb(target_idx)
        context_vec = self.context_emb(context_idx)
        target_bias = self.target_bias(target_idx).squeeze()
        context_bias = self.context_bias(context_idx).squeeze()

        x_max = 100.0
        alpha = 0.75
        weight = torch.pow(torch.clamp(cooc_val / x_max, max=1.0), alpha)

        loss = weight * (torch.sum(target_vec * context_vec, dim=1) 
                 + target_bias + context_bias 
                 - torch.log(cooc_val + 1e-8)) ** 2
        
        return loss.mean()