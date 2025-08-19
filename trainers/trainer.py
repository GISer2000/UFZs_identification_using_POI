import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataset, vocab, config, device=None):
        self.model = model
        self.dataset = dataset
        self.vocab = vocab
        self.config = config

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # dataloader
        if config.get("model_name") == "doc2vec":
            if config.get("model_type") == "dm":
                self.dataloader = DataLoader(
                    dataset,
                    batch_size=config.get("batch_size", 128),
                    shuffle=True,
                    collate_fn=self.collate_fn_dm
                )
            elif config.get("model_type") == "dbow":
                self.dataloader = DataLoader(
                    dataset,
                    batch_size=config.get("batch_size", 128),
                    shuffle=True,
                    collate_fn=self.collate_fn_dbow
                )
        else:
            self.dataloader = DataLoader(
                dataset,
                batch_size=config.get("batch_size", 128),
                shuffle=True
            )
        
        # optimizer and criterion
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=config.get("lr", 1e-3)
            )
        use_cross_entropy = (config.get("model_name") in ["word2vec", "doc2vec"] 
                            and config.get("model_type") != "skipgram")
        print(use_cross_entropy)
        self.criterion = torch.nn.CrossEntropyLoss() if use_cross_entropy else None

        # save
        for dir_path in [
            config.get("save_models", "checkpoints"), 
            config.get("get_embeddings", "embeddings")]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def collate_fn_dm(self, batch):
        doc_ids, contexts, targets = zip(*batch)
        max_len = max(len(c) for c in contexts)
        padded = [c + [0] * (max_len - len(c)) for c in contexts]
        return (
            torch.tensor(doc_ids, dtype=torch.long),
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )

    def collate_fn_dbow(self, batch):
        doc_ids, targets = zip(*batch)
        return (
            torch.tensor(doc_ids, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        loop = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False)

        for batch in loop:
            self.optimizer.zero_grad()
            if self.config.get("model_name") == "word2vec":
                if self.config.get("model_type") == "cbow":
                    context_ids, target_id = batch
                    context_ids, target_id = (
                        context_ids.to(self.device),
                        target_id.to(self.device)
                    )
                    
                    outputs = self.model(context_ids)
                    loss = self.criterion(outputs, target_id)
                elif self.config.get("model_type") == "skipgram":
                    target_id, pos_id, neg_ids = batch
                    target_id, pos_id, neg_ids = (
                        target_id.to(self.device),
                        pos_id.to(self.device),
                        neg_ids.to(self.device)
                    )
                    
                    loss = self.model(target_id, pos_id, neg_ids)

            elif self.config.get("model_name") == "doc2vec":
                if self.config.get("model_type") == "dm":
                    doc_ids, context_ids, target_ids = [b.to(self.device) for b in batch]
                    
                    outputs = self.model(doc_ids, context_ids)
                    loss = self.criterion(outputs, target_ids)
                elif self.config.get("model_type") == "dbow":
                    doc_ids, target_ids = [b.to(self.device) for b in batch]
                    
                    outputs = self.model(doc_ids)
                    loss = self.criterion(outputs, target_ids)

            else:  # GloVe
                target_idx, context_idx, cooc_val = batch
                target_idx, context_idx, cooc_val = (
                    target_idx.to(self.device),
                    context_idx.to(self.device),
                    cooc_val.to(self.device)
                )
                
                loss = self.model(target_idx, context_idx, cooc_val)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        best_loss = float("inf")
        epochs = self.config.get("epochs", 10)

        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(epoch)

            # 保存最好模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model()
        
        # 获取嵌入矩阵
        self.get_embeddings()

    def save_model(self):
        filename = self.get_filename(save_type="model")
        path = os.path.join(self.config.get("save_models"), filename)
        torch.save(self.model.state_dict(), path)
        print(f"保存最佳模型至 {path}")

    def get_embeddings(self):
        if self.config.get("model_name") == "word2vec":
            embeddings = self.model.word_emb.weight.data.cpu().numpy()
            idx2word = {i: w for w, i in self.vocab.items()}
            words = [idx2word[i] for i in range(len(self.vocab))]
            word_df = (
                pd.DataFrame(embeddings, index=words)
                .reset_index()
                .rename(columns={'index': 'word'})
            )
        elif self.config.get("model_name") == "doc2vec":
            if self.config.get("model_type") == "dm":
                word_embeddings = self.model.word_emb.weight.data.cpu().numpy()
                doc_embeddings = self.model.doc_emb.weight.data.cpu().numpy()
                idx2word = {i: w for w, i in self.vocab.items()}
                word_df = (
                    pd
                    .DataFrame(word_embeddings, index=idx2word.values())
                    .reset_index()
                    .rename(columns={'index': 'word'})
                )
                doc_df = (
                    pd
                    .DataFrame(doc_embeddings, index=[f"doc_{i}" for i in range(len(doc_embeddings))])
                    .reset_index()
                    .rename(columns={'index': 'doc_id'})
                )
            elif self.config.get("model_type") == "dbow":
                doc_embeddings = self.model.doc_emb.weight.data.cpu().numpy()
                doc_df = (
                    pd
                    .DataFrame(doc_embeddings, index=[f"doc_{i}" for i in range(len(doc_embeddings))])
                    .reset_index()
                    .rename(columns={'index': 'doc_id'})
                )
        elif self.config.get("model_name") == "glove":
                target_vectors = self.model.target_emb.weight.data.cpu().numpy()
                context_vectors = self.model.context_emb.weight.data.cpu().numpy()
                word_vectors = (target_vectors + context_vectors) / 2  # 两者的平均
                word_df = (
                    pd
                    .DataFrame(word_vectors, index=self.vocab.keys())
                    .reset_index()
                    .rename(columns={'index': 'word'})
                )
        # save embeddings
        try:
            if not word_df.empty:
                filename = self.get_filename(save_type="word_embeddings")
                path = os.path.join(self.config.get("get_embeddings"), filename)
                word_df.to_csv(path, index=False)
                print(f"\n词嵌入已保存至 {path}")
        except NameError:
            pass
        try:
            if not doc_df.empty:
                filename = self.get_filename(save_type="doc_embeddings")
                path = os.path.join(self.config.get("get_embeddings"), filename)
                doc_df.to_csv(path, index=False)
                print(f"\n文档嵌入已保存至 {path}")
        except NameError:
            pass

    def get_filename(self, save_type="model"):
        if save_type == "model":
            model_name = self.config.get("model_name", "model")
            model_type = self.config.get("model_type", "")
            if model_type:
                filename = f"{model_name}_{model_type}.pt"
            else:
                filename = f"{model_name}.pt"
        elif save_type == "word_embeddings":
            model_name = self.config.get("model_name", "model")
            model_type = self.config.get("model_type", "")
            if model_type:
                filename = f"{model_name}_{model_type}_word.csv"
            else:
                filename = f"{model_name}_word.csv"
        elif save_type == "doc_embeddings":
            model_name = self.config.get("model_name", "model")
            model_type = self.config.get("model_type", "")
            if model_type:
                filename = f"{model_name}_{model_type}_doc.csv"
            else:
                filename = f"{model_name}_doc.csv"
        return filename


def build_trainer(model, dataset, vocab, config):
    """
    外部调用入口，返回 Trainer 实例
    """
    return Trainer(model, dataset, vocab, config)