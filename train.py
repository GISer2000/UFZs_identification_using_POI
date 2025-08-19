import json

from models.word2vec import CBOW, SkipGram
from models.doc2vec import DM, DBOW
from models.glove import GloVe
from data.dataset import ProcessingData, BuildVocab, Word2VecDataset, Doc2VecDataset, BuildCoMatrix
from trainers.trainer import build_trainer

if __name__ == "__main__":

    # configs_path = "configs/word2vec_cbow.json"
    # configs_path = "configs/word2vec_sg.json"
    # configs_path = "configs/glove.json"
    configs_path = "configs/config.json"
    with open(configs_path, "r", encoding="utf-8") as f:
        config = json.load(f)


    if config.get("model_name") == "word2vec" and config.get("model_type") not in ["cbow", "skipgram"]:
        raise ValueError("Word2Vec 模型类型必须是 'cbow' 或 'skipgram'")
    if config.get("model_name") == "doc2vec" and config.get("model_type") not in ["dm", "dbow"]:
        raise ValueError("Doc2Vec 模型类型必须是 'dm' 或 'dbow'")
    if config.get("model_name") == "glove" and config.get("model_type") is not None:
        raise ValueError("GloVe 模型不需要指定模型类型")

    print("Training configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")


    print("\nStarting training...")
    corpus = ProcessingData(docs_path=config["docs_path"])
    vocab = BuildVocab(docs=corpus, min_count=config["min_count"])
    cooc_matrix = BuildCoMatrix(docs=corpus, vocab=vocab, window_size=config["window_size"])


    dataset_map = {
        "word2vec": lambda: Word2VecDataset(
            corpus, vocab, config.get("window_size"), 
            config.get("neg_sample_num",5), config.get("model_type")),
        "doc2vec": lambda: Doc2VecDataset(
            corpus, vocab, config.get("window_size"), config.get("model_type")),
        "glove": lambda: [(i, j, val) for (i, j), val in cooc_matrix.items()]
    }
    model_name = config.get("model_name")
    if model_name in dataset_map:
        dataset = dataset_map[model_name]()
    else:
        raise ValueError(f"未知模型名称: {model_name}")

    model_map = {
        "word2vec": {
            "cbow": lambda: CBOW(vocab_size=len(vocab), embedding_dim=config["word_embedding"]),
            "skipgram": lambda: SkipGram(vocab_size=len(vocab), embedding_dim=config["word_embedding"])
        },
        "doc2vec": {
            "dm": lambda: DM(vocab_size=len(vocab), doc_size=len(corpus), embedding_dim=config["word_embedding"]),
            "dbow": lambda: DBOW(vocab_size=len(vocab), doc_size=len(corpus), embedding_dim=config["word_embedding"])
        },
        "glove": {
            "_default": lambda: GloVe(vocab_size=len(vocab), embedding_dim=config["word_embedding"])
        }
    }
    model_type = config.get("model_type")
    builder = model_map.get(model_name, {}).get(
        model_type if model_name != "glove" else "_default"
    )
    if builder:
        model = builder()
    else:
        raise ValueError(f"不支持的模型组合: model_name={model_name}, model_type={model_type}")


    trainer = build_trainer(model, dataset, vocab, config)
    trainer.train()