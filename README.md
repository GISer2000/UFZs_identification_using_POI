# UFZ：用于城市功能区识别的POI表征模型

- 该项目提供了一个统一的训练框架，用于训练多种文本嵌入模型（Word2Vec、Doc2Vec、GloVe），目前仅支持处理好的中文语料库词。
- 可以训练获取词嵌入和文档嵌入。

## 项目结构

```plaintext
Structure/
├── configs/               # 模型配置文件目录
│   ├── doc2vec_dbow.json
│   ├── doc2vec_dm.json
│   ├── glove.json
│   ├── word2vec_cbow.json
│   └── word2vec_sg.json
├── data/                  # 数据处理模块
│   └── dataset.py         # 数据集构建和预处理
├── models/                # 模型定义
│   ├── doc2vec.py         # Doc2Vec模型(DM/DBOW)
│   ├── glove.py           # GloVe模型
│   └── word2vec.py        # Word2Vec模型(CBOW/SkipGram)
├── trainers/              # 训练器模块
│   └── trainer.py         # 训练逻辑实现
├── notebooks/             # 辅助工具
│   └── set_font.py        # 中文字体设置
├── dataset/               # 语料库目录
│   └── 地理语料库_taz.csv  # 示例语料库
├── checkpoints/           # 模型保存目录(自动生成)
├── embeddings/            # 嵌入结果保存目录(自动生成)
└── train.py               # 训练入口脚本
```

## 支持的模型

1. [Word2Vec](https://arxiv.org/abs/1301.3781)
    - CBOW (Continuous Bag-of-Words)
    - SkipGram
2. [Doc2Vec](https://arxiv.org/abs/1405.4053)
    - DM (Distributed Memory)
    - DBOW (Distributed Bag-of-Words)
3. [GloVe (Global Vectors for Word Representation)](https://aclanthology.org/D14-1162/)

## 环境依赖

### torch版本
- Python 3.11.11
- torch==2.8.0+cu128
- numpy==1.26.4
### gensim版本
- geopandas==0.14.3
- gensim==4.3.2
- momepy==0.6.0
- networkx==3.2.1
- osmnx==1.9.1

## 快速开始

### 1.准备语料库

将**语料库**文件放入**dataset目录**，默认使用**地理语料库_taz.csv**。
语料库格式要求：
- CSV文件包含**Doc**列。
- 每一行是一个文档，内容应为分词后的列表形式。
- 示例数据如下。
```
eID	Doc
254.0	[公司, 公司企业, 公司, 住宅区, 住宅区, 休闲场所, 科教文化场所, 政府及社会团体...]
258.0	[体育休闲服务场所, 科教文化场所, 运动场馆, 综合市场, 专卖店, 宾馆酒店, 服装鞋帽...]
239.0	[公共厕所, 旅行社, 旅行社, 公司, 生活服务场所, 便民商店/便利店, 糕饼店, 公司...]
255.0	[汽车维修, 服装鞋帽皮具店, 购物相关场所, 家居建材市场, 住宅区, 科教文化场所, 专...]
253.0	[公检法机构, 公检法机构, 公检法机构, 工商税务机构, 公检法机构, 商务住宅相关, 住...]
...	...
977.0	[公园广场, 公园广场]
1515.0	[公共厕所, 购物相关场所]
1439.0	[产业园区, 公司, 公司, 公司企业, 汽车维修, 汽车养护/装饰, 公司企业, 金融保险...]
1628.0	[公共厕所, 公共厕所]
1644.0	[公司, 公司企业]
```

### 2.配置模型参数
可通过修改**configs目录**下的**config.json**配置文件设置**模型参数**，主要参数包括：

| 参数名称        | 说明                                      |
|-----------------|-------------------------------------------|
| model_name      | 模型名称 (word2vec/doc2vec/glove)         |
| model_type      | 模型类型 (如word2vec可选cbow/skipgram) |
| docs_path       | 语料库文件路径                            |
| word_embedding  | 嵌入维度                                  |
| min_count       | 词频阈值，低于此值的词将被忽略            |
| window_size     | 上下文窗口大小                            |
| neg_sample_num  | 负采样数量，用于模型训练中的负例样本选取 (仅word2vec的skipgram模型可选)     |
| epochs          | 训练轮数                                  |
| batch_size      | 批次大小                                  |
| lr              | 学习率                                    |
| save_models     | 模型保存目录                              |
| get_embeddings  | 嵌入结果保存目录                          |

### 3. 启动训练

直接运行训练脚本：
```
python train.py
```

## 代码说明

### 数据处理
**data/dataset.py**提供了完整的数据处理流程：
- BuildVocab: 构建词汇表。
- BuildCoMatrix: 构建共现矩阵(用于GloVe)。
- 数据集类: Word2VecDataset和Doc2VecDataset分别为不同模型提供数据加载。

### 模型实现
- **models/word2vec.py**: 实现CBOW和SkipGram模型。
- **models/doc2vec.py**: 实现DM和DBOW模型。
- **models/glove.py**: 实现GloVe模型。

### 训练逻辑
**trainers/trainer.py**中的**Trainer类**封装了完整的训练流程：
- 自动根据模型类型选择数据加载方式
- 实现了不同模型的训练循环和损失计算
- 自动保存最佳模型和嵌入结果

### 结果输出
训练完成后，将生成两类文件：
1. **模型文件**：保存于**checkpoints目录**，格式为{model_name}_{model_type}.pt。
2. **嵌入结果**：保存于**embeddings目录**，包括: 
    - 词嵌入文件: {model_name}_{model_type}_word.csv。
    - 文档嵌入文件 (如适用): {model_name}_{model_type}_doc.csv。

## 自定义训练
如需使用自定义参数训练模型，可修改**train.py**中的**configs/config.json**，例如：
```
config = {
    "model_name": "doc2vec",
    "model_type": "dm",
    "docs_path": "dataset/自定义语料库.csv",
    "word_embedding": 100,
    "epochs": 50,
    "batch_size": 256,
    # 其他参数...
}
```