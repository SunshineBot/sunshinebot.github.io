# Paper Note: Pretrained Model for GEC

读了两篇和GEC以及语言模型相关的论文。

## Stronger Baselines for Grammatical Error Correction Using Pretrained Encoder–Decoder Model

作者：
- Satoru Katsumata(Tokyo Metropolitan University)
- Mamoru Komachi(Tokyo Metropolitan University)

### 论文思路

这篇论文用MASS和BART两个预训练语言模型作为预训练模型，在其基础上finetune得到GEC模型，评价其性能。
本以为作者进行了模型结果或者其它的改动，因为其参考的两个baseline中，第二个baseline对模型结构进行了大刀阔斧的改动（BERT-fuse mask/GED）。然而意外的是，作者完全没有改动模型，而是直接使用训练好的语言模型来finetune。这样其实就是完全在比拼预训练模型本身的效果了。这就让我很好奇：用于多个任务的泛化型预训练语言模型，为何能达到使用精心设计的语料（合成语料）或模型（BT）生成的语料来专门pretrain的预训练模型呢？单纯是因为海量的语料带来的提升吗（BART使用了160GB语料，大约是10亿量级的sentences来pretrain）

## BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

整体思路：构建了一个基于transformer的seq2seq预训练语言模型，对训练数据做了很多噪声处理。

### 具体思路
#### 模型结构

用的是与Transformer相近的模型，区别在于

- 使用GeLU替代ReLU
- decoder部分与transformer的decoder相同（加上了enc-dec attention，这是BERT和GPT都不具备的）
- BERT在softmax预测前加入了隐层，BART没有隐层。

#### Pretrain：给数据引入噪声

- Token Masking: randomly sample tokens and replace with [MASK]
- Token Deletion: randomly delete tokens
- Text Infilling: ramdomly sample spans with different lengths(sample from Poisson distribution with λ=3) and replace with [MASK]
- Sentence Permutation: split sentences and shuffle them
- Document Rotation: randomly choose a token and then rotate the document so that it begins with the token chosen.

#### Finetune

- 序列分类：decoder最后一个输出
- token分类：decoder输出
- sequence generation
- machine translation：add a small randomly initialized Encoder

## 小结

回顾这个BART这个语言模型，我觉得标题中的*Denoising*这个操作可能是关键所在。Grammatical Error其实也是Noise的一种。作者在训练BART时，不仅考虑了模型，还考虑如何降低输入到encoder中的数据中的噪声，这与GEC模型的pretrain目标不谋而合（只不过噪声分布存在差异），让BART模型更适用于GEC任务。

不过这样的结论需要更多的实验来验证，例如其它的预训练语言模型是不是没有对输入进行比较复杂的处理？性能更高但是没有对输入引入噪声的预训练语言模型，用来finetune GEC模型性能是不是会更差？


Back to [Blog Index](../index)
