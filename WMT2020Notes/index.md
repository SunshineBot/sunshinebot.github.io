# WMT2020 Notes

本文主要关注WMT2020 News赛道的英中双向翻译，依据人工评价结果选择了火山、微信、滴滴、腾讯、OPPO5篇论文精读，同时粗略读了Findings并选择了一些感兴趣的内容整理出来。

## Data
WMT语料包括平行语料和单语语料，对于中英翻译而言，规模较大的是UNPC和CWMT，分别为1600w和近1000w样本，其它语料规模较小（除了WMT2020新增的Back-Translated News，它只能算作伪平行语料）。Wiki Titles是一些词和短语。

WMT2020相比于之前的比赛，语料变化比较多，更新了2个规模不大的语料，新增了2个语料，其中Back-Translated News规模大但是是BT语料。单语语料更新了News Crawl和News Commentary。
constrained system只能使用提供的平行语料和单语语料，unconstrained system可以使用额外的语料。我们选取的5篇论文中，只有火山是unconstrained system，使用了额外的单语语料用于BT、KD等，其它4篇论文均为constrained system。

- Parallel
    - [Updated]News Commentary v15
    - [Updated]Wiki Titles v2(words and phrases)
    - UN Parallel Corpus V1.0(UNPCv1)
    - CCMT/CWMT corpus
    - [New]WikiMatrix
    - [New]Back-Translated News(produced for Edinburgh System in 2017 and 2018)
- Monolingual
    - [Updated]News Crawl
    - [Updated]News Commentary
    - Common Crawl

数据上比较有趣的一点是，WMT19以前，source和reference中是混合了双向语料的，即对于en->zh而言，测试集的reference包含了原始的英文，也包含了经由中文翻译得到的英文，source同理。常规情况下我们并不对平行语料进行区分，都视为原始的平行语料。但是显而易见的是，直接写的句子和翻译得到的句子差异是非常大的（这一点读过翻译的国外书籍的人我相信都能体会到）。

业界对这个问题似乎没有过多地关注，但从WMT2019开始，平行语料进行了区分（即平行语料的两边并不是完全平等的），分为Native/Original sentences和Translated sentences。Native sentences是直接使用Native Language撰写出来的句子，Translated sentences是人工将Native sentences从Native(Source) Language翻译到到Target Language而得到的句子。

WMT的想法是，现有的翻译模型和评价指标(BLEU)在面对Native->Translated构成的测试集和Translated->Native构成的测试集时表现存在偏差，且差异较大。既然是机器翻译，那么目标还是达到人工翻译的水平，因此从WMT2019开始，会将en->zh和zh->en的测试集分开来，保证**每一个**语种方向的测试集都只包含Native->Translated样本。


## Findings - 人工评价

人工评价离不开直接打分，这种方式被称为DA(Direc Assessment)，早期的DA是给定source和reference，对每一个candidate进行打分，分数在0~100之间，这样的方法被称为Reference-based DA。WMT19曾在en-cz尝试过改进后的Source-based DA，不依赖reference，只给定source，对每一个candidate进行打分，好处是可以对reference进行打分，用于对比模型和人工翻译的质量。在WMT2020中，所有的out-of-English翻译均使用Source-based DA，所有的into-English翻译均使用reference-based DA(因为非英语语种为native language并且英语很熟练的人较多，反之较少)。
打分时还会考虑到上下文(Document Context, DC)的含义，+DC表示评估人员在打分时可以看到source句子所在的段落或文章，-DC表示仅提供source句子。

对于Chinese <-> English，最终的打分方式为SR+DC(Segment Ranking DA + DC)，然后将每个人的打分的分数按照评价者给分的均值和标准差进行标准化(Ave. z)，作为最终的分数进行排名。同时也提供了未经标准化的分数(Ave.)。

人工评价结果如下，其中Online-X表示选取的在线翻译服务的结果。

**Chinese to English:**

|Ave.|Ave. z|System|
|---|---|---|
|77.5|0.102|**VolcTrans**|
|77.6|0.089|**DiDi-NLP**|
|77.4|0.077|**WeChat-AI**|
|76.7|0.063|**Tencent-Translation**|
|77.8|0.060|Online-B|
|78.0|0.051|DeepMind|
|77.5|0.051|**OPPO**|
|76.5|0.028|THUNLP|
|76.0|0.016|SJTU-NICT|
|72.4|0.000|Huawei-TSC|
|76.1|-0.017|Online-A|
|74.8|-0.029|HUMAN|
|71.7|-0.071|Online-G|
|74.7|-0.078|dong-nmt|
|72.2|-0.106|zlabs-nlp|
|72.6|-0.135|Online-Z|
|67.3|-0.333|WMTBiomedBaseline|

**English to Chinese:**

|Ave.|Ave. z|System|
|---|---|---|
|80.6|0.568|HUMAN-B|
|82.5|0.529|HUMAN-A|
|80.0|0.447|**OPPO**|
|79.0|0.420|**Tencent-Translation**|
|77.3|0.415|Huawei-TSC|
|77.4|0.404|NiuTrans|
|77.7|0.387|SJTU-NICT|
|76.6|0.373|**VolcTrans**|
|73.7|0.282|Online-B|
|73.0|0.241|Online-A|
|69.5|0.136|dong-nmt|
|68.5|0.135|Online-Z|
|70.1|0.122|Online-G|
|68.7|0.082|zlabs-nlp|


## General Methods

### Multiple Models

所有论文均以Transformer Big为主要模型，几乎都使用了多种模型来做Ensemble(除了OPPO，他们只使用了Transformer Big)。其中使用较多的有：

- Wide Transformer(4/4): 增加FFN size，一般为8192或15000，同时调大Dropout（一般0.3, rule dropout不一定）
- Deep Transformer(3/4): 增加了Encoder的层数，15~50层不等，太深的模型一般会将hidden size和ffn size调小或者使用base模型

其它用到的一些模型：

- transformer 128hdim/256hdim(VolcTrans): 增加了Multi-head Attention中单个Head的维度
- DLCL 25layers(VolcTrans): 小牛翻译在WMT19提出的模型的改进版
- Dynamic Conv 7e6d/25e6d(VolcTrans): 卷积网络
- Average Attention Transformer(WeChat): 将Decoder中的self-attention替换为对embedding的Average操作，略微降低效果的情况下加快推断
- DTMT(WeChat): RNN-based NMT Model
- Transformer With Relative Attention(DiDi)
- Transformer with reversed source(DiDi)
- Hybrid Transformer(Tencent): 35 self-attention encoder + 5 ON-LSTM encoder

尽管CNN和RNN-based模型也有使用，但是效果相比transformer模型有差距，但是不同的模型结构对于Ensemble是有益处的，这点在VolcTrans(使用了CNN)和WeChat(使用了RNN)都有提及。
VolcTrans原话大致是不在乎单个模型效果的好坏，更关注模型之间的差异。
微信尝试了很多finetune策略，每一个finetune后的模型都作为single model参与Ensemble，最终发现来自类Transformer模型（含deep、wide等）的翻译结果更相似，不利于Ensemble，进而提出了self-BLEU的方法用来挑选多样性较大的模型用于Ensemble。

### Data Preprocess

#### Data Preprocess

预处理流程上，BPE和truecase基本成为标配。
英文tokenizer均使用了Moses，中文分词各家不同，VolcTrans和Tencent使用了jieba，OPPO使用了pkuseg，WeChat、DiDi都使用了自研的分词工具。
其它预处理包括全半角转换、Unicode字符转换、HTML tag转换、标点符号正则化、不同的空格（\u3000等）

#### Data Filter

VolcTrans没有提到过滤具体的过滤策略，其它都有提及。比较常用的方法：
- 去掉太长的句子（超过100~140个token）
- 去掉包含长word的句子（单个word长度大于40不等）
- source和target长度比例控制在1:3~3:1之间
- fast align

个别论文使用的方法：
- 语言检测（fastText）
- LM score
- character-word ratio >1.5 <12
- edit distance

#### Data Sharding

多样性对Ensemble来说至关重要，除了模型的多样性，数据多样性得到了3篇论文的提及（VolcTrans/WeChat/DiDi）。总的来说，在整个机器翻译流程中，可以从3个角度引入数据多样性。
第一是对原始训练数据引入多样性，一般来说是采样（VolcTrans，对全量数据进行放回采样，比例为100%、110%、120%）或者分片（微信，分成3部分）；第二是对BT使用的单语数据进行引入多样性，通常可以直接分片，然后用于生成BT数据（VolcTrans）或者训练LM等无监督模型（DiDi）；
第三是对训练数据引入噪声，例如对source端数据进行增删改（微信）、在BT模型生成source数据时使用采样代替beam search（微信）

### Procedures

常规的训练步骤可以分为以下几步：
1. data preprocess/filter/sharding
2. train baselines
3. **out-of-domain methods**: BT, KD with(out) ensemble
4. **in-domain methods**: finetune, iterative joint training, etc
5. other methods
6. ensemble multi models
7. reranking

BT和KD作为简单有效的做法， 已经得到公认（除了Tencent，他们的BT策略在中英双向翻译上没有提升，因此放弃，但在英德上使用了BT）。
比赛的测试集newstest是新闻领域的语料，与训练集存在领域差异。因此，所有论文都使用以前的WMT测试集作为in-domain语料，对模型的领域适应性进行调整。
调整方法包括直接finetune、结合BT or KD一起训练、使用基于in-domain语料的分类器/LM来对平行语料进行过滤等。
大多数论文都使用了迭代式的训练方法（除了Tencent），区别在于是在out-of-domain阶段对BT/KD使用，还是在in-domain阶段。

比较值得注意的是，之前我并没有想明白单模型如何进行KD，后来DiDi给出了答案：对于每一个单模型，使用其它模型的ensemble作为teacher。

总结下来，BT、KD、in-domain finetune是比较公认和值得尝试的做法，迭代式训练方法可以视模型效果和资源决定。

#### Out-of-domain methods

Out-of-domain
## VolcTrans

### Overview: **Universal** methods for all translation model

### Multi Models
- Transformer 15e6d(deeper transformer)
- Transformer Mid 25e6d/50e6d: ffn size 4096->3072, embedding size 1024->768
- Transformer 15000FFN: performance of the Transformer model is largely dependent on the dimensions of feed forward network. Dropout 0.1->0.3, rule dropout 0.1->0.3
- Transformer 128hdim/256hdim: increase attention head dimension
- DLCL 25layers: deep transformer + DLCL
- Dynamic Conv 7e6d
- Dynamic Conv 25e6d

### Strategies
- unconstrained - which means trained with non-wmt20's monolingual data.
- Parallel Data Up-Sampling
    - Reason: data diversity matters for the whole system.
    - each model is trained with different part of parallel corpus.
- mRASP: universal pretrained model for low-resource languages.
- Tag BT
    - Different models decode different parts of monolingual data(10M/part).
- Iterative Joint Training
    - Iterative trained with BT data.
- KD & Ensemble
    - Ensemble Model: devide 9 models into 3 groups, employ models in one group as teacher models.
    - R2L Model: train one R2L model for each group(using the same data as any one model in last iteration).
    - Use pseudo parallel data from Ensemble/R2L model to train the student model w/o parallel data.
- Others
    - top-k checkpoint average(the same with ours).
    - Random Ensemble: select candidates from top-k checkpoints instead of from best checkpoint.
    - In domain finetuning: finetune the best single model with dev set for 1-2 epochs.

### Procedures(for en-zh)
- preprocess and sampling
    - ModesTokenizer/Jieba, BPE num_ops=32000.
    - upsampling 100%/110%/120%.
- model training
    - train 3 models(Mid 25e6d/Mid 50e6d/ Conv 25e6d) with 3 upsampled data, resulting in 9 baseline models.
    - monolingual data: NewsCrawl for en, NewsCrawl/CCMT/LDC for zh.
    - tag bt & iterative
        - en: disjoint data for each baseline model(90M for 9 models).
        - zh: use each part of data for 3 times(there're only 24M zh data).
    - KD: employ disjoint monolingual data as distilling data.
    - Final: ensemble 9 models.

### Results


## WeChat

### Overveiw: Push single model to best **in-domain performance**

### Multi Models
- Deeper Transformer: 30e6d(base), 20/24e6d(big)
- Wider Transformer: 10e6d15000FFN, 12e6d12288FFN
- AAN: Average Attention Transformer
- DTMT: RNN-based NMT model(train very slowly)

### Strategies
- Data filter: punctuation normalization, filter out long sentences/long words/duplicated sentences/word ratio
- Out-of-domain synthesis strategies - 2~4 BLEU promotion
    - BT: on monolingual corpus with L2R & R2L model
    - KD: on parallel corpus with L2R & R2L model
- In-domain synthesis strategies
    - In-domain finetuning: finetune on WMT17/18 tests for 400 steps - 5~7 BLEU promotion
    - Iterative In-domain Knowledge Transfer - 2~3 BLEU promotion
        - finetune the out-of-domain model(with BT&KD) with in-domain finetuning strategy
        - ensemble models and translate Chinese monolingual corpus into English to get pseudo in-domain parallel corpus
        - retrain models with both pseudo in-domain parallel corpus and original out-of-domain parallel corpus
        - procedures above can be conducted iteratively(0.1~0.4 BLEU for second iteration)
    - Parallel Scheduleed Sampling
        - first pass: generate predictions
        - second pass: feed mixture of reference's and prediction's tokens
    - Target Denoising: add noise to 30% sentences, in which replace a token with a random token from the sentence with  percentage of 15%(totally 4.5%)
    - Minimum Risk Training instead of CrossEntropy loss
- Data Augmentation
    - add token-level noise to source data: replace/delete/permutation(both proba=0.1) - Noisy Data
    - replace beam search with sampling in generation - Sample Data
    - original data - Clean Data
- Advanced Ensemble - 0.7 BLEU
    - Data Shards: split training data into 3 shards among Clean/Noisy/Sample data respectively(totally 9 shards)
    - problem: models are very similar(4 finetuning approaches over each model, totally 200+ models)
    - Self BLEU: using candidates as references to get BLEU from each other, ensemble 20 models with low BLEU scores

### Results


## DiDi

### Overview

### Multi Models
- Transformer Big
- Transformer with Relative Attention
- Transformer 8192FFN/15000FFN(dropout: 0.1->0.3, label smoothing: ->0.2)
- Transformer with reversed source

### Strategies
- Data filtering
    - common: normalize punctuation/long sentences(>120)/long words(>40)/length ratio(1:3).
    - others: HTML tags/Language detect(fastText)/fast-align/LM score.
    - feed src and ref into BERT and CNN and filter sentences.
- BT
    - decoding strategy: greedy/beam/sampling top-k/add noise(to input/beam output)
        - obtain imporvements: add noise to input + beam search.
        - delete/replace/swap tokens with the proba of 0.05.
    - iterative joint training
        - iteratively train t2s and s2t model to generate data for s2t and t2s model.
    - train LM and split data in different domains
        - train LM on different monolingual data(NewsCrawl, Gigaword, etc) and score both parallel and synthetic sentences.
        - train different models on different shards of parallel and synthetic data.
- KD & Ensemble - 1.5 BLEU(together with BT)
    - for each model, ensemble other models as teacher model.
    - if teacher model perform worse than student model, skip KD.
- In-domain data selection and finetuning - 10 BLEU
    - employ (sampled) parallel data as out-of-domain data, test data as in-domain data
    - N-grams: train 2 LM and filter the parallel data.
    - Binary Classification: finetune BERT classifier on the in-domain and out-of-domain data
- Ensemble - 0.4 BLEU
    - method: combine the full probability distribution over the target vocab of different models at each step.
        - log-avg achieves best performance among max/avg/log-avg strategies.
        - greedy search instead of beam search due to limits of computer resources.
    - different models are trained with different: random seeds/parameters/architectures/training data.
- Domain Style Translation - 1 BLEU
    - translations differs in different domains:
        - at least data can be devide into 2 domains: native style and translation style
    - procedures
        - use pretrained BERT and k-means to get 2 clusters(2 domains)
        - finetune BERT on data of 2 domains above
        - in predicting, multiply the output with the domain probability.
- Re-ranking - 0.5 BLEU
    - k-best MIRA
    - features: Length Feature(between src and hyp)/NMT Feature(score of NMT)/LM Features(scores of LM)

### Results


## Tencent

### Overview: **BEST** zh-en Baselines

### Multi Models(both use pre-norm)
- Deep Transformer: 40e(base)
- Hybrid Transformer: 35 self-attention encoder + 5 ON-LSTM encoder
- BigDeep Transformer: 20e(big)
- Large Transformer: 8192FFN based on BigDeep Transformer

### Strategies
- Data filter: langid/deplication/length(>150, 1:1.3)/invalid string/edit distance
- Data Augmentation
    - BT: **Only in English-German** due to lower translation quality in English-Chinese
        - ensemble of 2 models
        - add noise to source
        - add special tag at the head of synthetic source
        - filter out examples with BLEU <30
    - R2L training: translate source sentences with both L2R and R2L, filter out BLEU<15
- Finetuning
    - Generally: use WMT2017dev/WMT2017test/WMT2018test as in-domain corpus, finetune with small batch size for several thousands of steps
    - for zh-en: addtional WMT2019test, using R2L model(boost), batch size 2048, train 3k steps
    - for en-zh: reset optimizer, learning rate 8e-5, batch size 1024, train 900 steps
- re-ranking
    - source-to-target model: ensemble 4 models with beam size of 25
    - target-to-source model: translate candidates back to source, using big transformer
    - LM: train a small GPT-2(FFN8192) on target monolingual data.
    - length penalty: random search in range [0, 3)
- Ensemble
    - Greedy based ensemble: get 4 best single models, add other model which can benefit the translation performance
    - Iterative Transductive Ensemble: translate ensembly def/test set to get pesudo data and finetune model on it, iteratively

### Results

## OPPO

### Overview

### Model: Transformer Big

### Strategies
- Data filter
    - common: html tags/spaces/punctuations/long sentences/long words/length ratio/fastalign
    - special: charactar-word count ratio >1.5<12
- Data preprocess
    - Convert Traditional Chinese to Simplified Chinese
    - Convert Latin letters/digit chars/punctuations, some to full width and others to half width.
    - Chinese tokenizer: pkuseg
- BT: Iterative BT
- Finetune:
    - finetune on parallel data after BT
    - finetune on newstest2017
- Ensemble
- Re-ranking
    - forward scorer(source-to-target)
    - backward scorer(target-to-source)
    - LM socrer(LM)
- Entity substitution
    - get entity mapping with Standford NLP NER tools
    - replace the entity with <tag1>, <tag2>, etc and recover it with post-edit.

### Results

## Summary
- BT: promote data diversity
    - split data into different shards


Back to [Index](../index).
