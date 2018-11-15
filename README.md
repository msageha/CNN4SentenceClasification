# SentenceClasification
Sentence Classification on pytorch

## Introduction
研究で，文，あるいは文書のメディア依存性を図るためのテスト．
可視化したかったため，[A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130.pdf)の論文を参考に一部簡略化．

実装予定 Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.

## Requirement
* python 3.6
* pytorch > 0.4
* torchtext > 0.1
* numpy
* pandas

## Result
[BCCWJコーパス](https://pj.ninjal.ac.jp/corpus_center/bccwj/)にて，
* Yahoo!知恵袋
* Yahoo!ブログ
* 白書
* 書籍
* 雑誌
* 新聞
の6クラス分類．

データの内訳は以下のような感じ．
|media|知恵袋|ブログ|白書|書籍|雑誌|新聞|total|
|---|---|---|---|---|---|---|---|
|sentence|666,518|825,729|137,320|1,271,091|263,863|47,560|3,212,081|
|document|91,445|52,680|1,500|10,117|1,473|159,211|

## Usage
```
python3 main.py -h
```

You will get:

```
text classificer

optional arguments:
  -h, --help            show this help message and exit
  -batch-size N         batch size for training [default: 50]
  -lr LR                initial learning rate [default: 0.01]
  -epochs N             number of epochs for train [default: 10]
  -dropout              the probability for dropout [default: 0.5]
  -log-interval LOG_INTERVAL
                        how many batches to wait before logging training
                        status
```

## Train
```
python3 ./train.py
```

```
Batch[100] - loss: 0.655424  acc: hoge%
Evaluation - loss: 0.672396  acc: hoge%(615/1066) 
```

## Test
If you has construct you test set, you make testing like:

```
python3 test.py -test -model="hogehoge"
```

## Predict

|media|知恵袋|ブログ|白書|書籍|雑誌|新聞|total|
|---|---|---|---|---|---|---|---|
|sentence|---|---|---|---|---|---|---|
|document|86.8|78.0|84.5|66.3|28.4|50.0|

* **Example1**
```
[Label] Yahoo!知恵袋
[Predict] Yahoo!知恵袋
```
![Text](https://i.gyazo.com/fa025bdde243769baa9646c9b25c0934.png)

* **Example2**
```
[Label] Yahoo!知恵袋
[Predict] Yahoo!ブログ
```
![Text](https://i.gyazo.com/0eeeb7243d57b7517d368db35e85fe9a.png)


* **Example3**
```
[Label] Yahoo!ブログ
[Predict] Yahoo!ブログ
```
![Text](https://i.gyazo.com/b38693d63b29a1f4de193fc5fc325ef6.png)