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
[Text]
```
<span style="background-color: #FFFEFE">ブロッコリー</span> <span style="background-color: #FFF8F8">を</span> <span style="background-color: #FFFEFE">使っ</span> <span style="background-color: #FFFEFE">た</span> <span style="background-color: #FFF9F9">美味しい</span> <span style="background-color: #FFFEFE">おかず</span> <span style="background-color: #FFFEFE">の</span> <span style="background-color: #FFC8C8">レシピ</span> <span style="background-color: #FFF9F9">を</span> <span style="background-color: #FFC8C8">教え</span> <span style="background-color: #FFFEFE">て</span> <span style="background-color: #FFB2B2">ください</span> <span style="background-color: #FFFEFE">。</span> <span style="background-color: #FFFEFE">どうぞ</span> <span style="background-color: #FFFEFE">、</span> <span style="background-color: #FFEDED">参考</span> <span style="background-color: #FFEAEA">です</span> <span style="background-color: #FFFEFE">・</span> <span style="background-color: #FFF5F5">・・・・・・・・・・・・・・・・・・・・・</span>
```
[Label] Yahoo!知恵袋
[Predict] Yahoo!知恵袋
```

* **Example2**
```
[Text]
```
<span style="background-color: #FFFDFD">「</span> <span style="background-color: #FFFEFE">この</span> <span style="background-color: #FFFBFB">回答</span> <span style="background-color: #FFFEFE">者</span> <span style="background-color: #FFFEFE">の</span> <span style="background-color: #FFFEFE">ほか</span> <span style="background-color: #FFFEFE">の</span> <span style="background-color: #FFFBFB">回答</span> <span style="background-color: #FFFDFD">を</span> <span style="background-color: #FFFEFE">見る</span> <span style="background-color: #FFFEFE">」</span> <span style="background-color: #FFFEFE">という</span> <span style="background-color: #FFFEFE">の</span> <span style="background-color: #FFFEFE">は</span> <span style="background-color: #FFFEFE">、</span> <span style="background-color: #FFD8D8">ヤフー</span> <span style="background-color: #FFFEFE">によって</span> <span style="background-color: #FFFDFD">選ば</span> <span style="background-color: #FFFEFE">れる</span> <span style="background-color: #FFFEFE">ん</span> <span style="background-color: #FFDEDE">です</span> <span style="background-color: #FFFDFD">か</span> <span style="background-color: #FFFDFD">？</span> <span style="background-color: #FFFEFE">本人</span> <span style="background-color: #FFFEFE">に</span> <span style="background-color: #FFFEFE">、</span> <span style="background-color: #FFFEFE">他</span> <span style="background-color: #FFFEFE">の</span> <span style="background-color: #FFFEFE">人</span> <span style="background-color: #FFFEFE">に</span> <span style="background-color: #FFC0C0">見せ</span> <span style="background-color: #FFFEFE">て</span> <span style="background-color: #FFFEFE">いい</span> <span style="background-color: #FFFDFD">か</span> <span style="background-color: #FFFEFE">、</span> <span style="background-color: #FFFEFE">連絡</span> <span style="background-color: #FFFEFE">は</span> <span style="background-color: #FFFEFE">いく</span> <span style="background-color: #FFFEFE">ん</span> <span style="background-color: #FFE0E0">です</span> <span style="background-color: #FFFDFD">か</span> <span style="background-color: #FFFDFD">？</span> <span style="background-color: #FFFEFE">連絡</span> <span style="background-color: #FFFEFE">は</span> <span style="background-color: #FFFEFE">来</span> <span style="background-color: #FFFBFB">ない</span> <span style="background-color: #FFE0E0">です</span> <span style="background-color: #FFFEFE">よ</span> <span style="background-color: #FFFEFE">。</span> <span style="background-color: #FFFDFD">でも</span> <span style="background-color: #FFFDFD">まぁ</span> <span style="background-color: #FFFEFE">変</span> <span style="background-color: #FFFEFE">な</span> <span style="background-color: #FFF2F2">回答</span> <span style="background-color: #FFFEFE">し</span> <span style="background-color: #FFFEFE">てる</span> <span style="background-color: #FFFEFE">わけ</span> <span style="background-color: #FFFDFD">じゃ</span> <span style="background-color: #FFFCFC">ない</span> <span style="background-color: #FFFEFE">し</span> <span style="background-color: #FFFCFC">いっか</span> <span style="background-color: #FFFCFC">〜</span>
```
[Label] Yahoo!知恵袋
[Predict] Yahoo!ブログ
```

* **Example3**
```
[Text]
```
<span style="background-color: #FFFEFE">ｋｉｎｋｉ</span> <span style="background-color: #FFFEFE">が</span> <span style="background-color: #FFF4F4">ツアー</span> <span style="background-color: #FFFEFE">する</span> <span style="background-color: #FFFEFE">よ</span> <span style="background-color: #FFF4F4">！</span> <span style="background-color: #FFEEEE">！！！！！！！！</span> <span style="background-color: #FFF8F8">姫</span> <span style="background-color: #FFFEFE">が</span> <span style="background-color: #FFF6F6">住ん</span> <span style="background-color: #FFECEC">でる</span> <span style="background-color: #FFFDFD">大分</span> <span style="background-color: #FFFEFE">県</span> <span style="background-color: #FFFEFE">に</span> <span style="background-color: #FFFEFE">も</span> <span style="background-color: #FFFDFD">くる</span> <span style="background-color: #FFFEFE">よ</span> <span style="background-color: #FFFDFD">！</span> <span style="background-color: #FF8080">！！！！！！</span> <span style="background-color: #FFFCFC">ファン</span> <span style="background-color: #FFFEFE">クラ</span> <span style="background-color: #FFFEFE">入っ</span> <span style="background-color: #FFFEFE">て</span> <span style="background-color: #FFFCFC">ゲット</span> <span style="background-color: #FFF9F9">しよ</span> <span style="background-color: #FFF9F9">！</span> <span style="background-color: #FFF9F9">！</span> <span style="background-color: #FFF9F9">！</span> <span style="background-color: #FFF9F9">！</span> <span style="background-color: #FFFEFE">誰</span> <span style="background-color: #FFF9F9">か</span> <span style="background-color: #FFFEFE">一緒</span> <span style="background-color: #FFFEFE">に</span> <span style="background-color: #FFFEFE">いき</span> <span style="background-color: #FFFEFE">ましょ</span> <span style="background-color: #FFFBFB">！</span> <span style="background-color: #FFFBFB">！</span><br>
```
[Label] Yahoo!ブログ
[Predict] Yahoo!ブログ
```