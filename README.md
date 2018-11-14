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
<p style="background-color: #FFFEFE">ブロッコリー</p> <p style="background-color: #FFF8F8">を</p> <p style="background-color: #FFFEFE">使っ</p> <p style="background-color: #FFFEFE">た</p> <p style="background-color: #FFF9F9">美味しい</p> <p style="background-color: #FFFEFE">おかず</p> <p style="background-color: #FFFEFE">の</p> <p style="background-color: #FFC8C8">レシピ</p> <p style="background-color: #FFF9F9">を</p> <p style="background-color: #FFC8C8">教え</p> <p style="background-color: #FFFEFE">て</p> <p style="background-color: #FFB2B2">ください</p> <p style="background-color: #FFFEFE">。</p> <p style="background-color: #FFFEFE">どうぞ</p> <p style="background-color: #FFFEFE">、</p> <p style="background-color: #FFEDED">参考</p> <p style="background-color: #FFEAEA">です</p> <p style="background-color: #FFFEFE">・</p> <p style="background-color: #FFF5F5">・・・・・・・・・・・・・・・・・・・・・</p>
```
[Label] Yahoo!知恵袋
[Predict] Yahoo!知恵袋
```

* **Example2**
```
[Text]
```
<p style="background-color: #FFFDFD">「</p> <p style="background-color: #FFFEFE">この</p> <p style="background-color: #FFFBFB">回答</p> <p style="background-color: #FFFEFE">者</p> <p style="background-color: #FFFEFE">の</p> <p style="background-color: #FFFEFE">ほか</p> <p style="background-color: #FFFEFE">の</p> <p style="background-color: #FFFBFB">回答</p> <p style="background-color: #FFFDFD">を</p> <p style="background-color: #FFFEFE">見る</p> <p style="background-color: #FFFEFE">」</p> <p style="background-color: #FFFEFE">という</p> <p style="background-color: #FFFEFE">の</p> <p style="background-color: #FFFEFE">は</p> <p style="background-color: #FFFEFE">、</p> <p style="background-color: #FFD8D8">ヤフー</p> <p style="background-color: #FFFEFE">によって</p> <p style="background-color: #FFFDFD">選ば</p> <p style="background-color: #FFFEFE">れる</p> <p style="background-color: #FFFEFE">ん</p> <p style="background-color: #FFDEDE">です</p> <p style="background-color: #FFFDFD">か</p> <p style="background-color: #FFFDFD">？</p> <p style="background-color: #FFFEFE">本人</p> <p style="background-color: #FFFEFE">に</p> <p style="background-color: #FFFEFE">、</p> <p style="background-color: #FFFEFE">他</p> <p style="background-color: #FFFEFE">の</p> <p style="background-color: #FFFEFE">人</p> <p style="background-color: #FFFEFE">に</p> <p style="background-color: #FFC0C0">見せ</p> <p style="background-color: #FFFEFE">て</p> <p style="background-color: #FFFEFE">いい</p> <p style="background-color: #FFFDFD">か</p> <p style="background-color: #FFFEFE">、</p> <p style="background-color: #FFFEFE">連絡</p> <p style="background-color: #FFFEFE">は</p> <p style="background-color: #FFFEFE">いく</p> <p style="background-color: #FFFEFE">ん</p> <p style="background-color: #FFE0E0">です</p> <p style="background-color: #FFFDFD">か</p> <p style="background-color: #FFFDFD">？</p> <p style="background-color: #FFFEFE">連絡</p> <p style="background-color: #FFFEFE">は</p> <p style="background-color: #FFFEFE">来</p> <p style="background-color: #FFFBFB">ない</p> <p style="background-color: #FFE0E0">です</p> <p style="background-color: #FFFEFE">よ</p> <p style="background-color: #FFFEFE">。</p> <p style="background-color: #FFFDFD">でも</p> <p style="background-color: #FFFDFD">まぁ</p> <p style="background-color: #FFFEFE">変</p> <p style="background-color: #FFFEFE">な</p> <p style="background-color: #FFF2F2">回答</p> <p style="background-color: #FFFEFE">し</p> <p style="background-color: #FFFEFE">てる</p> <p style="background-color: #FFFEFE">わけ</p> <p style="background-color: #FFFDFD">じゃ</p> <p style="background-color: #FFFCFC">ない</p> <p style="background-color: #FFFEFE">し</p> <p style="background-color: #FFFCFC">いっか</p> <p style="background-color: #FFFCFC">〜</p>
```
[Label] Yahoo!知恵袋
[Predict] Yahoo!ブログ
```

* **Example3**
```
[Text]
```
<p style="background-color: #FFFEFE">ｋｉｎｋｉ</p> <p style="background-color: #FFFEFE">が</p> <p style="background-color: #FFF4F4">ツアー</p> <p style="background-color: #FFFEFE">する</p> <p style="background-color: #FFFEFE">よ</p> <p style="background-color: #FFF4F4">！</p> <p style="background-color: #FFEEEE">！！！！！！！！</p> <p style="background-color: #FFF8F8">姫</p> <p style="background-color: #FFFEFE">が</p> <p style="background-color: #FFF6F6">住ん</p> <p style="background-color: #FFECEC">でる</p> <p style="background-color: #FFFDFD">大分</p> <p style="background-color: #FFFEFE">県</p> <p style="background-color: #FFFEFE">に</p> <p style="background-color: #FFFEFE">も</p> <p style="background-color: #FFFDFD">くる</p> <p style="background-color: #FFFEFE">よ</p> <p style="background-color: #FFFDFD">！</p> <p style="background-color: #FF8080">！！！！！！</p> <p style="background-color: #FFFCFC">ファン</p> <p style="background-color: #FFFEFE">クラ</p> <p style="background-color: #FFFEFE">入っ</p> <p style="background-color: #FFFEFE">て</p> <p style="background-color: #FFFCFC">ゲット</p> <p style="background-color: #FFF9F9">しよ</p> <p style="background-color: #FFF9F9">！</p> <p style="background-color: #FFF9F9">！</p> <p style="background-color: #FFF9F9">！</p> <p style="background-color: #FFF9F9">！</p> <p style="background-color: #FFFEFE">誰</p> <p style="background-color: #FFF9F9">か</p> <p style="background-color: #FFFEFE">一緒</p> <p style="background-color: #FFFEFE">に</p> <p style="background-color: #FFFEFE">いき</p> <p style="background-color: #FFFEFE">ましょ</p> <p style="background-color: #FFFBFB">！</p> <p style="background-color: #FFFBFB">！</p><br>
```
[Label] Yahoo!ブログ
[Predict] Yahoo!ブログ
```