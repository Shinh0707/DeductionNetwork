# DeductionNetwork

---

このリポジトリは、Deduction Network（推論ネットワーク）に関連するPyTorchモデルを提供します。以下では、各クラスの概要とそれぞれのモデルの構造、そしてモデルの使用方法について説明します。

## KeyPair

KeyPairクラスは、入力されたキーと値のペアに対して、それらを結合して埋め込みベクトルを生成するためのモジュールです。

### 仕組み

このクラスは、キーと値を連結し、線形変換とドロップアウトを行い、それにランダム化されたLeaky ReLU非線形活性化関数を適用します。

### パラメータ

- `kdim`：キーの次元
- `vdim`：値の次元
- `embed_dim`：埋め込み次元（デフォルトではキーと値の次元の合計）
- `bias`：バイアスの有無
- `dropout`：ドロップアウト率

## FeedForwardNetwork

FeedForwardNetworkクラスは、入力されたベクトルに対して、線形変換と活性化関数を適用するモジュールです。[Attension Is All You Need.](https://arxiv.org/abs/1706.03762)で示されていた構造を元に活性化関数をReLUではなくLeakyReLUを用いて構成した。

### 仕組み

このモジュールは、入力されたベクトルに対して、線形変換、Leaky ReLU活性化関数、そしてドロップアウトを適用します。

### パラメータ

- `in_dim`：入力の次元
- `hid_dim`：隠れ層の次元（デフォルトは入力の2倍）
- `out_dim`：出力の次元（デフォルトは入力の次元）
- `dropout`：ドロップアウト率

## DeductionNetworkPartialLayer

DeductionNetworkPartialLayerクラスは、推論ネットワークの部分的なレイヤーを定義します。

### 仕組み

このクラスは、KeyPairとFeedForwardNetworkを組み合わせて、注意機構を持つレイヤーを実装します。具体的には、入力されたキーとクエリに対してアテンション機構を適用し、その後、残差接続とレイヤーノーマライゼーションを行います。

### パラメータ

- `embeded_dim`：埋め込み次元
- `vdim`：値の次元
- `num_heads`：アテンションヘッドの数
- `dropout`：ドロップアウト率

## CDeductionNetworkPartialLayer

CDeductionNetworkPartialLayerクラスは、推論ネットワークの部分レイヤーを定義します。

### 仕組み

このクラスは、複数のDeductionNetworkPartialLayerを組み合わせて、クエリ、キー、値の推論を行います。

### パラメータ

- `qdim`：クエリの次元
- `kdim`：キーの次元
- `vdim`：値の次元
- `embed_dim`：埋め込み次元
- `num_heads`：アテンションヘッドの数
- `bias`：バイアスの有無
- `dropout`：ドロップアウト率

## DeductionNetworkLayer

DeductionNetworkLayerクラスは、推論ネットワークのレイヤーを定義します。

### 仕組み

このクラスは、CDeductionNetworkPartialLayerを組み合わせて、クエリ、キー、値の推論を複数層にわたって行います。

### パラメータ

- `qdim`：クエリの次元
- `kdim`：キーの次元
- `vdim`：値の次元
- `embed_dim`：埋め込み次元
- `num_heads`：アテンションヘッドの数
- `bias`：バイアスの有無
- `dropout`：ドロップアウト率

## DeductionNetwork

DeductionNetworkクラスは、推論ネットワーク全体を定義します。

### 仕組み

このクラスは、複数のDeductionNetworkLayerを組み合わせて、クエリ、キー、値の推論を複数層にわたって行います。

### パラメータ

- `qdim`：クエリの次元
- `kdim`：キーの次元
- `vdim`：値の次元
- `embed_dim`：埋め込み次元
- `table_size`：内部パラメータのテーブルサイズ
- `num_layers`：レイヤーの数
- `num_heads`：アテンションヘッドの数
- `bias`：バイアスの有無
- `dropout`：ドロップアウト率
- `device`：デバイス
- `dtype`：データタイプ
