# Deduction Network (演驛ネットワーク)

## 事前定義

- 連結 (Concatenate)
  ${(X,Y)=Z\quad\{X\in\R^{M\times{f_X}},Y\in\R^{M\times{f_Y}},Z\in\R^{M\times(f_X+f_Y)}\}}$
- 線形変換
  ${L(X,f_O)=X\cdot{W^T}+b\quad\{X\in\R^{d_X\times{f_X}},W\in\R^{f_O\times{f_X}},b\in\R^{f_O}\}}$
- [フィードフォワード (Position-wise Feed-Forward Networks)](https://arxiv.org/abs/1706.03762)
  ${FF(X,f_O)=L(\text{ReLU}(L(X,2f_X)),f_O)\quad\{X\in\R^{d_X\times{f_X}}\}}$
  ${FF(X)=FF(X,f_X)\quad\{X\in\R^{d_X\times{f_X}}\}}$
- シングルエンコーダー
  ${SENC(X,f_h,f_O)=\text{Tanh}(L(L(X,f_h),f_O))\quad\{X\in\R^{d_X\times{f_X}}\}}$
  ${SENC(X,f_O)=SENC(X,2f_X,f_O)\quad\{X\in\R^{d_X\times{f_X}}\}}$
  ${SENC(X)=SENC(X,2f_X,f_X)\quad\{X\in\R^{d_X\times{f_X}}\}}$
- ペアエンコーダー
  ${PENC(X,Y,f_h,f_O)=SENC((X,Y),f_h,f_O)\quad\{X\in\R^{M\times{f_X}},Y\in\R^{M\times{f_Y}}\}}$
  ${PENC(X,Y,f_O)=SENC((X,Y),f_O)\quad\{X\in\R^{M\times{f_X}},Y\in\R^{M\times{f_Y}}\}}$
  ${PENC(X,Y)=SENC((X,Y))\quad\{X\in\R^{M\times{f_X}},Y\in\R^{M\times{f_Y}}\}}$
- [レイヤー正規化](https://arxiv.org/abs/1607.06450)
  ${\text{LayerNorm}}$と表記
- [マルチヘッドアテンション](https://arxiv.org/abs/1706.03762)
  ${\text{MHA}}$と表記
- [アテンション](https://arxiv.org/abs/1706.03762)
  Scaled Dot-Product Attentionを用いる
  ${\text{ATN}}$と表記

## DeductionNetworkSingleLayer (DedS)

TransformerのPositionalEncodingなしのEncoderを元に作成した。→『Attension Is All You Need』

- 入力
  ${Q\in\R^{S\times{f_q}}}$：Question
  ${H\in\R^{S\times{f_h}}}$：Hint
  ${A\in\R^{S\times{f_a}}}$：Answer
- 出力
  ${R\in\R^{S\times{f_a}}}$：Focused Answer
- 計算
  ${A'=\text{MHA}(Q,H,A)}$
  ${A''=\text{LayerNorm}(A'+\text{ATN}(Q,H,A))}$
  ${R=\text{LayerNorm}(FF(A'')+A'')}$

## DeductionNetworkPairLayer (DedP)

DeductionNetworkSingleLayerを二重に重ねたもの

- 入力
  ${Q\in\R^{S\times{f_q}}}$：Question
  ${H\in\R^{S\times{f_h}}}$：Hint
  ${Q_T\in\R^{S_T\times{f_q}}}$：QuestionTable
  ${H_T\in\R^{S_T\times{f_h}}}$：HintTable
  ${A_T\in\R^{S_T\times{f_a}}}$：AnswerTable
- 出力
  ${R\in\R^{S\times{f_a}}}$：Deduced Answer
- 計算
  ${E_{QH}=\text{PENC}(Q,H)}$
  ${E_{QHT}=\text{PENC}(Q_T,H_T)}$
  ${R=\text{DedS}(Q,H,\text{DedS}(E_{QH},E_{QHT},A_T))}$

## DeductionNetworkLayer (DedNL)

DedPを複数回適用できるようにしたもの。出力結果をDedNに再度入力することで推定結果が良くなるのではないかという考えのもと作成した。

- 入力
  ${Q\in\R^{S\times{f_q}}}$：Question
  ${H\in\R^{S\times{f_h}}}$：Hint
  ${A\in\R^{S\times{f_a}}}$：Answer
  ${Q_T\in\R^{S_T\times{f_q}}}$：QuestionTable
  ${H_T\in\R^{S_T\times{f_h}}}$：HintTable
  ${A_T\in\R^{S_T\times{f_a}}}$：AnswerTable
- 出力
  ${R_Q\in\R^{S\times{f_q}}}$：Deduced Question
  ${R_H\in\R^{S\times{f_h}}}$：Deduced Hint
  ${R_A\in\R^{S\times{f_a}}}$：Deduced Answer
- 計算
  ${R_Q=\text{DedP}(H,A,H_T,A_T,Q_T)}$
  ${R_H=\text{DedP}(Q,A,Q_T,A_T,H_T)}$
  ${R_A=\text{DedP}(Q,H,Q_T', H_T', A_T')}$

## DeductionNetworkStartLayer (DedSt)

入力で足りない部分をテーブルと入力を元に補完する

- 入力
  ${Q\in\R^{S\times{f_q}}\ \text{or}\ \text{None}}$：Question
  ${H\in\R^{S\times{f_h}}\ \text{or}\ \text{None}}$：Hint
  ${A\in\R^{S\times{f_a}}\ \text{or}\ \text{None}}$：Answer
  ${Q_T\in\R^{S_T\times{f_q}}}$：QuestionTable
  ${H_T\in\R^{S_T\times{f_h}}}$：HintTable
  ${A_T\in\R^{S_T\times{f_a}}}$：AnswerTable
- 出力
  ${R_Q\in\R^{S\times{f_q}}}$：Deduced Question
  ${R_H\in\R^{S\times{f_h}}}$：Deduced Hint
  ${R_A\in\R^{S\times{f_a}}}$：Deduced Answer
- 計算
  入力のNoneの数に応じて処理が変化する

  1. **Initialization**: The method checks if ${ Q }$, ${ H }$, and ${ A }$ are provided or need to be generated.
  2. **First Pass Transformation**:
      If `is_first` is `True`, the tensors are passed through their respective `SingleEncoder` layers:
      ${
      Q_T' = \text{SENC}(Q_T) \in \R^{B, S, f_q}
      }$
      ${
      H_T' = \text{SENC}(H_T) \in \R^{B, S, f_h}
      }$
      ${
      A_T' = \text{SENC}(A_T) \in \R^{B, S, f_a}
      }$
  3. **Conditional Forward Pass**:
      - If ${ Q }$ is provided:
          ${
          Q_c = \text{SENC}(Q) \in \R^{B, S, f_q}
          }$
      - If ${ H }$ is provided:
          ${
          H_c = \text{SENC}(H) \in \R^{B, S, f_h}
          }$
      - If ${ A }$ is provided:
          ${
          A_c = \text{SENC}(A) \in \R^{B, S, f_a}
          }$

  4. **Single Missing Tensor Estimation**:
      - If only ${ Q }$ is provided:
          ${
          H = \text{DedS}(Q_c, Q_T', H_T') \in \R^{B, S, f_h}
          }$
          ${
          A = \text{DedS}(Q_c, Q_T', A_T') \in \R^{B, S, f_a}
          }$
          Then, recursively:
          ${
          A' = \text{self}(Q, H, Q_T', H_T', A_T') \implies A' \in \R^{B, S, f_a}
          }$
          ${
          H' = \text{self}(Q, A, Q_T', H_T', A_T') \implies H' \in \R^{B, S, f_h}
          }$
          Averaging the estimates:
          ${
          A = \frac{A + A'}{2}
          }$
          ${
          H = \frac{H + H'}{2}
          }$
      - If only ${ H }$ is provided:
          ${
          Q = \text{DedS}(H_c, H_T', Q_T') \in \R^{B, S, f_q}
          }$
          ${
          A = \text{DedS}(H_c, H_T', A_T') \in \R^{B, S, f_a}
          }$
          Then, recursively:
          ${
          A' = \text{self}(Q, H, Q_T', H_T', A_T') \implies A' \in \R^{B, S, f_a}
          }$
          ${
          Q' = \text{self}(H, A, Q_T', H_T', A_T') \implies Q' \in \R^{B, S, f_q}
          }$
          Averaging the estimates:
          ${
          A = \frac{A + A'}{2}
          }$
          ${
          Q = \frac{Q + Q'}{2}
          }$
      - If only ${ A }$ is provided:
          ${
          Q = \text{DedS}(A_c, A_T', Q_T') \in \R^{B, S, f_q}
          }$
          ${
          H = \text{DedS}(A_c, A_T', H_T') \in \R^{B, S, f_h}
          }$
          Then, recursively:
          ${
          H' = \text{self}(Q, A, Q_T', H_T', A_T') \implies H' \in \R^{B, S, f_h}
          }$
          ${
          Q' = \text{self}(H, A, Q_T', H_T', A_T') \implies Q' \in \R^{B, S, f_q}
          }$
          Averaging the estimates:
          ${
          H = \frac{H + H'}{2}
          }$
          ${
          Q = \frac{Q + Q'}{2}
          }$

  5. **Two Missing Tensors Estimation**:
      If two tensors are provided:
      - If ${ A }$ is missing:
          ${
          A = \text{DedP}(Q_c, H_c, Q_T', H_T', A_T') \in \R^{B, S, f_a}
          }$
      - If ${ H }$ is missing:
          ${
          H = \text{DedP}(Q_c, A_c, Q_T', A_T', H_T') \in \R^{B, S, f_h}
          }$
      - If ${ Q }$ is missing:
          ${
          Q = \text{DedP}(H_c, A_c, H_T', A_T', Q_T') \in \R^{B, S, f_q}
          }$

  6. **All Tensors Missing**:
      If none of the tensors are provided:
      ${
      Q = \text{random}(B, S, f_q)
      }$
      ${
      (Q', H_1', A_1') = \text{self}(Q,H_T', A_T', Q_T')
      }$
      ${
      H = \text{random}(B, S, f_h)
      }$
      ${
      (Q_1', H', A_2') = \text{self}(H,H_T', A_T', Q_T')
      }$
      ${
      A = \text{random}(B, S, f_a)
      }$
      ${
      (Q_2', H_2', A') = \text{self}(A,H_T', A_T', Q_T')
      }$
      Averaging the estimates:
      ${
      Q = \frac{Q + Q_1' + Q_2'}{3}
      }$
      ${
      H = \frac{H + H_1' + H_2'}{3}
      }$
      ${
      A = \frac{A + A_1' + A_2'}{3}
      }$

  Finally, the method returns ${ Q \in \R^{B, S, f_q} }$, ${ H \in \R^{B, S, f_h} }$, and ${ A \in \R^{B, S, f_a} }$

## DeductionNetwork (DedN)

- 内部状態
  ${Q_T\in\R^{S_T\times{f_q}}}$：QuestionTable
  ${H_T\in\R^{S_T\times{f_h}}}$：HintTable
  ${A_T\in\R^{S_T\times{f_a}}}$：AnswerTable
- 入力
  ${Q\in\R^{S\times{f_q}}\ \text{or}\ \text{None}}$：Question
  ${H\in\R^{S\times{f_h}}\ \text{or}\ \text{None}}$：Hint
  ${A\in\R^{S\times{f_a}}\ \text{or}\ \text{None}}$：Answer
- 出力
  ${R_Q\in\R^{S\times{f_q}}}$：Deduced Question
  ${R_H\in\R^{S\times{f_h}}}$：Deduced Hint
  ${R_A\in\R^{S\times{f_a}}}$：Deduced Answer
- 計算
  ${Q',H',A'=\text{DedSt}(Q,H,A,Q_T', H_T', A_T')}$
  - 以下の計算を${N}$回実行する${\{n\in{N}\}}$
  ${Q'',H'',A''=\text{DedNL}_n(Q',H',A',Q_T', H_T', A_T')}$
  ${Q',H',A'\Leftarrow (1-p_q)Q'+p_qQ'',(1-p_h)H'+p_hH'',(1-p_a)A'+p_aA''}$

  ${R_Q=Q',R_H=H',R_A=A'}$
