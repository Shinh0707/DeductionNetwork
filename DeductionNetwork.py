import matplotlib.pyplot as plt
import math
from typing import Sequence
from math import sqrt
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module
from torch import SymInt, Tensor, unsqueeze
from torch.nn import functional as F
from torch.nn import init
import torch.optim as optim
from torch.nn.parameter import Parameter


def get_nearest_divisor(n: int, start: int, prefer_low: bool = True, must_low: bool = True):
    """
    指定した数の最も近い約数を取得します。

    Args:
        n (int): 入力数。
        start (int): 探索を開始する値。
        prefer_low (bool, optional): Trueの場合、低い値を優先します。Falseの場合、高い値を優先します。デフォルトはTrue。
        must_low (bool, optional): Trueの場合、開始値以下の約数を検索します。デフォルトはTrue。

    Returns:
        int: nの最も近い約数。
    """
    if n % start == 0:
        return start
    if must_low:
        divisors = [
            i for i in range(1, int(sqrt(n)) + 1) 
            if n % i == 0 and i <= start
            ]
        divisors += [
            n // i for i in divisors 
            if n // i != i and n // i <= start
            ]
    else:
        divisors = [i for i in range(1, int(sqrt(n)) + 1) if n % i == 0]
        divisors += [n // i for i in divisors if n // i != i]
    divisors.sort(reverse=not prefer_low)
    return min(divisors, key=lambda x: abs(x - start))

class HeadMatchedMultiHeadAttension(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        """
        ヘッド数をembed_dimに合わせたMulti-head Attentionレイヤー

        Args:
            embed_dim (int): 入力の埋め込み次元数。
            num_heads (int): アテンションヘッドの数。
            dropout (float, optional): ドロップアウトの確率。デフォルトは0.0。
            bias (bool, optional): バイアス項を含むかどうか。デフォルトはTrue。
            add_bias_kv (bool, optional): bias_kvを追加するかどうか。デフォルトはFalse。
            add_zero_attn (bool, optional): zero_attnを追加するかどうか。デフォルトはFalse。
            kdim (int, optional): keyベクトルの次元数。デフォルトはNone。
            vdim (int, optional): valueベクトルの次元数。デフォルトはNone。
            batch_first (bool, optional): 入力の形式がバッチサイズを先頭に持つかどうか。デフォルトはFalse。
            device (torch.device, optional): デバイスオプション。デフォルトはNone。
            dtype (torch.dtype, optional): データタイプオプション。デフォルトはNone。

        Forward:
            入力Q、K、Vに対してMulti-head Attentionを適用し、出力を返します。
        """
        super().__init__()
        num_heads = get_nearest_divisor(embed_dim,num_heads)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype
        )

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        return self.attn.forward(
            Q, K, V,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal
        )


class KeyPair(Module):
    def __init__(self, kdim: int, vdim: int, embed_dim: int = None, bias: bool = True):
        """
        KeyとValueを結合するモジュール。

        Args:
            kdim (int): keyベクトルの次元数。
            vdim (int): valueベクトルの次元数。
            embed_dim (int, optional): 結合後の埋め込み次元数。デフォルトはNone。
            bias (bool, optional): バイアス項を含むかどうか。デフォルトはTrue。

        Forward:
            入力のKeyとValueを結合し、線形層と活性化関数を適用して出力します。
        """
        super().__init__()
        if embed_dim == None:
            embed_dim = kdim+vdim
        self.layer = nn.Sequential(
            nn.Linear(kdim+vdim, embed_dim, bias=bias),
            nn.RReLU()
        )
        

    def forward(self, K: Tensor, V: Tensor):
        return self.layer.forward(torch.cat((K, V), dim=-1))


class FeedForwardNetwork(Module):
    def __init__(self, in_dim: int, hid_dim: int | None = None, out_dim: int | None = None, dropout: float = .0):
        """
        Feed Forward Networkモジュール。

        Args:
            in_dim (int): 入力の次元数。
            hid_dim (int, optional): 隠れ層の次元数。デフォルトはNone。
            out_dim (int, optional): 出力の次元数。デフォルトはNone。
            dropout (float, optional): ドロップアウトの確率。デフォルトは0.0。

        Forward:
            入力に対して線形層、活性化関数、ドロップアウトを適用し、出力します。
        """
        super().__init__()
        if hid_dim == None:
            hid_dim = in_dim * 2
        if out_dim == None:
            out_dim = in_dim
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim)
        )
        torch.nn.init.kaiming_uniform_(
            self.layer[0].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, X: Tensor):
        return self.layer.forward(X)


class DeductionNetworkPartialLayer(Module):
    def __init__(self, embeded_dim: int, vdim: int, num_heads: int, dropout: float = .0):
        """
        部分的な演驛ネットワークレイヤー。

        Args:
            embeded_dim (int): 埋め込み次元数。
            vdim (int): valueベクトルの次元数。
            num_heads (int): アテンションヘッドの数。
            dropout (float, optional): ドロップアウトの確率。デフォルトは0.0。

        Forward:
            Multi-head AttentionとFeed Forward Networkを適用し、出力を返します。
        """
        super().__init__()
        self.attn = HeadMatchedMultiHeadAttension(
            embeded_dim, num_heads, kdim=embeded_dim, vdim=vdim, dropout=dropout
        )
        self.ln1 = nn.LayerNorm(vdim)
        self.ffn = FeedForwardNetwork(vdim, dropout=dropout)
        self.ln2 = nn.LayerNorm(vdim)

    def forward(self, Q_embeded: Tensor, K_embeded: Tensor, V: Tensor):
        X, _ = self.attn.forward(Q_embeded, K_embeded, V)
        D = self.ln1.forward(Q_embeded+X)
        FF = self.ffn.forward(D)
        return self.ln2.forward(D+FF)


class CDeductionNetworkPartialLayer(Module):
    def __init__(self, qdim: int, kdim: int, vdim: int, num_heads: int, bias: bool = True, dropout: float = .0):
        """
        部分的な演驛ネットワークレイヤー

        Args:
            qdim (int): queryベクトルの次元数。
            kdim (int): keyベクトルの次元数。
            vdim (int): valueベクトルの次元数。
            num_heads (int): アテンションヘッドの数。
            bias (bool, optional): バイアス項を含むかどうか。デフォルトはTrue。
            dropout (float, optional): ドロップアウトの確率。デフォルトは0.0。
        
        Forward:
            入力のQueryとKeyを結合し、部分的な演驛ネットワークを適用します。
        """
        super().__init__()
        self.Embedding = KeyPair(qdim,kdim, vdim, bias=bias)
        self.DedQK = DeductionNetworkPartialLayer(
            vdim, vdim, num_heads, dropout)

    def forward(self, Q: Tensor, K: Tensor, targetQ: Tensor, targetK: Tensor, V: Tensor):
        EmbIn = self.Embedding.forward(Q,K)
        EmbTarget = self.Embedding.forward(targetQ, targetK)
        return self.DedQK.forward(EmbIn, EmbTarget, V)


class DeductionNetworkLayer(Module):
    def __init__(self, qdim: int, kdim: int, vdim: int, num_heads: int, bias: bool = True, dropout: float = .0):
        """
        演驛ネットワークレイヤー

        Args:
            qdim (int): queryベクトルの次元数。
            kdim (int): keyベクトルの次元数。
            vdim (int): valueベクトルの次元数。
            num_heads (int): アテンションヘッドの数。
            bias (bool, optional): バイアス項を含むかどうか。デフォルトはTrue。
            dropout (float, optional): ドロップアウトの確率。デフォルトは0.0。
        
        Forward:
            入力のQuery、Key、Valueに演驛ネットワークを適用し、出力を返します。
        """
        super().__init__()
        factory_kwargs = {
            'num_heads': num_heads,
            'bias': bias,
            'dropout': dropout
        }
        self.CDedQK = CDeductionNetworkPartialLayer(
            qdim, kdim, vdim, **factory_kwargs)
        self.CDedQV = CDeductionNetworkPartialLayer(
            qdim, vdim, kdim, **factory_kwargs)
        self.CDedKV = CDeductionNetworkPartialLayer(
            kdim, vdim, qdim, **factory_kwargs)

    def forward(self, Q: Tensor, K: Tensor, targetQ: Tensor, targetK: Tensor, V: Tensor):
        VDed = self.CDedQK.forward(Q, K, targetQ, targetK, V)
        KDed = self.CDedQV.forward(Q, VDed, targetQ, V, targetK)
        QDed = self.CDedKV.forward(K, VDed, targetK, V, targetQ)
        return QDed, KDed, VDed


class DeductionNetwork(Module):
    def __init__(self, qdim: int, kdim: int, vdim: int, table_size: int, num_layers: int, num_heads: int,bias:bool=True,dropout: float=.0,device=None, dtype=None):
        """
        演驛ネットワークモジュール。

        Args:
            qdim (int): queryベクトルの次元数。
            kdim (int): keyベクトルの次元数。
            vdim (int): valueベクトルの次元数。
            table_size (int): 内部パラメータのテーブルサイズ。
            num_layers (int): レイヤーの数。
            num_heads (int): アテンションヘッドの数。
            bias (bool, optional): バイアス項を含むかどうか。デフォルトはTrue。
            dropout (float, optional): ドロップアウトの確率。デフォルトは0.0。
            device (torch.device, optional): デバイスオプション。デフォルトはNone。
            dtype (torch.dtype, optional): データタイプオプション。デフォルトはNone。
        
        Forward:
            入力のQuery、Key、Valueに演驛ネットワークを適用し、出力を返します。
        """
        super().__init__()
        factory_kwargs = {'requires_grad': True,
                          'device': device, 'dtype': dtype}
        self.InternalParameters = nn.ParameterList(
            [
                Parameter(torch.empty(table_size, qdim,
                          **factory_kwargs), True),
                Parameter(torch.empty(table_size, kdim,
                          **factory_kwargs), True),
                Parameter(torch.empty(table_size, vdim, 
                          **factory_kwargs), True)
            ]
        )
        layer_kwargs = {
            'num_heads': num_heads,
            'bias': bias,
            'dropout': dropout
        }
        self.QKlayers = nn.ModuleList(
            [
                DeductionNetworkLayer(qdim, kdim, vdim, **layer_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.QVlayers = nn.ModuleList(
            [
                DeductionNetworkLayer(qdim, vdim, kdim, **layer_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.KVlayers = nn.ModuleList(
            [
                DeductionNetworkLayer(kdim, vdim, qdim, **layer_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for i in range(len(self.InternalParameters)):
            init.xavier_uniform_(self.InternalParameters[i],
                                 gain=nn.init.calculate_gain('tanh'))

    def DedLayerForward(self, Q: Tensor, K: Tensor, layers: list[DeductionNetworkLayer], Ql: Tensor, Kl: Tensor, Vl: Tensor):
        for l in layers:
            Ql, Kl, Vl = l.forward(Q, K, Ql, Kl, Vl)
        return Vl

    def forward(self, Q, K, V):
        if Q == None:
            assert K != None and V != None, "K and V must not be None"
            Q = self.DedLayerForward(
                K, V, self.KVlayers,
                Ql=self.InternalParameters[1],
                Kl=self.InternalParameters[2],
                Vl=self.InternalParameters[0]
            )
        if K == None:
            assert Q != None and V != None, "Q and V must not be None"
            K = self.DedLayerForward(
                Q, V, self.QVlayers,
                Ql=self.InternalParameters[0],
                Kl=self.InternalParameters[2],
                Vl=self.InternalParameters[1]
            )
        if V == None:
            assert Q != None and K != None, "Q and K must not be None"
            V = self.DedLayerForward(
                Q, K, self.QKlayers,
                Ql=self.InternalParameters[0],
                Kl=self.InternalParameters[1],
                Vl=self.InternalParameters[2]
            )
        return Q, K, V


# 足し算を検証してみる
def createAddingData(dataset: int = 100,max_val: float=100):
    # A + B = C ができるかどうか
    # Q : A, K : B, V : C
    C = np.random.randint(1, int(max_val), dataset)
    A = np.asarray([np.random.randint(0, c) for c in C])
    B = C-A
    return torch.tensor(A/max_val).unsqueeze(-1).float(), torch.tensor(B/max_val).unsqueeze(-1).float(), torch.tensor(C/max_val).unsqueeze(-1).float()


def train(epochs: int = 100):
    data_sets = 100
    #Q, K, V = createAddingData(data_sets)
    #print(Q, K, V)
    DedN = DeductionNetwork(1, 1, 1, 2000, 8, 8, dropout=.25)
    optimizer = optim.AdamW(DedN.parameters(), lr=.01)
    mean_errs = []  # mean_errを保存するリストを作成
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        partial_mean_err = 0
        for i in range(data_sets):
            Q,K,V = createAddingData(1,data_sets*100)
            optimizer.zero_grad()
            _, _, lV = DedN.forward(
                Q, K, None
            )
            _, lK, _ = DedN.forward(
                Q, None, V
            )
            lQ, _, _ = DedN.forward(
                None, K, V
            )
            err_fn = nn.L1Loss()
            err = err_fn.forward(
                lV, V
            )+err_fn.forward(
                lK, K
            )+err_fn.forward(
                lQ, Q
            )
            mean_err = err * data_sets
            mean_err.backward()
            torch.nn.utils.clip_grad_norm_(DedN.parameters(), 4.0)
            optimizer.step()
            if i % 10 == 0:
                mean_errs.append(mean_err.item())  # mean_errの値をリストに追加
            partial_mean_err += mean_err.item()
            #print(f"mean err = {mean_err},\n {
            #      lQ.item()}+{lK.item()} = {lV.item()}?")
        print(f"mean err = {partial_mean_err/data_sets}")
    # エポックごとのmean_errをプロット
    plt.plot(mean_errs)
    # plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Error')
    plt.show()

train(100)
