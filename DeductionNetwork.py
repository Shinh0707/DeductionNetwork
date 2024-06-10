import matplotlib.pyplot as plt
from math import sqrt
import torch
import torch.nn as nn
import numpy as np
from numpy.random import randint
from torch.nn import Module
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.onnx

# Function to Convert to ONNX


def Convert_ONNX(model: Module, dummy_input: Tensor, input_names: list[str], output_names: list[str], model_name: str):

    # set the model to inference mode
    model.eval()

    # Export the model
    torch.onnx.export(model,         # model being run
                      # model input (or a tuple for multiple inputs)
                      dummy_input,
                      f"{model_name}.onnx",       # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=input_names,   # the model's input names
                      output_names=output_names,  # the model's output names
                      )
    print(" ")
    print('Model has been converted to ONNX')

"""
モデルに必要なものたち↓
"""

# 必須ではない（ヘッド数自動で計算してるだけ）
def get_nearest_divisor(n: int, start: int, prefer_low: bool = True, must_low: bool = True):
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

# 必須ではない（ヘッド数自動で計算してるだけ）
class HeadMatchedMultiHeadAttension(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
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

"""
以下は必須
"""

class KeyPair(Module):
    def __init__(self, kdim: int, vdim: int, embed_dim: int = None, bias: bool = True ,dropout: float = .0):
        super().__init__()
        if embed_dim == None:
            embed_dim = kdim+vdim
        self.layer = nn.Sequential(
            nn.Linear(kdim+vdim, embed_dim, bias=bias),
            nn.Dropout(dropout),
            nn.RReLU()  # 絶対に負の値が必要
        )
        init.kaiming_uniform_(
            self.layer[0].weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, K: Tensor, V: Tensor):
        return self.layer.forward(torch.cat((K, V), dim=-1))


class FeedForwardNetwork(Module):
    def __init__(self, in_dim: int, hid_dim: int | None = None, out_dim: int | None = None, dropout: float = .0):
        super().__init__()
        if hid_dim == None:
            hid_dim = in_dim * 2
        if out_dim == None:
            out_dim = in_dim
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),  # nn.ReLU() # Attension Is All you needのFeedForward準拠ならReLUにすべき
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim)
        )
        init.kaiming_uniform_(
            self.layer[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_uniform_(
            self.layer[-1].weight, mode='fan_in')

    def forward(self, X: Tensor):
        return self.layer.forward(X)


class DeductionNetworkPartialLayer(Module):
    def __init__(self, embeded_dim: int, vdim: int, num_heads: int, dropout: float = .0):
        super().__init__()
        self.attn = HeadMatchedMultiHeadAttension(
            embeded_dim, num_heads, kdim=embeded_dim, vdim=vdim, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embeded_dim)
        self.ffn = FeedForwardNetwork(embeded_dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(embeded_dim)

    def forward(self, Q_embeded: Tensor, K_embeded: Tensor, V: Tensor):
        X, _ = self.attn.forward(Q_embeded, K_embeded, V)
        D = self.ln1.forward(Q_embeded+X)
        FF = self.ffn.forward(D)
        return self.ln2.forward(D+FF)


class CDeductionNetworkPartialLayer(Module):
    def __init__(self, qdim: int, kdim: int, vdim: int,embed_dim:int, num_heads: int, bias: bool = True, dropout: float = .0):
        super().__init__()
        self.Embedding = KeyPair(qdim, kdim, embed_dim, bias=bias, dropout=dropout)
        self.VEmbedding = FeedForwardNetwork(vdim,out_dim=embed_dim)
        self.lastLayer = FeedForwardNetwork(embed_dim, out_dim=vdim)
        self.DedQK = DeductionNetworkPartialLayer(
            embed_dim, embed_dim, num_heads, dropout
        )

    def forward(self, Q: Tensor, K: Tensor, targetQ: Tensor, targetK: Tensor, V: Tensor):
        EmbIn = self.Embedding.forward(Q,K)
        EmbTarget = self.Embedding.forward(targetQ, targetK)
        DedQK = self.DedQK.forward(EmbIn, EmbTarget, self.VEmbedding.forward(V))
        return self.lastLayer.forward(EmbIn+DedQK)


class DeductionNetworkLayer(Module):
    def __init__(self, qdim: int, kdim: int, vdim: int, embed_dim: int, num_heads: int, bias: bool = True, dropout: float = .0):
        super().__init__()
        factory_kwargs = {
            'embed_dim': embed_dim,
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
    def __init__(self, qdim: int, kdim: int, vdim: int,embed_dim: int, table_size: int, num_layers: int, num_heads: int,bias:bool=True,dropout: float=.0,device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'requires_grad': True,
                          'device': device, 'dtype': dtype}
        self.InternalParameters = nn.ParameterList(
            [
                Parameter(torch.empty(1,table_size, qdim,
                          **factory_kwargs), True),
                Parameter(torch.empty(1,table_size, kdim,
                          **factory_kwargs), True),
                Parameter(torch.empty(1,table_size, vdim, 
                          **factory_kwargs), True)
            ]
        )
        layer_kwargs = {
            'embed_dim': embed_dim,
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
        for param in self.InternalParameters:
            init.kaiming_uniform_(param, mode='fan_in')

    def DedLayerForward(self, Q: Tensor, K: Tensor, layers: list[DeductionNetworkLayer], Ql: Tensor, Kl: Tensor, Vl: Tensor):
        for l in layers:
            Ql, Kl, Vl = l.forward(Q, K, Ql, Kl, Vl)
        return Vl

    def forward(self, Q: Tensor | None, K: Tensor | None, V: Tensor | None, batch_size: int = None):
        """
        Noneになっているテンソルの入力を埋めて返す。Noneではない入力はそのまま返ってくる。
        """
        assert not (Q == None and K == None and V == None and batch_size == None) ,"If you want to all parameters are None, please at least set any number to batch_size (ex. 1)"
        batch_size, _, _ = Q.shape if Q is not None else K.shape if K is not None else V.shape if V is not None else (batch_size, 0, 0)
        if Q == None:
            # assert K != None and V != None, "K and V must not be None"
            if K == None:
                lK = self.InternalParameters[1][:,
                                                randint(0, self.InternalParameters[1].shape[1]), :].expand(batch_size, -1, -1)
            else:
                lK = K
            if V == None:
                lV = self.InternalParameters[2][:,
                                                randint(0, self.InternalParameters[2].shape[1]), :].expand(batch_size, -1, -1)
            else:
                lV = V
            Q = self.DedLayerForward(
                lK, lV, self.KVlayers,
                Ql=self.InternalParameters[1].expand(batch_size, -1, -1),
                Kl=self.InternalParameters[2].expand(batch_size, -1, -1),
                Vl=self.InternalParameters[0].expand(batch_size, -1, -1)
            )
        if K == None:
            # assert Q != None and V != None, "Q and V must not be None"
            if Q == None:
                lQ = self.InternalParameters[0][:,
                                                randint(0, self.InternalParameters[0].shape[1]), :].expand(batch_size, -1, -1)
            else:
                lQ = Q
            if V == None:
                lV = self.InternalParameters[2][:,
                                                randint(0, self.InternalParameters[2].shape[1]), :].expand(batch_size, -1, -1)
            else:
                lV = V
            K = self.DedLayerForward(
                lQ, lV, self.QVlayers,
                Ql=self.InternalParameters[0].expand(batch_size, -1, -1),
                Kl=self.InternalParameters[2].expand(batch_size, -1, -1),
                Vl=self.InternalParameters[1].expand(batch_size, -1, -1)
            )
        if V == None:
            # assert Q != None and K != None, "Q and K must not be None"
            if Q == None:
                lQ = self.InternalParameters[0][:,
                                                randint(0, self.InternalParameters[0].shape[1]), :].expand(batch_size, -1, -1)
            else:
                lQ = Q
            if K == None:
                lK = self.InternalParameters[1][:,
                                                randint(0, self.InternalParameters[1].shape[1]), :].expand(batch_size, -1, -1)
            else:
                lK = K
            V = self.DedLayerForward(
                lQ, lK, self.QKlayers,
                Ql=self.InternalParameters[0].expand(batch_size, -1, -1),
                Kl=self.InternalParameters[1].expand(batch_size, -1, -1),
                Vl=self.InternalParameters[2].expand(batch_size, -1, -1)
            )
        return Q, K, V

"""
モデルに必要なものたちここまで

以下検証用
"""

# 足し算を検証してみる
def createAddingData(dataset: int = 100, max_val: float = 100, normalize_base: float = 100):
    # A + B = C ができるかどうか
    # Q : A, K : B, V : C
    C = np.random.randint(1, int(max_val), dataset)
    A = np.asarray([np.random.randint(0, c) for c in C])
    B = C-A
    return torch.tensor(A/normalize_base).unsqueeze(-1).unsqueeze(1).float(), torch.tensor(B/normalize_base).unsqueeze(-1).unsqueeze(1).float(), torch.tensor(C/normalize_base).unsqueeze(-1).unsqueeze(1).float()


# 剰余を検証してみる
def createModData(dataset: int = 100, max_val: float = 100, normalize_base: float= 100):
    # A % B = C ができるかどうか
    # Q : A, K : B, V : C
    A = np.random.randint(1, int(max_val), dataset)
    B = np.asarray([np.random.randint(1, a+1) for a in A])
    C = A%B
    return torch.tensor(A/normalize_base).unsqueeze(-1).unsqueeze(1).float(), torch.tensor(B/normalize_base).unsqueeze(-1).unsqueeze(1).float(), torch.tensor(C/normalize_base).unsqueeze(-1).unsqueeze(1).float()

# ベクトルの差を検証してみる
def createSubVectorData(dataset: int = 100):
    # A - B = C ができるかどうか
    # Q : A, K : B, V : C
    A = np.random.rand(dataset, 1, 2)
    B = np.random.rand(dataset, 1, 2)
    C = A-B
    return torch.tensor(A).float(), torch.tensor(B).float(), torch.tensor(C).float()

# 形状が異なっててもいけるかどうか
def createUnevenVectorData(dataset,dims:tuple[int,int,int],even_size:int,uneven_size:int,evenPair:tuple[int,int]=(0,1)):
    # A,B,Cの対応するもの以外の形状が異なっても学習可能かどうか
    # Q : A, K : B, V : C
    A = np.random.rand(
        dataset, even_size if 0 in evenPair else uneven_size, dims[0])
    B = np.random.rand(
        dataset, even_size if 1 in evenPair else uneven_size, dims[1])
    C = np.random.rand(
        dataset, even_size if 2 in evenPair else uneven_size, dims[2])
    return torch.tensor(A).float(), torch.tensor(B).float(), torch.tensor(C).float()

def train(epochs: int = 100):
    data_sets = 100
    #Q, K, V = createAddingData(data_sets)
    #print(Q, K, V)
    DedN = DeductionNetwork(5, 3, 2, 8, 5, 4, 8, dropout=.25) 
    # 特徴次元の４倍くらいの大きさの埋め込み次元があれば良さそう（ヒューリスティック）
    # テーブルサイズを変えてもあまり結果は変わらない。計算時間は微妙に伸びるし収束が遅くなる。変化量が小さくなる。必要最低限でいい。
    # テーブルサイズの大きさに比例して実行時間が伸びる、テーブルサイズが小さいと精度が下がるかと思われたがそうでもない
    # テーブルサイズが小さいほど精度が上がる！？？？→上がらない・限界をすぐ迎える
    # テーブルサイズ0でも動く→必要なものさえ与えられていれば
    # 結論：特徴次元が大事
    # テーブルサイズを上げると時間が経つにつれて大きく誤差が減るようになって精度がいい...というわけでもなさそう
    optimizer = optim.AdamW(DedN.parameters(), lr=.01)
    total_errs = []  # mean_errを保存するリストを作成
    v_errs = []
    k_errs = []
    q_errs = []
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        # ちゃんと処理できるなら動的なサイズでも学習できる
        spl = randint(2,4) # 動的なサイズ
        print(f"shape {spl}") # 動的なサイズ
        Q, K, V = createUnevenVectorData(data_sets, (5,3,2), spl*2, 2, (0, 2))
        #createModData(
        #    data_sets, (data_sets+epoch)*100, (data_sets+epochs)*100)  # 過適合対策
        optimizer.zero_grad()
        # 無から生み出すやつ, 適当に数字あげて違うって言われてるみたいなもの（精度保証不可）
        #lQ, lK, lV = DedN.forward(
        #    None, None, None, Q.shape[0] # 全部Noneのときはバッチサイズだけ入れとく
        #)
        # １つの情報から生み出すやつ, 不十分な質問されて質問を補完させて解答を出すみたいなもの
        #　Q → K生成 →　Qと生成したKでV生成 → 生成したK,VからQ生成
        #_, lK, lV = DedN.forward(
        #    Q, None, None
        #)
        #lQ, _, _ = DedN.forward(
        #    None, lK, lV
        #)
        # 演驛するやつ, 不十分な質問と解答見せて、質問を補完させるみたいなもの
        _, lK, _ = DedN.forward(
            Q, None, V
        )
        _, _, lV = DedN.forward(
            Q, lK, None
        )
        lQ, _, _ = DedN.forward(
            None, lK, V
        )
        # lK の形状だけ違うので、いい感じに使って関連性持たせてあげる（Kだけ学習しないというのも手）
        lK = torch.cat((torch.mean(lK[:, spl:, :], dim=1, keepdim=True), torch.mean(lK[:, :spl, :], dim=1, keepdim=True)),dim=1)
        # 通常パターン, 穴埋め解かせるだけ
        #_, lK, _ = DedN.forward(
        #    Q, None, V
        #)
        #_, _, lV = DedN.forward(
        #    Q, K, None
        #)
        #lQ, _, _ = DedN.forward(
        #    None, K, V
        #)
        err_fn = nn.L1Loss()
        v_err = err_fn.forward(
            lV, V
        ) * data_sets
        k_err = err_fn.forward(
            lK, K
        ) * data_sets
        q_err = err_fn.forward(
            lQ, Q
        ) * data_sets
        total_err = v_err+k_err+q_err 
        total_err.backward()
        torch.nn.utils.clip_grad_norm_(DedN.parameters(), 4.0)
        optimizer.step()
        total_errs.append(total_err.item()/data_sets)
        v_errs.append(v_err.item()/data_sets)
        k_errs.append(k_err.item()/data_sets)
        q_errs.append(q_err.item()/data_sets)
        print(
            f"Total Error = {total_errs[-1]}\nQ:{q_errs[-1]}\nK:{k_errs[-1]}\nV:{v_errs[-1]}"
            )
    # エポックごとの誤差をプロット
    plt.plot(total_errs, label='Total Error',color='m')
    plt.plot(q_errs, label='Q Error', color='r')
    plt.plot(k_errs, label='K Error', color='g')
    plt.plot(v_errs, label='V Error', color='b')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    return DedN

model = train(300)
# testQ, testK, testV = createModData(1, 10000, 10000)
# Convert_ONNX(model,(testQ,None,testV),['QIn','KIn','VIn'],['QOut','KOut','Vout'],'DeductionNetwork')
