from typing import Literal
import warnings
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

"""
以下は必須
"""


class Attension(Module):
    def __init__(self, use_scale: bool = True, score_mode: Literal['dot', 'cat'] = 'dot', dropout: float = .1):
        super().__init__()
        self.use_scale = use_scale
        self.score_mode = score_mode
        self.dropout = nn.Dropout(dropout)
        if score_mode not in ['dot', 'cat']:
            raise ValueError(
                "Invalid value for argment score_mode"
                "Expected one of {'dot', 'cat'}"
                f"Recived: score_mode={score_mode}"
            )
        if self.use_scale:
            self.scale = Parameter(torch.ones(1))
        else:
            self.scale = None
        if self.score_mode == "cat":
            self.concat_score_weight = Parameter(torch.ones(1))
        else:
            self.concat_score_weight = None
        self.softmax = nn.Softmax(dim=-1)

    def _calculate_scores(self, query: Tensor, key: Tensor):
        if self.score_mode == "dot":
            scores = torch.matmul(query, key.transpose(-2, -1))
            if self.scale is not None:
                scores = scores * self.scale
        elif self.score_mode == "cat":
            q_reshaped = query.unsqueeze(-2)
            k_reshaped = key.unsqueeze(-3)
            if self.scale is not None:
                scores = self.concat_score_weight * torch.sum(
                    torch.tanh(self.scale * (q_reshaped+k_reshaped)), dim=-1
                )
            else:
                scores = self.concat_score_weight * torch.sum(
                    torch.tanh(q_reshaped+k_reshaped), dim=-1
                )
        return scores

    def _apply_scores(self, scores: Tensor, value: Tensor, score_mask: Tensor | None = None):
        if score_mask is not None:
            padding_mask = -score_mask
            max_value = 1.0e9 if scores.dtype == torch.float32 else 65504.0
            scores = scores.masked_fill((padding_mask, -max_value))

        weights = self.softmax.forward(scores)
        if self.training:
            weights = self.dropout.forward(weights)

        return torch.matmul(weights, value), weights, scores

    def _calculate_score_mask(self, scores: Tensor, v_mask: Tensor|None, is_causal: bool):
        if is_causal:
            score_shape = scores.size()
            mask_shape = (1, score_shape[-2], score_shape[-1])
            causal_mask = torch.tril(torch.ones(
                mask_shape, dtype=bool, device=scores.device))

            if v_mask is not None:
                v_mask = v_mask.unsqueeze(1)
                return v_mask & causal_mask
            return causal_mask
        else:
            return v_mask

    def forward(self, query: Tensor, key: Tensor | None = None, value: Tensor | None = None, q_mask: Tensor | None = None, v_mask: Tensor | None = None, return_attn_scores: bool = False, return_attn_weights: bool = False, use_causal_mask: bool = False):
        scores = self._calculate_scores(query, key)
        score_mask = self._calculate_score_mask(
            scores, v_mask, use_causal_mask)

        result, attn_weights, attn_scores = self._apply_scores(scores, value, score_mask)

        if q_mask is not None:
            q_mask = q_mask.unsqueeze(-1)
            result = result * q_mask.float()

        if return_attn_scores:
            if return_attn_weights:
                return result, attn_scores, attn_weights
            return result, attn_scores
        if return_attn_weights:
            return result, attn_weights
        return result


class MHA(nn.Module):
    def __init__(self, num_heads, key_dim, value_dim=None, dropout=0.0, bias=True, out_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.dropout = dropout
        self.use_bias = bias
        self.output_shape = out_dim
        self.scale = 1.0 / sqrt(self.key_dim)

        self.query_linear = nn.Linear(
            key_dim, num_heads * key_dim, bias=bias)
        self.key_linear = nn.Linear(
            key_dim, num_heads * key_dim, bias=bias)
        self.value_linear = nn.Linear(
            value_dim, num_heads * self.value_dim, bias=bias)
        self.out_linear = nn.Linear(
            num_heads * self.value_dim, key_dim if out_dim is None else out_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor = None, value: Tensor = None, attention_mask=None, return_attention_scores=False, use_causal_mask=False):
        if value is None:
            value = query
        if key is None:
            key = value

        batch_size = query.size(0)
        seq_length_q = query.size(1)
        seq_length_k = key.size(1)

        query = self.query_linear(query).view(
            batch_size, seq_length_q, self.num_heads, self.key_dim)
        key = self.key_linear(key).view(
            batch_size, seq_length_k, self.num_heads, self.key_dim)
        value = self.value_linear(value).view(
            batch_size, seq_length_k, self.num_heads, self.value_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention_scores = torch.matmul(
            query, key.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, float('-inf'))

        if use_causal_mask:
            causal_mask = torch.tril(torch.ones(seq_length_q, seq_length_k)).unsqueeze(
                0).unsqueeze(0).to(attention_scores.device)
            attention_scores = attention_scores.masked_fill(
                causal_mask == 0, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1)

        if self.dropout > 0:
            attention_probs = self.dropout_layer(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_length_q, -1)

        output = self.out_linear(context)

        if return_attention_scores:
            return output, attention_probs
        return output

class FeedForwardNetwork(Module):
    def __init__(self, in_dim: int, hid_dim: int | None = None, out_dim: int | None = None, dropout: float = .0, activation_func: Literal['relu', 'leaky_relu'] = 'relu'):

        super().__init__()
        if hid_dim == None:
            hid_dim = in_dim * 2
        if out_dim == None:
            out_dim = in_dim
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU() if activation_func == 'relu' else nn.LeakyReLU(),  # nn.ReLU() # Attension Is All you needのFeedForward準拠ならReLUにすべき
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim)
        )
        init.kaiming_uniform_(
            self.layer[0].weight, mode='fan_in', nonlinearity=activation_func)
        init.kaiming_uniform_(
            self.layer[-1].weight, mode='fan_in')

    def forward(self, X: Tensor) -> Tensor:
        return self.layer.forward(X)


class SingleEncoder(Module):
    def __init__(self, in_dim: int, hid_dim: int | None = None, out_dim: int | None = None, dropout: float = .0):
        super().__init__()
        # FeedForwardNetwork(kdim+vdim, hid_dim,out_dim),
        if hid_dim is None:
            hid_dim = 2*in_dim
        if out_dim is None:
            out_dim = in_dim
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
            nn.Tanh()
        )
        init.kaiming_normal_(self.layer[0].weight)
        init.xavier_uniform_(self.layer[-2].weight)

    def forward(self, X: Tensor) -> Tensor:
        return self.layer.forward(X)


class PairEncoder(Module):
    def __init__(self, kdim: int, vdim: int, hid_dim: int | None = None, out_dim: int | None = None, dropout: float = .0):
        super().__init__()
        # FeedForwardNetwork(kdim+vdim, hid_dim,out_dim),
        self.enc = SingleEncoder(kdim+vdim,hid_dim,out_dim,dropout)

    def forward(self, K: Tensor, V: Tensor) -> Tensor:
        return self.enc.forward(torch.cat((K, V), dim=-1))


class DeductionNetworkSingleLayer(Module):
    def __init__(self, embed_dim: int, srcdim: int, num_heads: int, dropout: float = .0, bias: bool = True):
        super().__init__()
        MHA_kwargs = {
            'num_heads': num_heads,
            'dropout': dropout,
            'bias': bias
        }
        self.AMHA = MHA(key_dim=embed_dim, value_dim=srcdim,
                        out_dim=srcdim, **MHA_kwargs)
        self.LayerNormA = nn.LayerNorm(srcdim)
        self.AFF = FeedForwardNetwork(srcdim)
        self.attn = Attension(dropout=dropout)

    def forward(self, Q: Tensor, H: Tensor, A: Tensor):
        """
        print("at DedS")
        print(f"Q = {Q.shape}")
        print(f"H = {H.shape}")
        print(f"A = {A.shape}")
        """
        A_m = self.AMHA.forward(Q, H, A)
        # print(f"MHA = {Q.shape},{H.shape},{A.shape} => {A_m.shape}")
        Ad = self.LayerNormA.forward(
            A_m + self.attn.forward(Q,H,A)
        )
        A_FF = self.AFF.forward(Ad)
        # print(f"FF(A) => {A_FF.shape}")
        return self.LayerNormA.forward(A_FF+Ad)


class DeductionNetworkPairLayer(Module):
    def __init__(self, qdim: int, hdim: int, adim: int, embed_dim: int, num_heads: int, dropout: float = .0, bias: bool = True):
        super().__init__()
        MHA_kwargs = {
            'num_heads': num_heads,
            'dropout': dropout,
            'bias': bias
        }
        self.PE_QH = PairEncoder(
            qdim, hdim, out_dim=embed_dim, dropout=dropout)
        self.DedSN2 = DeductionNetworkSingleLayer(qdim, adim, **MHA_kwargs)
        self.DedSN1 = DeductionNetworkSingleLayer(embed_dim, adim,**MHA_kwargs)

    def forward(self, Q: Tensor, H: Tensor, targetQ: Tensor, targetH: Tensor, A: Tensor):
        """
        print(f"Q = {Q.shape}")
        print(f"H = {H.shape}")
        """
        E_QH = self.PE_QH.forward(Q, H)
        E_QHT = self.PE_QH.forward(targetQ, targetH)
        return self.DedSN2.forward(Q, H, self.DedSN1.forward(E_QH, E_QHT, A))


class DeductionNetworkLayer(Module):
    def __init__(self, qdim: int, hdim: int, adim: int, embed_dim: int, num_heads: int, dropout: float = .0, bias: bool = True):
        super().__init__()
        MHA_kwargs = {
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'dropout': dropout,
            'bias': bias
        }
        self.DedQH = DeductionNetworkPairLayer(
            qdim, hdim, adim, **MHA_kwargs)
        self.DedQA = DeductionNetworkPairLayer(
            qdim, adim, hdim, **MHA_kwargs)
        self.DedHA = DeductionNetworkPairLayer(
            hdim, adim, qdim, **MHA_kwargs)

    def forward(self, Qd: Tensor, Hd: Tensor, Ad: Tensor, targetQ: Tensor, targetH: Tensor, targetA: Tensor):
        A = self.DedQH.forward(Qd, Hd, targetQ, targetH, targetA)
        H = self.DedQA.forward(Qd, Ad, targetQ, targetA, targetH)
        Q = self.DedHA.forward(Hd, Ad, targetH, targetA, targetQ)
        return Q, H, A


class DeductionNetworkStartLayer(Module):
    def __init__(self, qdim: int, hdim: int, adim: int, embed_dim: int, num_heads: int, bias: bool = True, dropout: float = .0):
        super().__init__()
        self.qdim = qdim
        self.hdim = hdim
        self.adim = adim
        dlayer_kwargs = {
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'dropout': dropout,
            'bias':bias
        }
        dslayer_kwargs = {
            'num_heads': num_heads,
            'dropout': dropout,
            'bias': bias
        }
        self.q_layers: tuple[SingleEncoder, DeductionNetworkPairLayer, DeductionNetworkSingleLayer, DeductionNetworkSingleLayer] = DeductionNetworkStartLayer._create_module_list(
            qdim, hdim, adim, dlayer_kwargs, dslayer_kwargs)
        self.h_layers: tuple[SingleEncoder, DeductionNetworkPairLayer, DeductionNetworkSingleLayer, DeductionNetworkSingleLayer] = DeductionNetworkStartLayer._create_module_list(
            hdim, qdim, adim, dlayer_kwargs, dslayer_kwargs)
        self.a_layers: tuple[SingleEncoder, DeductionNetworkPairLayer, DeductionNetworkSingleLayer, DeductionNetworkSingleLayer] = DeductionNetworkStartLayer._create_module_list(
            adim, qdim, hdim, dlayer_kwargs, dslayer_kwargs)
    def single_encoders(self):
        return [self.q_layers[0], self.h_layers[0], self.a_layers[0]]
    @staticmethod
    def _create_module_list(qdim: int, hdim: int, adim: int, dlayer_kwargs, dslayer_kwargs):
        return nn.ModuleList(
            [
                SingleEncoder(qdim),
                DeductionNetworkPairLayer(
                    hdim, adim, qdim, **dlayer_kwargs),
                DeductionNetworkSingleLayer(
                    qdim, hdim, **dslayer_kwargs),
                DeductionNetworkSingleLayer(
                    qdim, adim, **dslayer_kwargs)
            ]
        )
    
    def forward(self, tQ: Tensor, tH: Tensor, tA: Tensor, Q: Tensor | None = None, H: Tensor | None = None, A: Tensor | None = None, batch_size: int = None, seq_length: int = None,is_first:bool=True):
        """
        Noneになっているテンソルの入力を埋めて返す。Noneではない入力はそのまま返ってくる。
        """
        rec_kwargs = {
            'tQ': tQ,
            'tH': tH,
            'tA': tA,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'is_first': False
        }
        Qnone = Q is None
        Hnone = H is None
        Anone = A is None
        if not Qnone and not Hnone and not Anone:  # すべて与えられている
            return Q, H, A
        if is_first:
            tQ = self.q_layers[0].forward(tQ)
            tH = self.h_layers[0].forward(tH)
            tA = self.a_layers[0].forward(tA)
        if not Qnone:
            Qc = self.q_layers[0].forward(Q)
        if not Hnone:
            Hc = self.h_layers[0].forward(H)
        if not Anone:
            Ac = self.a_layers[0].forward(A)
        if (not Qnone) ^ (not Hnone) ^ (not Anone):
            """
            print(f"Q is None ? : {Qnone}")
            print(f"H is None ? : {Hnone}")
            print(f"A is None ? : {Anone}")
            """
            # いずれか１つだけ与えられている
            if not Qnone:
                # H,AをQから推測
                H = self.q_layers[2].forward(Qc, tQ, tH)
                # print(f"Gen H from Q : {H.shape}")
                A = self.q_layers[3].forward(Qc, tQ, tA)
                # print(f"Gen A from Q : {A.shape}")
                _, _, Ap = self.forward(Q=Q, H=H, **rec_kwargs)
                _, Hp, _ = self.forward(Q=Q, A=A, **rec_kwargs)
                A = torch.mean(torch.stack((A, Ap), dim=-2), dim=-2)
                H = torch.mean(torch.stack((H, Hp), dim=-2), dim=-2)
            elif not Hnone:
                # Q,AをHから推測
                Q = self.h_layers[2].forward(Hc, tH, tQ)
                A = self.h_layers[3].forward(Hc, tH, tA)
                _, _, Ap = self.forward(Q=Q, H=H, **rec_kwargs)
                Qp, _, _ = self.forward(H=H, A=A, **rec_kwargs)
                Q = torch.mean(torch.stack((Q, Qp), dim=-2), dim=-2)
                A = torch.mean(torch.stack((A, Ap), dim=-2), dim=-2)
            elif not Anone:
                # Q,HをAから推測
                Q = self.a_layers[2].forward(Ac, tA, tQ)
                H = self.a_layers[3].forward(Ac, tA, tH)
                _, Hp, _ = self.forward(Q=Q, A=A, **rec_kwargs)
                Qp, _, _ = self.forward(H=H, A=A, **rec_kwargs)
                Q = torch.mean(torch.stack((Q, Qp), dim=-2), dim=-2)
                H = torch.mean(torch.stack((H, Hp), dim=-2), dim=-2)
        else:  # すべて与えられていないか、いずれか２つが与えられている
            if not Qnone or not Hnone or not Anone:  # いずれか２つが与えられている
                if Anone:
                    A = self.a_layers[1].forward(Qc, Hc, tQ, tH, tA)
                elif Hnone:
                    H = self.h_layers[1].forward(Qc, Ac, tQ, tA, tH)
                else:
                    Q = self.q_layers[1].forward(Hc, Ac, tH, tA, tQ)
            else:  # すべて与えられていない
                print("Genarate All")
                Q = torch.randn(batch_size, seq_length,
                                self.qdim, requires_grad=True)
                # print(f"Gen Q : {Q.shape}")
                _, Hp1, Ap1 = self.forward(Q=Q, **rec_kwargs)
                H = torch.randn(batch_size, seq_length,
                                self.hdim, requires_grad=True)
                Qp1, _, Ap2 = self.forward(H=H, **rec_kwargs)
                A = torch.randn(batch_size, seq_length,
                                self.adim, requires_grad=True)
                Qp2, Hp2, _ = self.forward(A=A, **rec_kwargs)
                Q = torch.mean(torch.stack((Q, Qp1, Qp2), dim=-2), dim=-2)
                H = torch.mean(torch.stack((H, Hp1, Hp2), dim=-2), dim=-2)
                A = torch.mean(torch.stack((A, Ap1, Ap2), dim=-2), dim=-2)
        return Q, H, A

class DeductionNetwork(Module):
    def __init__(self, qdim: int, hdim: int, adim: int, embed_dim: int, table_size: int, num_layers: int, num_heads: int, bias: bool = True, dropout: float = .0, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'requires_grad': True,
                          'device': device, 'dtype': dtype}
        self.InternalParameters: list[Parameter] = nn.ParameterList(
            [
                Parameter(torch.empty(1, table_size, qdim,
                          **factory_kwargs), True),
                Parameter(torch.empty(1, table_size, hdim,
                          **factory_kwargs), True),
                Parameter(torch.empty(1, table_size, adim,
                          **factory_kwargs), True)
            ]
        )
        self.update_weights = Parameter(
            torch.ones(3,num_layers, **factory_kwargs)*.5, True)
        self.pre_layer = DeductionNetworkStartLayer(
            qdim=qdim,
            hdim=hdim,
            adim=adim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout
        )
        dlayer_kwargs = {
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'dropout': dropout
        }
        self.main_layers: list[DeductionNetworkLayer] = nn.ModuleList(
            [
                DeductionNetworkLayer(qdim, hdim, adim, **dlayer_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.reset_parameters()
    def reset_parameters(self) -> None:
        for param in self.InternalParameters:
            init.ones_(param)

    def forward(self, Q: Tensor | None, H: Tensor | None, A: Tensor | None, batch_size: int = None, seq_len: int = None):
        assert not (Q == None and H == None and A == None and batch_size ==
                    None), "If you want to all parameters are None, please at least set any number to batch_size (ex. 1)"
        batch_size, seq_len, _ = Q.shape if Q is not None else H.shape if H is not None else A.shape if A is not None else (
            batch_size, seq_len, 0)
        tQ, tH, tA = [Param.expand(batch_size, -1, -1)
                      for Param in self.InternalParameters]
        Q, H, A = self.pre_layer.forward(
            tQ, tH, tA, Q, H, A, batch_size, seq_len
        )
        for i,l in enumerate(self.main_layers):
            nQ, nH, nA = l.forward(
                Q, H, A,
                tQ, tH, tA
            )
            update_weight = self.update_weights[:,i]
            Q = (1.-update_weight[0])*Q + update_weight[0]*nQ
            H = (1.-update_weight[1])*H + update_weight[1]*nH
            A = (1.-update_weight[2])*A + update_weight[2]*nA
        return Q, H, A


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
def createModData(dataset: int = 100, max_val: float = 100, normalize_base: float = 100):
    # A % B = C ができるかどうか
    # Q : A, K : B, V : C
    A = np.random.randint(1, int(max_val), dataset)
    B = np.asarray([np.random.randint(1, a+1) for a in A])
    C = A % B
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


def createUnevenVectorData(dataset, dims: tuple[int, int, int], even_size: int, uneven_size: int, evenPair: tuple[int, int] = (0, 1)):
    # A,B,Cの対応するもの以外の形状が異なっても学習可能かどうか
    # Q : A, K : B, V : C
    A = np.random.rand(
        dataset, even_size if 0 in evenPair else uneven_size, dims[0])
    B = np.random.rand(
        dataset, even_size if 1 in evenPair else uneven_size, dims[1])
    C = np.random.rand(
        dataset, even_size if 2 in evenPair else uneven_size, dims[2])
    return torch.tensor(A).float(), torch.tensor(B).float(), torch.tensor(C).float()

def cor(X:Tensor,Y:Tensor):
    return F.softmax((X.transpose(-2, -1) @ Y)/sqrt(Y.shape[-2]),dim=-1)

def train(epochs: int = 100):
    data_sets = 100
    # Q, K, V = createAddingData(data_sets)
    # print(Q, K, V)
    DedN = DeductionNetwork(
        qdim=2,
        hdim=2,
        adim=2,
        embed_dim=4,
        table_size=2,
        num_layers=4,
        num_heads=8, dropout=.1)
    # 特徴次元の４倍くらいの大きさの埋め込み次元があれば良さそう（ヒューリスティック）
    # テーブルサイズを変えてもあまり結果は変わらない。計算時間は微妙に伸びるし収束が遅くなる。変化量が小さくなる。必要最低限でいい。
    # テーブルサイズの大きさに比例して実行時間が伸びる、テーブルサイズが小さいと精度が下がるかと思われたがそうでもない
    # テーブルサイズが小さいほど精度が上がる！？？？→上がらない・限界をすぐ迎える
    # テーブルサイズ0でも動く→必要なものさえ与えられていれば
    # 結論：特徴次元が大事
    # テーブルサイズを上げると時間が経つにつれて大きく誤差が減るようになって精度がいい...というわけでもなさそう
    optimizer = optim.AdamW(DedN.parameters(), lr=.01)
    total_errs = []  # mean_errを保存するリストを作成
    a_errs = []
    h_errs = []
    q_errs = []
    sim_errs = []
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        # ちゃんと処理できるなら動的なサイズでも学習できる
        # spl = randint(2,4) # 動的なサイズ
        # print(f"shape {spl}") # 動的なサイズ
        # createUnevenVectorData(data_sets, (5,3,2), spl*2, 2, (0, 2))
        Q, H, A = createSubVectorData(data_sets)
        # createModData(
        #    data_sets, (data_sets+epoch)*100, (data_sets+epochs)*100)  # 過適合対策
        optimizer.zero_grad()
        # 無から生み出すやつ, 適当に数字あげて違うって言われてるみたいなもの（精度保証不可）
        # lQ, lK, lV = DedN.forward(
        #    None, None, None, Q.shape[0] # 全部Noneのときはバッチサイズだけ入れとく
        # )
        # １つの情報から生み出すやつ, 不十分な質問されて質問を補完させて解答を出すみたいなもの
        # 　Q → K生成 →　Qと生成したKでV生成 → 生成したK,VからQ生成
        # _, lK, lV = DedN.forward(
        #    Q, None, None
        # )
        # lQ, _, _ = DedN.forward(
        #    None, lK, lV
        # )
        # 演驛するやつ, 不十分な質問と解答見せて、質問を補完させるみたいなもの
        # _, lK, _ = DedN.forward(
        #    Q, None, V
        # )
        # _, _, lV = DedN.forward(
        #    Q, lK, None
        # )
        # lQ, _, _ = DedN.forward(
        #    None, lK, V
        # )
        # lK の形状だけ違うので、いい感じに使って関連性持たせてあげる（Kだけ学習しないというのも手）
        # lK = torch.cat((torch.mean(lK[:, spl:, :], dim=1, keepdim=True), torch.mean(lK[:, :spl, :], dim=1, keepdim=True)),dim=1)
        # 通常パターン, 穴埋め解かせるだけ
        _, lH, _ = DedN.forward(
            Q, None, A
        )
        _, _, lA = DedN.forward(
            Q, H, None
        )
        lQ, _, _ = DedN.forward(
            None, H, A
        )
        gQ, gH, gA = DedN.forward(
            None, None, None, A.shape[0],A.shape[1]
        )
        err_fn = nn.L1Loss()
        a_err = err_fn.forward(
            lA, A
        ) * data_sets
        h_err = err_fn.forward(
            lH, H
        ) * data_sets
        q_err = err_fn.forward(
            lQ, Q
        ) * data_sets
        sim_err = err_fn(cor(gQ, gH), cor(Q, H))+err_fn(
            cor(gQ, gA), cor(Q, A))+err_fn(cor(gH, gA), cor(H, A))*data_sets
        total_err = a_err+h_err+q_err+sim_err
        total_err.backward()
        torch.nn.utils.clip_grad_norm_(DedN.parameters(), 4.0)
        optimizer.step()
        total_errs.append(total_err.item()/data_sets)
        a_errs.append(a_err.item()/data_sets)
        h_errs.append(h_err.item()/data_sets)
        q_errs.append(q_err.item()/data_sets)
        sim_errs.append(sim_err.item()/data_sets)
        print(
            f"Total Error = {total_errs[-1]}\nQ:{q_errs[-1]}\nH:{h_errs[-1]}\nA:{a_errs[-1]}\nSim:{sim_errs[-1]}"
        )
    # エポックごとの誤差をプロット
    plt.plot(total_errs, label='Total Error', color='m')
    plt.plot(q_errs, label='Q Error', color='r')
    plt.plot(h_errs, label='H Error', color='g')
    plt.plot(a_errs, label='A Error', color='b')
    plt.plot(sim_errs, label='Sim Error', color='c')
    plt.plot(np.diff(total_errs), label='Total Error Grad', color='gold')
    # plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid()
    plt.show()
    return DedN


model = train(100)
# testQ, testK, testV = createModData(1, 10000, 10000)
# Convert_ONNX(model,(testQ,None,testV),['QIn','KIn','VIn'],['QOut','KOut','Vout'],'DeductionNetwork')
