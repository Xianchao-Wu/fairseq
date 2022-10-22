# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum, auto
import math
import numpy as np
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from fairseq import checkpoint_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    SamePad,
    TransposeLast,
)


class SegmentationType(Enum):
    NONE = auto()
    RANDOM = auto()
    UNIFORM_RANDOM = auto()
    UNIFORM_RANDOM_JOIN = auto()
    JOIN = auto()


@dataclass
class SegmentationConfig(FairseqDataclass):
    type: SegmentationType = SegmentationType.NONE
    subsample_rate: float = 0.25
    mean_pool: bool = True
    mean_pool_join: bool = False
    remove_zeros: bool = False


@dataclass
class Wav2vec_UConfig(FairseqDataclass):
    discriminator_kernel: int = 3
    discriminator_dilation: int = 1
    discriminator_dim: int = 256
    discriminator_causal: bool = True
    discriminator_linear_emb: bool = False
    discriminator_depth: int = 1
    discriminator_max_pool: bool = False
    discriminator_act_after_linear: bool = False
    discriminator_dropout: float = 0.0
    discriminator_spectral_norm: bool = False
    discriminator_weight_norm: bool = False

    generator_kernel: int = 4
    generator_dilation: int = 1
    generator_stride: int = 1
    generator_pad: int = -1
    generator_bias: bool = False
    generator_dropout: float = 0.0
    generator_batch_norm: int = 0
    generator_residual: bool = False

    blank_weight: float = 0
    blank_mode: str = "add"
    blank_is_sil: bool = False
    no_softmax: bool = False

    smoothness_weight: float = 0.0
    smoothing: float = 0.0
    smoothing_one_sided: bool = False
    gradient_penalty: float = 0.0
    probabilistic_grad_penalty_slicing: bool = False
    code_penalty: float = 0.0
    mmi_weight: float = 0.0
    target_dim: int = 64
    target_downsample_rate: int = 2
    gumbel: bool = False
    hard_gumbel: bool = True
    temp: Tuple[float, float, float] = (2, 0.1, 0.99995)
    input_dim: int = 128

    segmentation: SegmentationConfig = SegmentationConfig()


class Segmenter(nn.Module):
    cfg: SegmentationConfig

    def __init__(self, cfg: SegmentationConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample_rate = cfg.subsample_rate

    def pre_segment(self, dense_x, dense_padding_mask):
        return dense_x, dense_padding_mask

    def logit_segment(self, logits, padding_mask):
        return logits, padding_mask


class RandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        target_num = math.ceil(dense_x.size(1) * self.subsample_rate)
        ones = torch.ones(dense_x.shape[:-1], device=dense_x.device)
        indices, _ = ones.multinomial(target_num).sort(dim=-1)
        indices_ld = indices.unsqueeze(-1).expand(-1, -1, dense_x.size(-1))
        dense_x = dense_x.gather(1, indices_ld)
        dense_padding_mask = dense_padding_mask.gather(1, index=indices)
        return dense_x, dense_padding_mask


class UniformRandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        bsz, tsz, fsz = dense_x.shape

        target_num = math.ceil(tsz * self.subsample_rate)

        rem = tsz % target_num

        if rem > 0:
            dense_x = F.pad(dense_x, [0, 0, 0, target_num - rem])
            dense_padding_mask = F.pad(
                dense_padding_mask, [0, target_num - rem], value=True
            )

        dense_x = dense_x.view(bsz, target_num, -1, fsz)
        dense_padding_mask = dense_padding_mask.view(bsz, target_num, -1)

        if self.cfg.mean_pool:
            dense_x = dense_x.mean(dim=-2)
            dense_padding_mask = dense_padding_mask.all(dim=-1)
        else:
            ones = torch.ones((bsz, dense_x.size(2)), device=dense_x.device)
            indices = ones.multinomial(1)
            indices = indices.unsqueeze(-1).expand(-1, target_num, -1)
            indices_ld = indices.unsqueeze(-1).expand(-1, -1, -1, fsz)
            dense_x = dense_x.gather(2, indices_ld).reshape(bsz, -1, fsz)
            dense_padding_mask = dense_padding_mask.gather(2, index=indices).reshape(
                bsz, -1
            )
        return dense_x, dense_padding_mask


class JoinSegmenter(Segmenter):
    def logit_segment(self, logits, padding_mask):
        import ipdb; ipdb.set_trace()
        preds = logits.argmax(dim=-1) # generator的预测结果
        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad, 只保留预测的有效位置
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True) # 合并同类项, 1 1 -> 1
            )

        new_tsz = max(u[0].numel() for u in uniques) # (合并同类项之后：)当前batch中的最长有效长度
        new_logits = logits.new_zeros(bsz, new_tsz, csz) # 因为长度变短了（合并了），所以重新生成new_logits
        new_pad = padding_mask.new_zeros(bsz, new_tsz) # 这是全0的一个padding matrix

        for b in range(bsz): # 对当前batch中每个序列循环
            u, idx, c = uniques[b]
            keep = u != -1 # pad.id=-1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join:
                u[0] = 0
                u[1:] = c.cumsum(0)[:-1]
                m = c > 1
                r = torch.rand(m.sum()) # m.sum()代表有重复的地方的数量，随机生成这么多(0,1)之间的随机变量
                o = (c[m] * r).long() # 对重复数采样，例如c[m]=2, r=0.3, 则0.6 -> 0; c[m]=2, r=0.6, 则1.2->1。选择出来的0或者1，代表new_logits用哪个位置的old_logit:
                u[m] += o # 采样之后，更新索引，决定重复位置的哪个具体位置的logits扔给new_logits:
                new_logits[b, : u.numel()] = logits[b, u] # b = sequence index in a batch; 赋值之前new_logits是全0的张量, [batch.size, new.batch.max.seq.len, phoneme.vocab.size]
            else:
                new_logits[b].index_add_(
                    dim=0, index=idx.to(new_logits.device), source=logits[b]
                ) # NOTE 重复位置的logits求和
                new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device) # 求和之后，取平均

            new_sz = keep.sum() # 当前序列的更新后的长度
            if not keep.all(): # NOTE 这块没啥变化啊。。。有用吗?
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz # 本batch的序列最长长度 - 本序列的长度=需要pad的位置的数量
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad # NOTE 需要特别注意的是，这里的new_logits是有grad_fn的，即：支持梯度反向传播！


class UniformRandomJoinSegmenter(UniformRandomSegmenter, JoinSegmenter):
    pass


SEGMENT_FACTORY = {
    SegmentationType.NONE: Segmenter,
    SegmentationType.RANDOM: RandomSegmenter,
    SegmentationType.UNIFORM_RANDOM: UniformRandomSegmenter,
    SegmentationType.UNIFORM_RANDOM_JOIN: UniformRandomJoinSegmenter,
    SegmentationType.JOIN: JoinSegmenter,
}


class Discriminator(nn.Module):
    def __init__(self, dim, cfg: Wav2vec_UConfig):
        super().__init__()

        inner_dim = cfg.discriminator_dim
        kernel = cfg.discriminator_kernel
        dilation = cfg.discriminator_dilation
        self.max_pool = cfg.discriminator_max_pool

        if cfg.discriminator_causal:
            padding = kernel - 1
        else:
            padding = kernel // 2

        def make_conv(in_d, out_d, k, p=0, has_dilation=True):
            conv = nn.Conv1d(
                in_d,
                out_d,
                kernel_size=k,
                padding=p,
                dilation=dilation if has_dilation else 1,
            )
            if cfg.discriminator_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            elif cfg.discriminator_weight_norm:
                conv = nn.utils.weight_norm(conv)
            return conv

        inner_net = [
            nn.Sequential(
                make_conv(inner_dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
                nn.Dropout(cfg.discriminator_dropout),
                nn.GELU(),
            )
            for _ in range(cfg.discriminator_depth - 1)
        ] + [
            make_conv(inner_dim, 1, kernel, padding, has_dilation=False),
            SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
        ]

        if cfg.discriminator_linear_emb:
            emb_net = [make_conv(dim, inner_dim, 1)]
        else:
            emb_net = [
                make_conv(dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
            ]

        if cfg.discriminator_act_after_linear:
            emb_net.append(nn.GELU())

        self.net = nn.Sequential(
            *emb_net,
            nn.Dropout(cfg.discriminator_dropout),
            *inner_net,
        )

    def forward(self, x, padding_mask):
        import ipdb; ipdb.set_trace() # discriminator's forward
        x = x.transpose(1, 2)  # BTC -> BCT
        x = self.net(x) # [B, 1, T]
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1) # 获取每个序列的真实长度 = T - padded.num
        x = x.squeeze(-1)
        if self.max_pool:
            x, _ = x.max(dim=-1)
        else:
            x = x.sum(dim=-1) # here, 一行的所有有效位置的概率之和
            x = x / x_sz # 按照每个序列的长度进行求平均
        return x # [B]


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, cfg: Wav2vec_UConfig):
        super().__init__()

        self.cfg = cfg
        self.output_dim = output_dim
        self.stride = cfg.generator_stride
        self.dropout = nn.Dropout(cfg.generator_dropout)
        self.batch_norm = cfg.generator_batch_norm != 0
        self.residual = cfg.generator_residual

        padding = (
            cfg.generator_kernel // 2 if cfg.generator_pad < 0 else cfg.generator_pad # NOTE default=-1
        ) # 目前的generator的padding是动态计算出来的
        self.proj = nn.Sequential(
            TransposeLast(),
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=cfg.generator_kernel,
                stride=cfg.generator_stride,
                dilation=cfg.generator_dilation,
                padding=padding,
                bias=cfg.generator_bias,
            ),
            TransposeLast(),
        )

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(input_dim)
            self.bn.weight.data.fill_(cfg.generator_batch_norm)
        if self.residual:
            self.in_proj = nn.Linear(input_dim, input_dim)

    def forward(self, dense_x, tokens, dense_padding_mask):
        import ipdb; ipdb.set_trace()
        result = {}
        if self.batch_norm:
            dense_x = self.bn_padded_data(dense_x, dense_padding_mask)
        if self.residual:
            inter_x = self.in_proj(self.dropout(dense_x))
            dense_x = dense_x + inter_x
            result["inter_x"] = inter_x

        dense_x = self.dropout(dense_x)
        import ipdb; ipdb.set_trace()
        dense_x = self.proj(dense_x) # NOTE the major module for the 'generator'
        if self.stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride]

        if dense_padding_mask.size(1) != dense_x.size(1):
            new_padding = dense_padding_mask.new_zeros(dense_x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)

            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding

        token_x = None
        if tokens is not None:
            token_x = dense_x.new_zeros(tokens.numel(), self.output_dim)
            token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
            token_x = token_x.view(tokens.shape + (self.output_dim,)) # 这是0,1化了

        result["dense_x"] = dense_x # generator's predicted output, 长度是wave frame的数量，有可能多个连续的frame对应到一个phoneme
        result["token_x"] = token_x # "pseudo reference" random labels (phoneme sequence batch)
        result["dense_padding_mask"] = dense_padding_mask # mask on dense_x, 是对wave frame长度的masking in one batch
        import ipdb; ipdb.set_trace()
        return result

    def bn_padded_data(self, feature, padding_mask):
        normed_feature = feature.clone()
        normed_feature[~padding_mask] = self.bn(
            feature[~padding_mask].unsqueeze(-1)
        ).squeeze(-1)
        return normed_feature


@register_model("wav2vec_u", dataclass=Wav2vec_UConfig)
class Wav2vec_U(BaseFairseqModel):
    def calc_gradient_penalty(self, real_data, fake_data):
        import ipdb; ipdb.set_trace()
        b_size = min(real_data.size(0), fake_data.size(0))
        t_size = min(real_data.size(1), fake_data.size(1))

        if self.cfg.probabilistic_grad_penalty_slicing:

            def get_slice(data, dim, target_size):

                size = data.size(dim)
                diff = size - target_size
                if diff <= 0:
                    return data

                start = np.random.randint(0, diff + 1)
                return data.narrow(dim=dim, start=start, length=target_size)

            real_data = get_slice(real_data, 0, b_size)
            real_data = get_slice(real_data, 1, t_size)
            fake_data = get_slice(fake_data, 0, b_size)
            fake_data = get_slice(fake_data, 1, t_size)

        else:
            real_data = real_data[:b_size, :t_size]
            fake_data = fake_data[:b_size, :t_size]

        alpha = torch.rand(real_data.size(0), 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self.discriminator(interpolates, None)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=real_data.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        import ipdb; ipdb.set_trace()
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def discrim_step(self, num_updates):
        return num_updates % 2 == 1 # discriminator step or generator step NOTE

    def get_groups_for_update(self, num_updates):
        return "discriminator" if self.discrim_step(num_updates) else "generator"

    def __init__(self, cfg: Wav2vec_UConfig, target_dict):
        super().__init__()

        self.cfg = cfg
        self.zero_index = target_dict.index("<SIL>") if "<SIL>" in target_dict else 0
        self.smoothness_weight = cfg.smoothness_weight

        output_size = len(target_dict)
        self.pad = target_dict.pad()
        self.eos = target_dict.eos()
        self.smoothing = cfg.smoothing
        self.smoothing_one_sided = cfg.smoothing_one_sided
        self.no_softmax = cfg.no_softmax
        self.gumbel = cfg.gumbel
        self.hard_gumbel = cfg.hard_gumbel
        self.last_acc = None

        self.gradient_penalty = cfg.gradient_penalty
        self.code_penalty = cfg.code_penalty
        self.mmi_weight = cfg.mmi_weight
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.blank_index = target_dict.index("<SIL>") if cfg.blank_is_sil else 0
        assert self.blank_index != target_dict.unk()

        self.discriminator = Discriminator(output_size, cfg) # NOTE
        for p in self.discriminator.parameters():
            p.param_group = "discriminator"

        self.pca_A = self.pca_b = None
        d = cfg.input_dim

        self.segmenter = SEGMENT_FACTORY[cfg.segmentation.type](cfg.segmentation)

        self.generator = Generator(d, output_size, cfg) # NOTE

        for p in self.generator.parameters():
            p.param_group = "generator"

        for p in self.segmenter.parameters():
            p.param_group = "generator"

        self.max_temp, self.min_temp, self.temp_decay = cfg.temp
        self.curr_temp = self.max_temp
        self.update_num = 0

        if self.mmi_weight > 0:
            self.target_downsample_rate = cfg.target_downsample_rate
            self.decoder = nn.Linear(d, cfg.target_dim)
            for p in self.decoder.parameters():
                p.param_group = "generator"

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task.target_dictionary)

    def get_logits(
        self,
        net_output: Optional[Dict[str, List[Optional[torch.Tensor]]]],
        normalize: bool = False,
    ):
        logits = net_output["logits"]

        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., self.blank_index] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., self.blank_index] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        padding = net_output["padding_mask"]
        if padding.any():
            logits[padding] = float("-inf")
            logits[padding][..., self.blank_index] = float("inf")

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits.transpose(0, 1)

    def get_normalized_probs(
        self,
        net_output: Tuple[
            torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]
        ],
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ):
        logits = self.get_logits(net_output)

        probs = super().get_normalized_probs(logits, log_probs, sample)
        # BTC -> TBC for ctc
        probs = probs.transpose(0, 1)
        return probs

    def normalize(self, dense_x):

        bsz, tsz, csz = dense_x.shape

        if dense_x.numel() == 0:
            raise Exception(dense_x.shape)
        _, k = dense_x.max(-1) # k.shape=[bsz, tsz], 每个序列中，每个frame所对应的最大可能的phoneme id (43个候选, 0 to 42)
        hard_x = (
            dense_x.new_zeros(bsz * tsz, csz) # all 0
            .scatter_(-1, k.view(-1, 1), 1.0) # 根据k的指引，把对应位置设置为1.0，其他位置为0
            .view(-1, csz) # [bsz * tsz, csz]
        )
        hard_probs = torch.mean(hard_x.float(), dim=0) # [csz=43], 对于每个phoneme，其在bsz*tsz个位置上出现的概率（只能是0或者1 NOTE）的均值
        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1) # 目标phoneme词表的“乱度”
        )

        avg_probs = torch.softmax(dense_x.reshape(-1, csz).float(), dim=-1).mean(dim=0) # 这是使用“真实”概率得分得到的，目标phoneme词表的"概率"的在bsz*tsz个位置上的均值. 本质上是衡量词表中所有词被使用的程度--我们希望它们尽可能被“均匀”使用---问题，不应该是和english语言本身的phoneme的使用程度概率分布，正相关吗？ perplexity=e^H(p), 希望H(p)越大越好！所以这里的意思是，希望code_perplexity和prob_perplexity越大越好！ 越大，越均匀... 
        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        )
        # 例子1：p=[0.5, 0.5], perp=2; 例子2：p=[0.25, 0.75], perp=1.7548
        if not self.no_softmax:
            if self.training and self.gumbel:
                dense_x = F.gumbel_softmax(
                    dense_x.float(), tau=self.curr_temp, hard=self.hard_gumbel
                ).type_as(dense_x)
            else:
                dense_x = dense_x.softmax(-1) # here

        return dense_x, code_perplexity, prob_perplexity

    def forward(
        self,
        features,
        padding_mask,
        random_label=None, # pad.id=1
        dense_x_only=False,
        segment=True, # 合并同类项
        aux_target=None,
    ):
        import ipdb; ipdb.set_trace() # Wav2vec_U's forward
        if segment:
            features, padding_mask = self.segmenter.pre_segment(features, padding_mask)
        orig_size = features.size(0) * features.size(1) - padding_mask.sum() # TODO not used... why?

        gen_result = self.generator(features, random_label, padding_mask)

        orig_dense_x, token_x = gen_result["dense_x"], gen_result["token_x"]
        orig_dense_padding_mask = gen_result["dense_padding_mask"]

        if segment:
            dense_x, dense_padding_mask = self.segmenter.logit_segment( # NOTE TODO what for? 合并phoneme相同的“同类项”，例如1 1 -> 1，然后修正logis和padding（更短了）
                orig_dense_x, orig_dense_padding_mask
            )
        else:
            dense_x = orig_dense_x
            dense_padding_mask = orig_dense_padding_mask

        dense_logits = dense_x
        prob_perplexity = None
        code_perplexity = None

        if not (self.no_softmax and dense_x_only):
            dense_x, code_perplexity, prob_perplexity = self.normalize(dense_logits) # NOTE

        if dense_x_only or self.discriminator is None:
            return {
                "logits": dense_x,
                "padding_mask": dense_padding_mask,
            }

        token_padding_mask = random_label == self.pad # self.pad=1

        dense_y = self.discriminator(dense_x, dense_padding_mask) # generator产出的预测结果, shape=[B=160]
        token_y = self.discriminator(token_x, token_padding_mask) # 生成器generator输出的对random_labels的后处理结果 0,1化之后的了, shape=[B=160]

        sample_size = features.size(0)

        d_step = self.discrim_step(self.update_num) # NOTE d_step=true=discriminator, or false=generator，判定当前是生成器步骤，还是判别器步骤. 偶数0,2...对应生成器；奇数1,3,5...对应判别器

        fake_smooth = self.smoothing
        real_smooth = self.smoothing
        if self.smoothing_one_sided:
            fake_smooth = 0

        zero_loss = None
        smoothness_loss = None
        code_pen = None
        mmi_loss = None
        import ipdb; ipdb.set_trace()
        if d_step: # for discriminator
            loss_dense = F.binary_cross_entropy_with_logits(
                dense_y, # generator产出的结果，经过discriminator判定得到的logits
                dense_y.new_ones(dense_y.shape) - fake_smooth,
                reduction="sum",
            )
            loss_token = F.binary_cross_entropy_with_logits(
                token_y, # pseudo reference, from random labels，随机挑选的所谓“参考答案”
                token_y.new_zeros(token_y.shape) + real_smooth,
                reduction="sum",
            )
            if self.training and self.gradient_penalty > 0:
                grad_pen = self.calc_gradient_penalty(token_x, dense_x)
                grad_pen = grad_pen.sum() * self.gradient_penalty
            else:
                grad_pen = None
        else: # for generator
            grad_pen = None # gradient penalty
            loss_token = None
            loss_dense = F.binary_cross_entropy_with_logits(
                dense_y,
                dense_y.new_zeros(dense_y.shape) + fake_smooth, # 这是希望来自'生成器'的预测结果越接近0越好, 越接近0，则二元交叉熵的值越小
                reduction="sum",
            )
            num_vars = dense_x.size(-1) # phoneme词表大小
            if prob_perplexity is not None:
                code_pen = (num_vars - prob_perplexity) / num_vars
                code_pen = code_pen * sample_size * self.code_penalty

            if self.smoothness_weight > 0:
                smoothness_loss = F.mse_loss(
                    dense_logits[:, :-1], dense_logits[:, 1:], reduction="none"
                ) # Segment smoothness penalty in the paper (equation 7) arxiv:2105.11084v3; NOTE
                smoothness_loss[dense_padding_mask[:, 1:]] = 0
                smoothness_loss = (
                    smoothness_loss.mean() * sample_size * self.smoothness_weight
                )

            if (self.mmi_weight > 0) and (aux_target is not None): # not in NOTE, w2vu2.yaml中，有self.mmi_weight
                inter_x = self.decoder(gen_result["inter_x"])
                if self.target_downsample_rate > 1:
                    aux_target = aux_target[:, :: self.target_downsample_rate]
                max_t_len = min(aux_target.shape[1], inter_x.shape[1])
                mmi_loss = F.cross_entropy(
                    inter_x[:, :max_t_len].transpose(1, 2),
                    aux_target[:, :max_t_len],
                    ignore_index=-1,
                    reduction="none",
                )
                mmi_loss = mmi_loss.mean() * mmi_loss.shape[0] * self.mmi_weight

        result = {
            "losses": {
                "grad_pen": grad_pen, # g: none
                "code_pen": code_pen, # g: 音素多样性loss
                "smoothness": smoothness_loss, # g: 切片平滑惩罚,segment smoothness penalty
                "mmi": mmi_loss, # not in yet wav2vec-u1.0
            },
            "temp": self.curr_temp, # g: temperature
            "code_ppl": code_perplexity, # g: hardcode perp
            "prob_ppl": prob_perplexity, # g: soft prob perp
            "d_steps": int(d_step), # g: 0,2,...
            "sample_size": sample_size,
        }
        import ipdb; ipdb.set_trace()
        suff = "_d" if d_step else "_g"
        result["losses"]["dense" + suff] = loss_dense
        result["losses"]["token" + suff] = loss_token

        return result

