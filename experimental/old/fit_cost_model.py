"""
An old script for fitting the cost model for OPT in FlexGen.

Warning:
The script has not been cleaned for release.
It has been placed here for study purposes only. There is no promise of reproduction.
"""

import argparse
import math
import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass

# def get_filename(model_name, batch_size, prompt_len, gen_len,
#                  cpu_offload, disk_offload, num_nodes, num_gpus_per_node,
#                  use_deepspeed):
#     modelsize = model_name.split('-')[-1]
#     if use_deepspeed:
#         filename = "ds-"
#     else:
#         filename = "hf-"
#     filename += f"{modelsize}-bs{batch_size}-prompt{prompt_len}-gen{gen_len}-"
#     filename += f"n{num_nodes}x{num_gpus_per_node}-"
#     if cpu_offload:
#         filename += "cpu"
#     elif disk_offload:
#         filename += "disk"
#     else:
#         filename += "gpu"
#     return filename

def get_filename(case):
    tokens = case.command.split()

    model_name = tokens[tokens.index('--model') + 1]
    batch_size = int(tokens[tokens.index('--gpu-batch-size') + 1])
    prompt_len = int(tokens[tokens.index('--prompt-len') + 1]) if '--prompt-len' in tokens else 256  # Default value if not specified
    gen_len = int(tokens[tokens.index('--gen-len') + 1]) if '--gen-len' in tokens else 32  # Default value if not specified
    use_deepspeed = 'deepspeed' in case.command

    cpu_offload = '--cpu' in case.command
    disk_offload = case.use_page_maga if hasattr(case, 'use_page_maga') else False

    num_nodes = 1
    num_gpus_per_node = 1

    modelsize = model_name.split('-')[-1]
    filename = "ds-" if use_deepspeed else "hf-"
    filename += f"{modelsize}-bs{batch_size}-prompt{prompt_len}-gen{gen_len}-"
    filename += f"n{num_nodes}x{num_gpus_per_node}-"
    if cpu_offload:
        filename += "cpu"
    elif disk_offload:
        filename += "disk"
    else:
        filename += "gpu"
    return filename

# from experiments.run_exp import ExpConfig, cases, get_filename
from flexgen.opt_config import get_opt_config
from flexgen.utils import GB, T

class CostModel(nn.Module):
    def __init__(self):
        super(CostModel, self).__init__()
        a = torch.abs(torch.rand([]))
        b = torch.abs(torch.rand([]))

        self.ctog_bdw        = nn.Parameter(a)
        self.gtoc_bdw_cache  = nn.Parameter(a)
        self.gtoc_bdw_hidden = nn.Parameter(a)

        self.dtoc_bdw          = nn.Parameter(a)
        self.ctod_bdw_cache_p  = nn.Parameter(a)
        self.ctod_bdw_hidden_p = nn.Parameter(a)
        self.ctod_bdw_g        = nn.Parameter(a)

        self.mm_flops_p  = nn.Parameter(b)
        self.mm_flops_g  = nn.Parameter(b)
        self.bmm_flops_p = nn.Parameter(b)
        self.bmm_flops_g = nn.Parameter(b)
        self.cpu_flops   = nn.Parameter(b)

        self.c0 = nn.Parameter(torch.tensor(0.0))
        self.c1 = nn.Parameter(torch.tensor(0.0))
        self.c2 = nn.Parameter(torch.tensor(0.0))
        self.c3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, xs):
        (wi, l, h1, h2, s, n,
         gbs, bls, wg, wc, wn, cg, cc, cn, hg, hc, hn) = xs.split(1, dim=1)

        ctogp = (self.ctog_bdw / GB) * (wi * (wc + wn)
                 + 2 * s * h1 * bls * (hc + hn))
        gtocp = (self.gtoc_bdw_cache / GB) * (4 * (s + 1) * h1 * bls * (cc + cn)) \
                 + (self.gtoc_bdw_hidden / GB) * 2 * s * h1 * bls * (hc + hn)
        dtocp = (self.dtoc_bdw / GB) * (wi * wn + 2 * s * h1 * bls * hn)
        ctodp = (self.ctod_bdw_cache_p / GB) * 4 * bls * (s + 1) * h1 * cn \
                  + (self.ctod_bdw_hidden_p / GB) * 2 * s * h1 * bls * hn
        compp = (self.mm_flops_p / T) * bls * (8 * s * h1 ** 2  + 4 * s * h1 * h2) \
                  + (self.bmm_flops_p / T) * 4 * bls * s ** 2 * h1
        tpre = torch.maximum(ctogp + dtocp, torch.maximum(gtocp + ctodp, compp))

        ctogg = (self.ctog_bdw / GB) * (wi * (wc + wn)
                  + 2 * h1 * bls * (hc + hn))
        gtocg = (self.gtoc_bdw_hidden / GB) * 2 * h1 * bls * (hc + hn)
        dtocg = (self.dtoc_bdw / GB) * (4 * bls * (s + n / 2) * h1 * cn
                                        + 2 * h1 * bls * hn) \
                  + (self.dtoc_bdw / GB / 0.95) * wi * wn 
        ctodg = (self.ctod_bdw_g / GB) * (4 * bls * h1 * cn
                                               + 2 * h1 * bls * hn)


        # non-linear cpu_flops
        cpu_flops_real = self.cpu_flops / torch.clamp(
            1 + self.c1 * torch.log2(64 / gbs).clamp(min=0) * torch.log2(4096 / h1).clamp(min=0)
            - self.c2 * torch.log2(64 / gbs).clamp(min=0)
            - self.c3 * torch.log2(4096 / h1).clamp(min=0),
            min=0.5)
        compg = (self.mm_flops_g / T) * bls * (8 * h1 ** 2 + 4 * h1 * h2) \
             + (self.bmm_flops_g / T) * 4 * bls * (s + n / 2) * h1 * cg \
             + (cpu_flops_real / T) * 4 * bls * (s + n / 2) * h1 * (cc + cn)

        #cpu_time_delta = (
        #    self.c0 +
        #    self.c1 * torch.log2(torch.clamp(gbs, max=64)) +
        #    self.c2 * torch.log2(torch.clamp(h1, max=4096)) +
        #    self.c3 * torch.log2(torch.clamp(gbs, max=64)) * torch.log2(torch.clamp(h1, max=4096))
        #)
        #compg = (self.mm_flops_g / T) * bls * (8 * h1 ** 2 + 4 * h1 * h2) \
        #     + (self.bmm_flops_g / T) * 4 * bls * (s + n / 2) * h1 * cg \
        #     + (self.cpu_flops / T) * 4 * bls * (s + n / 2) * h1 * (cc + cn) * (1 + cpu_time_delta) \

        tgen = ctogg + torch.maximum(gtocg,
                                     torch.maximum(dtocg,
                                     torch.maximum(ctodg, compg)))
        return torch.cat([tpre * l, tgen * (n - 1) * l], dim=1)



@dataclass
class Case:
    command: str
    name: str = ""
    use_page_maga: bool = False

    def __post_init__(self):
        self.parse_command()

    def parse_command(self):
        tokens = self.command.split()
        self.model_name = tokens[tokens.index('--model') + 1]
        self.gbs = int(tokens[tokens.index('--gpu-batch-size') + 1])
        self.prompt_len = int(tokens[tokens.index('--prompt-len') + 1]) if '--prompt-len' in tokens else 256
        self.gen_len = int(tokens[tokens.index('--gen-len') + 1]) if '--gen-len' in tokens else 32
        self.use_deepspeed = 'deepspeed' in self.command

        self.percent = self.extract_percentages(tokens)
        self.num_gpus_per_node = 1
        self.num_nodes = 1
        self.cpu_offload = '--cpu' in self.command
        self.disk_offload = self.use_page_maga
        
        # what is bls?
        self.bls = self.gbs * 1 # reference cost_model.py

    def extract_percentages(self, tokens):
        if '--percent' in tokens:
            index = tokens.index('--percent') + 1
            return list(map(int, tokens[index:index+6]))
        else:
            return [100, 0, 100, 0, 100, 0]  # Default?

suite_1b3_test = [
    # All GPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 100 0 100 0 --cut-gen-len 8", "All GPU"),
    # Weight on CPU, cache on GPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 0 100 100 0 100 0 --cut-gen-len 8", "Weight on CPU, cache on GPU"),
    # Weight on GPU, cache on CPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 0 100 100 0 --cut-gen-len 8 --cpu", "Weight on GPU, cache on CPU"),
    # Weight on CPU, cache on CPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 0 100 0 100 100 0 --cut-gen-len 8 --cpu", "Weight on CPU, cache on CPU"),
    # Weight on disk, cache on GPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 0 0 100 0 100 0 --cut-gen-len 8", "Weight on disk, cache on GPU", True),
    # Weight on GPU, cache on disk
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 0 0 100 0 --cut-gen-len 8 --cpu", "Weight on GPU, cache on disk", True),
    # Weight on CPU/GPU (50-50 split), cache on GPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 50 50 100 0 100 0 --cut-gen-len 8", "Weight on both CPU/GPU (50-50 split), cache on GPU"),
    # Weight on GPU, cache on CPU/GPU (50-50 split)
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 50 50 100 0 --cut-gen-len 8 --cpu", "Weight on GPU, cache on CPU/GPU (50-50 split)"),
    # Weight on GPU, cache on disk, sparse attention
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 0 0 100 0 --cut-gen-len 8 --cpu --attn-sparsity 0.1", "Weight on GPU, cache on disk, sparse attention", True),
    # Weight on GPU, cache on disk, cache quantization
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 0 0 100 0 --cut-gen-len 8 --compress-cache", "Weight on GPU, cache on disk, cache quantization", True),
    # All GPU, 2 GPU batches
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 100 0 100 0 --cut-gen-len 8 --num-gpu-batches 2", "All GPU, 2 gpu batches"),
]

suite_6b7_1x1 = [
    # seq_len = 256, gen_len = 32
    # 53.29 token/s
    Case("--model facebook/opt-6.7b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 100 0 100 0 100 0 --gpu-batch-size 4 --overlap False"),
    # seq_len = 512, gen_len = 32
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 2 --overlap False"),
    # seq_len = 1024, gen_len = 32
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 1 --overlap False --prompt-len 1024"),
]

suite_6b7_1x1_comp = [
    # seq_len = 256, gen_len = 32
    # 56.72 token/s
    Case("--model facebook/opt-6.7b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 100 0 100 0 100 0 --gpu-batch-size 128 --overlap False --compress-weight --compress-cache"),
    # seq_len = 512, gen_len = 32
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 72 --overlap False --compress-weight --compress-cache"),
    # seq_len = 1024, gen_len = 32
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 28 --overlap False --compress-weight --compress-cache --prompt-len 1024"),
]

suite_30b_1x1 = [
    # seq_len = 256, gen_len = 32
    # 16.01 token/s
    Case("--model facebook/opt-30b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 10 90 0 100 0 100 --gpu-batch-size 160 --num-gpu-batches 2 --cpu --debug fewer_batch", "", False),
    # seq_len = 512, gen_len = 32
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    # seq_len = 1024, gen_len = 32
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 4 96 0 100 0 100 --gpu-batch-size 20 --num-gpu-batches 4 --cpu --debug fewer_batch --prompt-len 1024"),
]

suite_30b_1x1_comp = [
    Case("--model facebook/opt-30b --path _DUMMY_ --prompt-len 512 --gen-len 32 --percent 0 100 0 100 0 100 --gpu-batch-size 32 --num-gpu-batches 1 --debug fewer_batch --compress-cache"), 
    
    # seq_len = 256, gen_len = 32
    # 16.86 token/s
    # Case("--model facebook/opt-30b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 0 100 0 100 0 100 --gpu-batch-size 128 --num-gpu-batches 8 --debug fewer_batch --compress-cache"),
    # # seq_len = 512, gen_len = 32
    # Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 64 --num-gpu-batches 8 --debug fewer_batch --compress-cache"),
    # # Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 16 --num-gpu-batches 20 --debug fewer_batch --compress-cache"),
    # # seq_len = 1024, gen_len = 32
    # Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 20 --num-gpu-batches 12 --debug fewer_batch --compress-cache --prompt-len 1024"),
]

suite_175b_1x1 = [
    # seq_len = 256
    # 1.36 token/s
    Case("--model facebook/opt-175b --path _DUMMY_ --prompt-len 256 --gen-len 32 --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 64 --num-gpu-batches 8 --cpu --debug fewer_batch"),
    # seq_len = 512
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch"),
    # seq_len = 1024
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 12 --num-gpu-batches 12 --cpu --debug fewer_batch --prompt-len 1024"),
]

suite_175b_1x1_comp = [
    # seq_len = 256
    # 2.26 token/s
    Case("--model facebook/opt-175b --path _DUMMY_ --prompt-len 256 --gen-len 32 --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --debug fewer_batch --compress-weight --compress-cache"),
    # seq_len = 512
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --debug fewer_batch --compress-weight --compress-cache"),
    # seq_len = 1024
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 12 --num-gpu-batches 4 --debug fewer_batch --compress-weight --compress-cache --prompt-len 1024"),
]

suite_ablation_ds = [
    # 30B
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 100 0 100 0 --gpu-batch-size 8 --debug fewer_batch"),
    # 175B
    Case("--model facebook/opt-175b --path _DUMMY_ --percent 0 0 100 0 100 0 --gpu-batch-size 2 --debug fewer_batch"),
]

suite_ablation = [
    # 30B

    # 175B
    # no policy search
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    # no overlapping
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch --overlap False"),
    # no cpu compute
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --debug fewer_batch"),
    # use deepspeed policy
    Case("--model facebook/opt-175b --path _DUMMY_ --percent 0 0 100 0 100 0 --gpu-batch-size 2 --debug fewer_batch"),
]

suite_ablation_policy = [
    # 30B
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 0 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 1 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 50 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --cpu --debug fewer_batch", use_page_maga=True),

    # 175B
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 0 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 1 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 20 80 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --cpu --debug fewer_batch", use_page_maga=True),
]

suite_175b_breakdown = [
    # seq_len = 512
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug breakdown"),
]

suite_175b_stage = [
    # 1x1 policy
    Case("--model facebook/opt-175b-stage --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", "", True),

    # full cpu policy
    Case("--model facebook/opt-175b-stage --path _DUMMY_ --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 32 --num-gpu-batches 6 --cpu --debug fewer_batch", "", True),
]

suites = {
    "1b3_test": suite_1b3_test,

    "6b7_1x1": suite_6b7_1x1,
    "6b7_1x1_comp": suite_6b7_1x1_comp,

    "30b_1x1": suite_30b_1x1,
    "30b_1x1_comp": suite_30b_1x1_comp,

    "175b_1x1": suite_175b_1x1,
    "175b_1x1_comp": suite_175b_1x1_comp,

    "ablation": suite_ablation,
    "ablation_policy": suite_ablation_policy,
    "175b_breakdown": suite_175b_breakdown,
    "175b_stage": suite_175b_stage,

    "all_1x1": (suite_6b7_1x1 + suite_6b7_1x1_comp +
                suite_30b_1x1 + suite_30b_1x1_comp +
                suite_175b_1x1 + suite_175b_1x1_comp),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", type=str, default="../experiments/results")
    args = parser.parse_args()
    torch.manual_seed(0)

    model = CostModel()
    model.double()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-3)

    dataset = []
    for suite in ["6b7_1x1"]:
        cases = suites[suite]
        for case in cases:
            # pkl_file = os.path.join("/home/cusgadmin/FlexGen/benchmark/flexgen/", get_filename(case) + ".pkl")
            # print(f"loading {pkl_file}")
            # if os.path.exists(pkl_file):
            #     with open(pkl_file, "rb") as f:
            #         stats = pkl.load(f)
            s = case.prompt_len
            n = case.gen_len
            opt_config = get_opt_config(case.model_name)
            l = opt_config.num_hidden_layers
            h1 = opt_config.hidden_size
            h2 = opt_config.ffn_embed_dim
            wi = 8 * h1 ** 2 + 4 * h1 * h2
            gbs = case.gbs
            bls = case.bls
            wg, wc, cg, cc, hg, hc = case.percent
            wn = 100 - wg - wc
            cn = 100 - cg - cc
            hn = 100 - hg - hc
            wg, wc, wn, cg, cc, cn, hg, hc, hn = (
                wg / 100, wc / 100, wn / 100, cg / 100, cc / 100,
                cn / 100, hg / 100, hc / 100, hn  / 100)
            x = torch.tensor([[wi, l, h1, h2, s, n,
                            gbs, bls, wg, wc, wn, cg, cc, cn, hg, hc, hn]])
            
            # NOTE: where to extract the stats from? 
            # y = torch.tensor([[stats.prefill_latency, stats.decode_latency]])
            y = torch.tensor([[4.945, 152.639]])
            dataset.append((x, y))

    xs = torch.cat([row[0] for row in dataset])
    ys = torch.cat([row[1] for row in dataset])

    indices = torch.randperm(xs.shape[0])
    xs = xs[indices]
    ys = ys[indices]
    split = int(0.9 * len(xs))

    xs_train, xs_test = xs[:split], xs[split:]
    ys_train, ys_test = ys[:split], ys[split:]

    def compute_loss(xs, ys):
        ys_pred = model(xs)
        return loss_fn(ys_pred / ys, torch.ones_like(ys))

    num_epoches = 30000

    def set_update_cpu_delta(flag):
        model.c0.requires_grad = flag
        model.c1.requires_grad = flag
        model.c2.requires_grad = flag
        model.c3.requires_grad = flag

    def freeze_all_params():
        for param in model.parameters():
            param.requires_grad = False

    set_update_cpu_delta(False)

    for i in range(num_epoches):
        reg_loss = compute_loss(xs_train, ys_train)
        penalty_loss = 0
        for name, p in model.named_parameters():
            penalty_loss += F.relu(-p)
        penalty_loss += F.relu(model.gtoc_bdw_hidden - model.gtoc_bdw_cache)
        penalty_loss += F.relu(model.mm_flops_p - model.bmm_flops_p)
        penalty_loss += F.relu(model.mm_flops_g - model.bmm_flops_g)
        penalty_loss += F.relu(model.bmm_flops_p - model.bmm_flops_g)
        penalty_loss += F.relu(model.mm_flops_p - model.mm_flops_g)

        loss = reg_loss + penalty_loss * 100
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == int(num_epoches * 0.8):
            freeze_all_params()
            set_update_cpu_delta(True)
            optimizer.param_groups[0]['lr'] = 1e-3

        if i % 200 == 0:
            eval_loss = compute_loss(xs_train, ys_train)
            print(f"epoch: {i}, train_loss: {loss.item():.6f}, "
                  f"eval_loss: {eval_loss.item():.6f}")

        #for name, p in model.named_parameters():
            #print(name, p.grad)

    for name, param in model.named_parameters():
        if "bdw" in name:
            print(f"{name}:\t {1 / param.item():.4f} GB/s")
        elif "flops" in name:
            print(f"{name}:\t {1 / param.item():.4f} T")
        elif "c" in name:
            print(f"{name}:\t {param.item():.4f}")

    print(f"len(dataset): {len(dataset)}")
