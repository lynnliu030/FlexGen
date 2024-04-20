"""
The OPT model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import glob
import os
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open
import torch

import numpy as np
from tqdm import tqdm


def download_weights(model_name, path):

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    hf_model_name = model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.safetensors")
    st_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for st_file in tqdm(st_files, desc="Convert format"):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                param = f.get_tensor(name)
                param_path = os.path.join(path, name)
                with open(param_path, "wb") as outf:
                    np.save(outf, param.cpu().to(torch.float16).detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="~/opt_weights")
    args = parser.parse_args()

    download_weights(args.model, args.path)
