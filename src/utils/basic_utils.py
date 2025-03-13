import importlib

import numpy as np
import torch
from calflops import calculate_flops
from PIL import Image


def save_tensor_as_image(image_tr: torch.Tensor, path: str):
    image_tr = image_tr[0].permute(1, 2, 0) # [1, c, h, w] -> [h, w, c]
    image_tr = image_tr * 127.5 + 127.5
    image_tr = torch.clamp(image_tr, 0, 255)

    image_np = image_tr.detach().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image_np)
    image.save(path)


def calculate_model_flops(model, input_shape=(1, 3, 224, 224)):
    """
    计算 FLOPs、MACs 和参数量
    Args:
        model:
        input_shape:
    Returns:
    """

    flops, macs, params = calculate_flops(model=model, input_shape=input_shape, output_as_string=True)
    print(f"{model.__class__.__name__} has {params} params.")
    print(f"MACs: {macs}")
    print(f"FLOPs: {flops}")
    return flops, macs, params


def count_params(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    """
      target: models.ESDNet
      params:
        en_channels: 48
        en_growth_rate: 32
        de_channels: 64
        de_growth_rate: 32
        sam_layers: 2
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)