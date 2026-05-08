import json
import os
import re
from typing import Any

import requests
from numpy.typing import NDArray
from safetensors import safe_open
from tqdm import tqdm

from .encoder import Encoder, get_encoder

Array = NDArray[Any]

MODEL_NAMES = {
    '124M': 'gpt2',
    '355M': 'gpt2-medium',
    '774M': 'gpt2-large',
    '1558M': 'gpt2-xl',
}

# Maps HuggingFace weight key suffixes to (outer_key, inner_key, leaf_key) in params dict.
# inner_key=None means the tensor lives directly at params[block][outer_key][leaf_key].
_BLOCK_KEY_MAP = {
    'ln_1.weight': ('ln_1', None, 'g'),
    'ln_1.bias': ('ln_1', None, 'b'),
    'ln_2.weight': ('ln_2', None, 'g'),
    'ln_2.bias': ('ln_2', None, 'b'),
    'attn.c_attn.weight': ('attn', 'c_attn', 'w'),
    'attn.c_attn.bias': ('attn', 'c_attn', 'b'),
    'attn.c_proj.weight': ('attn', 'c_proj', 'w'),
    'attn.c_proj.bias': ('attn', 'c_proj', 'b'),
    'mlp.c_fc.weight': ('mlp', 'c_fc', 'w'),
    'mlp.c_fc.bias': ('mlp', 'c_fc', 'b'),
    'mlp.c_proj.weight': ('mlp', 'c_proj', 'w'),
    'mlp.c_proj.bias': ('mlp', 'c_proj', 'b'),
}


def _download_file(url: str, dest_path: str) -> None:
    r = requests.get(url, stream=True)
    r.raise_for_status()
    file_size = int(r.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f:
        with tqdm(
            ncols=100,
            desc='Fetching ' + os.path.basename(dest_path),
            total=file_size,
            unit_scale=True,
            unit='b',
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1000):
                f.write(chunk)
                pbar.update(1000)


_HF_FILENAMES = {
    'model.safetensors': 'model.safetensors',
    'vocab.json': 'encoder.json',
    'merges.txt': 'vocab.bpe',
    'config.json': 'config.json',
}


def download_gpt2_files(model_size: str, model_dir: str) -> None:
    hf_model = MODEL_NAMES[model_size]
    for hf_name, local_name in _HF_FILENAMES.items():
        dest = os.path.join(model_dir, local_name)
        if not os.path.exists(dest):
            _download_file(f'https://huggingface.co/{hf_model}/resolve/main/{hf_name}', dest)


def load_gpt2_params_from_safetensors(model_dir: str, hparams: dict[str, int]) -> dict[str, Any]:
    params: dict[str, Any] = {'blocks': [{} for _ in range(hparams['n_layer'])]}

    with safe_open(os.path.join(model_dir, 'model.safetensors'), framework='numpy') as f:
        for key in f.keys():
            if key.startswith('lm_head'):
                continue
            tensor = f.get_tensor(key)
            name = key.removeprefix('transformer.')

            if name == 'wte.weight':
                params['wte'] = tensor
            elif name == 'wpe.weight':
                params['wpe'] = tensor
            elif name == 'ln_f.weight':
                params.setdefault('ln_f', {})['g'] = tensor
            elif name == 'ln_f.bias':
                params.setdefault('ln_f', {})['b'] = tensor
            else:
                m = re.match(r'h\.(\d+)\.(.*)', name)
                if not m:
                    continue
                n, rest = int(m[1]), m[2]
                if rest not in _BLOCK_KEY_MAP:
                    continue
                outer, inner, leaf = _BLOCK_KEY_MAP[rest]
                block = params['blocks'][n]
                if inner is None:
                    block.setdefault(outer, {})[leaf] = tensor
                else:
                    block.setdefault(outer, {}).setdefault(inner, {})[leaf] = tensor

    return params


def load_encoder_hparams_and_params(
    model_size: str, models_dir: str
) -> tuple[Encoder, dict[str, int], dict[str, Any]]:
    assert model_size in MODEL_NAMES

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)
    download_gpt2_files(model_size, model_dir)

    encoder = get_encoder(model_size, models_dir)

    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)
    hparams = {
        'n_vocab': config['vocab_size'],
        'n_ctx': config['n_ctx'],
        'n_embd': config['n_embd'],
        'n_head': config['n_head'],
        'n_layer': config['n_layer'],
    }

    params = load_gpt2_params_from_safetensors(model_dir, hparams)
    return encoder, hparams, params
