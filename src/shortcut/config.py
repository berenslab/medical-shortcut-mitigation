from typing import Any, Optional
import argparse
import json
from omegaconf import OmegaConf

import yaml

def flatten(xs: list) -> list:
    """Flattens a nested list of arbitrary depth."""
    ys = []
    for x in xs:
        if type(x) is list:
            ys = ys + flatten(x)
        else:
            ys.append(x)
    return ys


def nest_dict(x: dict) -> dict:
    y = {}

    def deep_insert(y: dict, key: Any, val: Any) -> None:
        split = key.split("/")
        if len(split) == 1:
            y[key] = val
        else:
            prefix = split[0]
            suffix = "/".join(split[1:])
            if prefix not in y:
                y[prefix] = {}
            deep_insert(y[prefix], suffix, val)

    for key, val in x.items():
        deep_insert(y, key, val)
    return y


def extract_argument_keys(cfg: dict, prefix: Optional[str] = None) -> list:
    """Extract argument keys and their default values from a given config dict."""
    if prefix is None:
        prefix = ""
    if prefix != "":
        prefix = f"{prefix}/"

    def f(key, val):
        if isinstance(val, dict):
            return extract_argument_keys(val, prefix=f"{prefix}{key}")
        else:
            return f"{prefix}{key}", val

    return [f(key, val) for key, val in cfg.items()]


def merge_dicts(*dicts: list[dict]) -> dict:
    """Merge dictionaries with ascending priority (later ones override earlier ones)."""
    y = {}

    def deep_update(a, b, path=None):
        """Dict a is updated with the values of dict b."""
        path = path or []
        for key, val in b.items():
            current_path = path + [str(key)]

            # If key not in a, just add it
            if key not in a:
                a[key] = val
                continue

            # If both are dicts → recurse
            if isinstance(a[key], dict) and isinstance(val, dict):
                deep_update(a[key], val, current_path)

            # If both are lists → overwrite (or concat if you prefer)
            elif isinstance(a[key], list) and isinstance(val, list):
                a[key] = val  # replace instead of merge to stay predictable

            # If types differ → overwrite but log warning
            elif type(a[key]) != type(val):
                print(
                    f"[merge_dicts] Type conflict at {'.'.join(current_path)}: "
                    f"{type(a[key]).__name__} → {type(val).__name__}. Overwriting."
                )
                a[key] = val

            # Otherwise (same type but not dict/list) → overwrite
            else:
                a[key] = val

    for x in dicts:
        deep_update(y, x)
    return y


def get_config(path_to_default_cfg: str) -> dict:
    """
    cfg_default < cfg_special < cfg_args
    """
    # load defaul config file (default.yaml)
    cfg_default = load_yaml_config(path_to_default_cfg)

    # extract argument keys and their default values from the default config
    arguments = dict(flatten(extract_argument_keys(cfg_default)))

    # build parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="Path where the config yaml file is located.")
    for key, default_val in arguments.items():
        if isinstance(default_val, list) or default_val is None:
             parser.add_argument(f"--{key}", type=parse_list)
        elif isinstance(default_val, dict) or default_val is None:
            parser.add_argument(f"--{key}", type=parse_dict)
        else:
            parser.add_argument(
                f"--{key}",
                help=f"Default: {default_val}",
                type=type(default_val),
            )

    # parse arguments
    args = parser.parse_args()
    cfg_args = nest_dict(
        {key: val for key, val in vars(args).items() if val is not None}
    )

    # load special config file
    path_to_special_cfg = args.cfg
    if path_to_special_cfg is not None:
        cfg_special = load_yaml_config(path_to_special_cfg)
    else:
        cfg_special = {}
        
    # return the final configuration
    cfg = merge_dicts(cfg_default, cfg_special, cfg_args)
    return OmegaConf.create(cfg)

def parse_list(s: str):
    """Parse JSON-style lists from CLI."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(
            f"Invalid list syntax: {s}\n"
            f"Use JSON, e.g. --metrics '[\"Accuracy\", \"AUROC\"]'"
        )

def parse_dict(s: str):
    """Parse JSON-style dicts from CLI."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(
            f"Invalid dict syntax: {s}\n"
            f"Use JSON, e.g. --data/confounder_strengths '{{\"drusen\": 0.7, \"normal\": 0.3}}'"
        )

def load_yaml_config(config_filename: str) -> dict:
    """Load yaml config.

    Args:
        config_filename: Filename to config.

    Returns:
        Loaded config (auto-unwraps config["config"] if present).
    """
    with open(config_filename) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Special case: config nested under top-level "config"
    if isinstance(cfg, dict) and "cfg" in cfg and isinstance(cfg["cfg"], dict):
        cfg = cfg["cfg"]

    return cfg