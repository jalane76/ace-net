import json
from types import SimpleNamespace


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))


def save_config(config, config_file_path):
    with open(config_file_path, "w") as f:
        json.dump(
            config,
            f,
            default=lambda o: o.__dict__,
            ensure_ascii=False,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
        )
