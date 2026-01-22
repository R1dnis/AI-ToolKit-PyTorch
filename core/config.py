from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Загружает конфигурационный файл YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str):
    """Сохраняет словарь конфигурации в файл YAML."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)


def get_default_config() -> dict[str, Any]:
    """
    Возвращает словарь с базовой конфигурацией для новой модели.
    """
    return {
        "display_name": "Новая модель",
        "model": {"name": "CustomPlantModel", "image_size": 640},
        "data": {
            "dataset_path": "",
            "format": "json",
            "val_dataset_path": None,
            "data_yaml_path": None,
        },
        "task": {
            "type": "multitask",
            "outputs": {"main_detection": {"type": "detection", "classes": ["background", "plant"]}},
        },
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "save_dir": "runs",
            "num_workers": 0,
            "augmentation_level": "Легкая",
        },
    }
