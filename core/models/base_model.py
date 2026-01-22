from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Абстрактный базовый класс для всех моделей в проекте.
    Определяет общий интерфейс для обучения и инференса.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Выполняет прямой проход данных через модель.
        Должен быть реализован в каждом наследнике.
        """
        pass

    def get_name(self) -> str:
        """
        Возвращает имя модели на основе имени класса.
        """
        return self.__class__.__name__
