import datetime
import os
import threading
import time
import traceback
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from core.data.dataset import CustomDataset
from core.models.base_model import BaseModel
from core.data.transforms import get_augmentation_transforms, get_default_transforms


try:
    matplotlib.use("Agg")
    print("INFO (Trainer): Matplotlib backend set to 'Agg'.")
except ImportError:
    print(
        "WARNING (Trainer): Could not set Matplotlib backend to 'Agg'. "
        "GUI interactions might still occur."
    )


def collate_fn(batch):
    """
    Пользовательская функция для сборки батча (копия из dataset.py).
    """
    try:
        # Пытаемся извлечь данные
        images, targets = list(zip(*batch, strict=True))
        return list(images), list(targets)
    except Exception as e:
        # Если batch "битый" (например, __getitem__ вернул None из-за ошибки)
        # DataLoader может передать сюда None.
        print(f"WARNING (collate_fn): Ошибка сборки батча: {e}. Фильтрация...")
        # Фильтруем None'ы, которые могли возникнуть
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            # Если весь батч "битый", возвращаем None,
            # _train_one_epoch должен это обработать.
            return None, None
        images, targets = list(zip(*valid_batch, strict=True))
        return list(images), list(targets)




class Trainer:
    """
    Класс, инкапсулирующий логику обучения и валидации модели.
    """

    def __init__(
            self,
            model: BaseModel,
            config: dict[str, Any],
            queue: Any,  # queue.Queue
            vlog_callback: Callable[[str], None],
            interrupt_event: threading.Event,
    ):
        self.config = config
        self.queue = queue
        self.vlog = vlog_callback
        self.interrupt_event = interrupt_event
        self.history = {"train_loss": [], "val_loss": []}

        # --- Конфигурация Тренировки ---
        self.training_params = self.config.get("training", {})
        self.epochs = self.training_params.get("epochs", 100)
        self.learning_rate = self.training_params.get("learning_rate", 0.001)
        self.batch_size = self.training_params.get("batch_size", 8)
        self.num_workers = self.training_params.get("num_workers", 0)
        self.save_dir = self.training_params.get("save_dir", "runs")

        device_str = (
            self.training_params.get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        self.device = torch.device(device_str)

        self.model = model.to(self.device)

        # --- Оптимизатор ---
        optimizer_name = self.training_params.get("optimizer", "Adam")
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Неподдерживаемый оптимизатор: {optimizer_name}")

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=5, factor=0.1
        )

        # --- Загрузчики данных (DataLoaders) ---
        self.log(f"Настройка Загрузчиков Данных (NumWorkers: {self.num_workers})...")

        self.data_config = self.config.get("data", {})
        data_yaml_path = self.data_config.get("data_yaml_path")
        if not data_yaml_path:
            raise ValueError(
                "Критическая ошибка: 'data_yaml_path' не найден в конфигурации."
            )

        cfg_image_size = self.config.get("model", {}).get("image_size", 640)
        if isinstance(cfg_image_size, int):
            self.image_size = (cfg_image_size, cfg_image_size)
        else:
            self.image_size = tuple(cfg_image_size)

        # 1. Получаем уровень аугментации
        aug_level = self.training_params.get("augmentation_level", "Легкая")

        # 2. Получаем transforms
        if aug_level == "Отключена":
            train_transforms = None  # В Dataset будет использован default_transforms
            self.log("INFO: Аугментации отключены.")
        else:
            train_transforms = get_augmentation_transforms(self.image_size, aug_level)
            self.log(f"INFO: Аугментации уровня '{aug_level}' включены.")

        # Валидация *всегда* использует только default_transforms (ресайз и нормализация)
        val_transforms = None

        try:
            self.log("Загрузка обучающего датасета...")
            self.train_dataset = CustomDataset(
                data_yaml_path=data_yaml_path,
                config=self.config,
                split="train",
                transforms=train_transforms,  # Передаем готовые
            )
            if not self.train_dataset or len(self.train_dataset) == 0:
                raise ValueError("Обучающий датасет пуст или не удалось его загрузить.")

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn,  # Используем collate_fn из этого модуля
            )

            # --- Валидационный датасет ---
            val_yaml_path = data_yaml_path  # По умолчанию
            val_split = "val"

            # Проверяем, указан ли отдельный валидационный датасет
            val_dataset_path_override = self.data_config.get("val_dataset_path")
            if val_dataset_path_override and os.path.isdir(val_dataset_path_override):
                val_yaml_override_path = os.path.join(
                    val_dataset_path_override, "data.yaml"
                )
                if os.path.exists(val_yaml_override_path):
                    self.log(
                        f"INFO: Используется отдельный валидационный датасет: {val_dataset_path_override}"
                    )
                    val_yaml_path = val_yaml_override_path
                    # val_split = "val" # (Этап 3) Используем 'train' или 'val' из data.yaml
                else:
                    self.log(
                        f"WARNING: Указан 'val_dataset_path', но '{val_yaml_override_path}' не найден. "
                        "Используется 'val' сплит из основного data.yaml."
                    )
            else:
                self.log(
                    "INFO: Отдельный валидационный датасет не указан. "
                    "Используется 'val' сплит из основного data.yaml."
                )

            self.log("Загрузка валидационного датасета...")
            try:
                self.val_dataset = CustomDataset(
                    data_yaml_path=val_yaml_path,
                    config=self.config,
                    split=val_split,
                    transforms=val_transforms,  # Передаем None (будет использован default)
                )
                if not self.val_dataset or len(self.val_dataset) == 0:
                    # Эта проверка перехватится внешним FileNotFoundError,
                    # если _scan_files вернет 0, но мы для надежности оставим.
                    raise FileNotFoundError("Валидационный датасет пуст (0 файлов).")

                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=collate_fn,  # Используем collate_fn из этого модуля
                )
            except FileNotFoundError as e:
                self.log(
                    f"ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить валидационный датасет. Ошибка: {e}"
                )
                self.log("Валидация будет пропущена.")
                self.val_dataset = None
                self.val_loader = None

        except FileNotFoundError as e:
            self.log(f"ОШИБКА: Файл не найден при загрузке датасета: {e}")
            raise
        except Exception as e:
            self.log(f"ОШИБКА: Не удалось создать DataLoaders: {e}")
            self.log(traceback.format_exc())
            raise

        self.log("DataLoaders успешно созданы.")

    def log(self, message: str):
        """Отправляет лог в основной (тихий) канал очереди."""
        print(message)
        self.queue.put(("log", message))

    def train(self):
        """
        Главный цикл обучения модели.
        """
        self.log(f"Запуск обучения на {self.epochs} эпох на устройстве: {self.device}")
        self.queue.put(("status", "Запуск..."))
        time.sleep(1)  # Даем UI время обновиться

        try:
            for epoch in range(1, self.epochs + 1):
                if self.interrupt_event.is_set():
                    self.log("Обучение прервано пользователем")
                    self._on_training_end(interrupted=True)
                    return

                self.queue.put(("epoch", (epoch, self.epochs)))

                # --- Фаза Обучения ---
                avg_train_loss = self._train_one_epoch(epoch)
                self.history["train_loss"].append(avg_train_loss)

                # --- Фаза Валидации ---
                avg_val_loss = None
                if self.val_loader:
                    avg_val_loss = self._validate_one_epoch(epoch)
                    if avg_val_loss is not None:
                        self.history["val_loss"].append(avg_val_loss)

                # --- Обновление метрик в UI ---
                metrics = {"train_loss": avg_train_loss, "val_loss": avg_val_loss}
                self.queue.put(("metrics", metrics))

                # --- Обновление LR ---
                if avg_val_loss:
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step(avg_train_loss)

                # --- Сохранение лучшей модели ---
                current_best_val_loss = min(self.history["val_loss"] or [float("inf")])
                current_best_train_loss = min(self.history["train_loss"] or [float("inf")])

                if epoch == 1:
                    self._save_checkpoint(epoch, avg_val_loss or avg_train_loss, is_best=True)
                elif avg_val_loss is not None and avg_val_loss < current_best_val_loss:
                    self._save_checkpoint(epoch, avg_val_loss, is_best=True)
                elif avg_val_loss is None and avg_train_loss < current_best_train_loss:
                    self._save_checkpoint(epoch, avg_train_loss, is_best=True)

            self._on_training_end(interrupted=False)

        except Exception as e:
            self.log(f"Критическая ошибка во время цикла обучения: {e}")
            self.log(traceback.format_exc())
            self.queue.put(("finished", (False, str(e))))

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Выполняет одну эпоху обучения.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        batch_errors = 0
        batches_processed = 0

        if not self.train_loader:
            self.log(f"КРИТИЧЕСКАЯ ОШИБКА: Train Loader не инициализирован.")
            raise ValueError("Train Loader не инициализирован.")

        loader_len = len(self.train_loader)
        if loader_len == 0:
            self.log(f"КРИТИЧЕСКАЯ ОШИБКА: Train Loader пуст (0 батчей).")
            raise ValueError("Train Loader пуст (0 батчей).")

        error_threshold = max(1, int(loader_len * 0.1))  # 10% или минимум 1

        i = 0
        try:
            for i, batch_data in enumerate(self.train_loader):
                if self.interrupt_event.is_set():
                    self.log("Обучение прервано")
                    break

                try:
                    # 1. Попытка загрузки данных
                    if batch_data is None or batch_data[0] is None:
                        # Это может произойти, если collate_fn вернул None
                        raise ValueError("DataLoader вернул пустой батч (None).")

                    images, targets = batch_data

                    if not images or not targets:
                        raise ValueError("Батч не содержит изображений или целей.")

                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # 2. Попытка прямого/обратного прохода
                    if not targets:
                        self.vlog(f"Эпоха {epoch}, Шаг {i + 1}/{loader_len}: Пропуск 'пустого' батча.")
                        continue

                    # Прямой и обратный проход
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    if not torch.isfinite(losses):
                        raise ValueError("Loss is not finite (inf/nan).")

                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                    total_loss += losses.item()
                    batches_processed += 1

                except Exception as e:
                    batch_errors += 1
                    self.log(
                        f"ОШИБКА (Эпоха {epoch}, Батч {i + 1}): Не удалось обработать батч (ошибка Model/CUDA?). {e}"
                    )
                    self.vlog(traceback.format_exc())
                    if "cuda" in self.device.type:
                        torch.cuda.empty_cache()

                    if batch_errors > error_threshold:
                        self.log(
                            f"Критическая ошибка: Более {error_threshold} ошибок ({batch_errors}) при ОБРАБОТКЕ батчей. Обучение остановлено.")
                        raise ValueError(f"Слишком много 'битых' данных (>{error_threshold})")

                    continue  # Пропускаем "битый" батч

                # --- Логирование и Прогресс-бар ---
                if i % 10 == 0 or i == loader_len - 1:
                    elapsed = time.time() - start_time
                    avg_loss_so_far = total_loss / (batches_processed or 1)
                    percent_done = (i + 1) / loader_len * 100

                    eta_epoch = 0
                    if batches_processed > 0:
                        eta_epoch = (elapsed / (i + 1)) * (loader_len - (i + 1))

                    status_msg = (
                        f"Эпоха: {epoch}/{self.epochs} [Обучение] "
                        f"({i + 1}/{loader_len} | {percent_done:.0f}%) "
                        f"Loss: {avg_loss_so_far:.4f} "
                        f"ETA Эпохи: {datetime.timedelta(seconds=int(eta_epoch))}"
                    )
                    self.queue.put(("status", status_msg))
                    self.queue.put(("progress", percent_done))

                    # Подробный лог потерь
                    loss_details = " | ".join(
                        [f"{k}: {v.item():.4f}" for k, v in loss_dict.items()]
                    )
                    self.vlog(
                        f"Эпоха {epoch}, Шаг {i + 1}: Batch Loss: {losses.item():.4f} | {loss_details}"
                    )

        except Exception as e:
            batch_errors += 1
            self.log(
                f"ОШИБКА (Эпоха {epoch}, Батч {i}): Не удалось ЗАГРУЗИТЬ батч (ошибка Dataset/JSON?). {e}"
            )
            self.vlog(traceback.format_exc())
            if "cuda" in self.device.type:
                torch.cuda.empty_cache()

            if batch_errors > error_threshold:
                self.log(
                    f"Критическая ошибка: Более {error_threshold} ошибок ({batch_errors}) при ЗАГРУЗКЕ данных. Обучение остановлено.")
                raise ValueError(f"Слишком много 'битых' данных (>{error_threshold})")

            # цикл, так как итератор сломан. Мы должны завершить эпоху досрочно.
            self.log(f"Итератор DataLoader остановлен из-за ошибки. Завершение эпохи {epoch}...")

        if self.interrupt_event.is_set():
            return total_loss / (batches_processed or 1)

        if batches_processed == 0 and batch_errors > 0:
            self.log(
                f"КРИТИЧЕСКАЯ ОШИБКА: Эпоха {epoch} [Обучение] завершена, но не обработано НИ ОДНОГО батча (Ошибок: {batch_errors}).")
            raise ValueError(f"Эпоха {epoch} не обработала ни одного батча.")

        avg_loss = total_loss / (batches_processed or 1)
        self.log(
            f"Эпоха {epoch} [Обучение] завершена. Средний Loss: {avg_loss:.4f} (Обработано: {batches_processed}/{loader_len} батчей)"
        )
        return avg_loss


    def _validate_one_epoch(self, epoch: int) -> float | None:
        """
        Выполняет одну эпоху валидации.
        """
        if not self.val_loader:
            self.vlog(f"Эпоха {epoch}: Пропуск валидации (Val Loader отсутствует).")
            return None

        self.model.eval()
        total_loss = 0.0
        batch_errors = 0
        batches_processed = 0
        loader_len = len(self.val_loader)
        if loader_len == 0:
            self.vlog(f"Эпоха {epoch}: Пропуск валидации (Val Loader пуст).")
            return None

        error_threshold = max(1, int(loader_len * 0.1))  # 10% или минимум 1

        self.queue.put(("status", f"Эпоха: {epoch}/{self.epochs} [Валидация]..."))
        self.queue.put(("progress", 0))

        i = 0
        with torch.no_grad():
            try:
                for i, batch_data in enumerate(self.val_loader):
                    if self.interrupt_event.is_set():
                        self.log("Обучение прервано (во время _validate_one_epoch).")
                        break

                    try:
                        # 1. Попытка загрузки данных
                        if batch_data is None or batch_data[0] is None:
                            raise ValueError("DataLoader вернул пустой батч (None).")

                        images, targets = batch_data

                        if not images or not targets:
                            raise ValueError("Батч не содержит изображений или целей.")

                        images = [img.to(self.device) for img in images]
                        targets = [
                            {k: v.to(self.device) for k, v in t.items()} for t in targets
                        ]

                        if not targets:
                            self.vlog(
                                f"Эпоха {epoch}, Шаг {i + 1}/{loader_len} [Вал]: Пропуск 'пустого' батча."
                            )
                            continue

                        # 2. Попытка прямого прохода
                        # Модель в режиме .train() для получения loss, но без .backward()
                        self.model.train()
                        loss_dict = self.model(images, targets)
                        self.model.eval()

                        losses = sum(loss for loss in loss_dict.values())

                        if torch.isfinite(losses):
                            total_loss += losses.item()
                            batches_processed += 1
                        else:
                            raise ValueError("Loss is not finite (inf/nan).")

                    except Exception as e:
                        batch_errors += 1
                        self.log(
                            f"ОШИБКА [Валидация] (Эпоха {epoch}, Батч {i + 1}): Не удалось обработать батч. {e}"
                        )
                        self.vlog(traceback.format_exc())
                        if "cuda" in self.device.type:
                            torch.cuda.empty_cache()

                        if batch_errors > error_threshold:
                            self.log(
                                f"Критическая ошибка [Валидация]: Более {error_threshold} ошибок ({batch_errors}). Валидация остановлена.")
                            # Мы не останавливаем все обучение, только валидацию на этой эпохе
                            return None

                        continue  # Пропускаем "битый" батч

                    percent_done = (i + 1) / loader_len * 100
                    self.queue.put(("progress", percent_done))

            except Exception as e:
                batch_errors += 1
                self.log(
                    f"ОШИБКА [Валидация] (Эпоха {epoch}, Батч {i}): Не удалось ЗАГРУЗИТЬ батч. {e}"
                )
                self.vlog(traceback.format_exc())
                if "cuda" in self.device.type:
                    torch.cuda.empty_cache()

                if batch_errors > error_threshold:
                    self.log(
                        f"Критическая ошибка [Валидация]: Более {error_threshold} ошибок ({batch_errors}) при ЗАГРУЗКЕ. Валидация остановлена.")
                    return None

                self.log(f"Итератор DataLoader (Валидация) остановлен из-за ошибки. Завершение эпохи {epoch}...")

        if self.interrupt_event.is_set():
            return total_loss / (batches_processed or 1)

        if batches_processed == 0:
            if batch_errors > 0:
                self.log(
                    f"Эпоха {epoch} [Валидация] НЕВЫПОЛНЕНА. Не обработано ни одного батча (Ошибок: {batch_errors})."
                )
            return None

        avg_loss = total_loss / batches_processed
        self.log(
            f"Эпоха {epoch} [Валидация] завершена. Средний Loss: {avg_loss:.4f} (Обработано: {batches_processed}/{loader_len} батчей)"
        )
        self.queue.put(("progress", 100))
        return avg_loss


    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        Сохраняет чекпоинт модели.
        """
        if not self.save_dir:
            return

        try:
            weights_dir = os.path.join(self.save_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)

            filename = "last_model.pt"
            if is_best:
                filename = "best_model.pt"

            save_path = os.path.join(weights_dir, filename)

            # Конфиг/веса идут в один файл
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                    "config": self.config,  # В чекпоинте также включен конфиг
                },
                save_path,
            )

            if is_best:
                self.log(
                    f"СОХРАНЕНО: Новая лучшая модель (Epoch: {epoch}, Loss: {loss:.4f}) в {save_path}"
                )
            self.vlog(f"Checkpoint сохранен: {save_path}")

        except Exception as e:
            self.log(f"ОШИБКА: Не удалось сохранить checkpoint: {e}")

    def _on_training_end(self, interrupted: bool = False):
        """
        Вызывается при завершении или прерывании обучения.
        """
        self.log("--- Обучение завершено ---")
        self.queue.put(("status", "Завершение..."))
        self.queue.put(("progress", 100))

        if interrupted:
            self.log("Сохранение последней модели перед выходом...")
            last_loss = self.history["train_loss"][-1] if self.history["train_loss"] else 0.0
            self._save_checkpoint(last_loss, 0, is_best=False)
            self.queue.put(("finished", (False, "Обучение прервано пользователем.")))
            return

        # --- Сохранение финального графика потерь ---
        try:
            save_path = os.path.join(self.save_dir, "loss_plot.png")
            self.save_loss_plot(self.history, save_path)
            self.log(f"График потерь сохранен в {save_path}")

        except Exception as e:
            self.log(f"Не удалось сохранить график потерь: {e}")

        # --- Отправка сообщения об успехе в UI ---
        best_model_path = os.path.join(self.save_dir, "weights", "best_model.pt")
        self.queue.put(("finished", (True, (best_model_path, self.history))))

    def save_loss_plot(self, history: dict, save_path: str):
        """
        Сохраняет график потерь в файл.
        """
        self.vlog(f"Попытка сохранения графика потерь в {save_path}...")
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Training Loss")

        if history.get("val_loss"):
            plt.plot(history["val_loss"], label="Validation Loss", linestyle="--")

        plt.title("История потерь (Loss) за Эпохи")
        plt.xlabel("Эпоха")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        self.vlog("График потерь сохранен.")