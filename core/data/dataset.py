import json
import os
import traceback
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import albumentations as A

from constants import IMAGES_DIR, LABELS_DIR

from .transforms import (
    get_augmentation_transforms,
    get_default_transforms,
)




class CustomDataset(Dataset):
    """
    Кастомный датасет для сегментации и детекции.
    Читает изображения и JSON-аннотации.
    """

    def __init__(
            self,
            data_yaml_path: str,
            config: dict[str, Any],
            split: str = "train",
            transforms: Any = None,
    ):
        self.config = config
        self.data_config = config.get("data", {})
        self.training_params = config.get("training", {})

        if not data_yaml_path or not os.path.exists(data_yaml_path):
            raise FileNotFoundError(
                f"Не найден data.yaml по пути: {data_yaml_path}. "
                "Укажите путь в config['data']['data_yaml_path']"
            )

        # Определение размера изображения из конфига
        cfg_image_size = self.config.get("model", {}).get("image_size", 640)
        if isinstance(cfg_image_size, int):
            self.image_size = (cfg_image_size, cfg_image_size)
        else:
            self.image_size = tuple(cfg_image_size)

        data_root = os.path.dirname(data_yaml_path)

        # 2. Загрузка data.yaml
        with open(data_yaml_path, encoding="utf-8") as f:
            data_yaml = yaml.safe_load(f)

        self.class_names = data_yaml.get("names", [])
        if not self.class_names:
            raise ValueError(f"Список 'names' (классов) не найден в {data_yaml_path}")

        # 3. Определение директорий (train/val)
        if split == "train":
            image_dir_name = data_yaml.get("train")
            label_dir_name = "labels"  # Предполагаем, что лейблы там же
        elif split == "val":
            image_dir_name = data_yaml.get("val")
            label_dir_name = "labels_val"  # ИЛИ data_yaml.get("val_labels")
        else:
            raise ValueError(f"Неизвестный режим: {split}. Допустимо 'train' или 'val'.")

        self.image_dir = os.path.join(data_root, image_dir_name)
        # Сначала ищем рядом с папкой images (например, datasets/v1/images и datasets/v1/labels)
        potential_label_dir = os.path.join(os.path.dirname(self.image_dir), label_dir_name)
        if os.path.exists(potential_label_dir):
            self.label_dir = potential_label_dir
        else:
            # Альтернативный путь (например, datasets/images и datasets/labels)
            flat_label_dir = os.path.join(data_root, label_dir_name)
            if os.path.exists(flat_label_dir):
                self.label_dir = flat_label_dir
            else:
                raise FileNotFoundError(
                    f"Директория лейблов не найдена. Проверены: {potential_label_dir} и {flat_label_dir}")

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Директория изображений не найдена: {self.image_dir}")

        # 4. Сбор фидбэка (если есть и режим 'train')
        feedback_items = []
        feedback_dir = self.data_config.get("feedback_dir")  # Используем data_config
        if split == "train" and feedback_dir and os.path.exists(feedback_dir):
            fb_image_dir = os.path.join(feedback_dir, IMAGES_DIR)
            fb_label_dir = os.path.join(feedback_dir, LABELS_DIR)
            if os.path.exists(fb_image_dir) and os.path.exists(fb_label_dir):
                print(f"INFO: Обнаружена папка обратной связи: {feedback_dir}")
                feedback_items = self._scan_files(fb_image_dir, fb_label_dir)
                print(f"INFO: Найдено {len(feedback_items)} файлов для дообучения из обратной связи.")

        # 5. Сбор основных файлов
        main_items = self._scan_files(self.image_dir, self.label_dir)
        self.dataset_items = main_items + feedback_items  # Добавляем фидбэк к основным данным

        if not self.dataset_items:
            raise FileNotFoundError(
                f"Не найдено совпадений изображений и .json файлов в: \nIMG: {self.image_dir}\nLBL: {self.label_dir}"
            )

        # 6. Трансформации
        self.default_transforms = get_default_transforms(self.image_size)
        self.transforms = transforms  # Принимаем None или Compose из Trainer

    def _scan_files(self, image_dir: str, label_dir: str) -> list[dict[str, str]]:
        """
        Сканирует директории и создает список пар (изображение, json).
        """
        items = []
        for img_name in os.listdir(image_dir):
            if not img_name.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                continue

            base_name, _ = os.path.splitext(img_name)
            json_name = f"{base_name}.json"
            json_path = os.path.join(label_dir, json_name)

            if os.path.exists(json_path):
                items.append(
                    {"image": os.path.join(image_dir, img_name), "json": json_path}
                )
            else:
                # Тихий пропуск (для датасетов, где есть картинки без разметки)
                pass
                # print(f"Warning: Пропуск {img_name}, не найден {json_name}")
        return items

    def __len__(self) -> int:
        return len(self.dataset_items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        item = self.dataset_items[idx]
        image_path = item["image"]
        json_path = item["json"]

        try:
            # 1. Загрузка изображения
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.exif_transpose(image)
            image_np = np.array(image)
            h, w = image_np.shape[:2]
            if h == 0 or w == 0:
                raise ValueError(f"Изображение {image_path} имеет нулевые размеры.")

            # 2. Загрузка JSON
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            # 3. Парсинг аннотаций (маски и bboxes)
            masks = []
            bboxes = []
            labels = []

            for ann in data.get("annotations", []):
                class_name = ann.get("class_name")
                if class_name not in self.class_names:
                    # Игнорируем классы, которых нет в data.yaml
                    continue

                label_idx = self.class_names.index(class_name)
                coords = ann.get("coords")
                ann_type = ann.get("type", "poly")  # По умолчанию полигон

                if not coords:
                    continue

                if ann_type == "poly":
                    # Создаем бинарную маску из полигона
                    mask = np.zeros((h, w), dtype=np.uint8)

                    polygon_points = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                    if polygon_points.ndim != 3 or polygon_points.shape[1] != 1 or polygon_points.shape[2] != 2:
                        raise ValueError(f"Некорректная форма полигона: {coords}")

                    cv2.fillPoly(mask, [polygon_points], 1)

                    # Создаем bbox из маски
                    pos = np.where(mask)
                    if pos[0].size == 0 or pos[1].size == 0:
                        continue  # Пустая маска
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])

                elif ann_type == "rect":
                    x, y, rect_w, rect_h = map(float, coords)
                    xmin, ymin, xmax, ymax = int(x), int(y), int(x + rect_w), int(y + rect_h)

                    # Создаем маску из bbox
                    mask = np.zeros((h, w), dtype=np.uint8)
                    # Обрезаем по границам изображения
                    ymin, ymax = max(0, ymin), min(h, ymax)
                    xmin, xmax = max(0, xmin), min(w, xmax)
                    mask[ymin:ymax, xmin:xmax] = 1

                else:
                    print(f"Warning: Неизвестный тип аннотации '{ann_type}' в {json_path}. Пропуск.")
                    continue

                # Проверка валидности bbox
                if xmin >= xmax or ymin >= ymax:
                    # Игнорируем аннотации с нулевой или отрицательной площадью
                    continue

                masks.append(mask)
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(label_idx)

            if not masks:
                # Если аннотаций нет (или все отфильтрованы), возвращаем "пустой" таргет (фон)
                return self._get_empty_target(image_np)

            # 4. Применение трансформаций (Albumentations)
            transformed = self._apply_transforms(image_np, masks, bboxes, labels)

            image_tensor = transformed["image"]
            valid_masks = transformed["masks"]
            valid_bboxes = transformed["bboxes"]
            valid_labels = transformed["labels"]

            if not valid_bboxes:
                # Если все аннотации отфильтровались (например, вышли за кадр), возвращаем фон
                return self._get_empty_target(image_np, use_transformed=True, original_tensor=image_tensor)

            # 5. Сборка таргета (target)
            target = {}
            target["boxes"] = torch.as_tensor(valid_bboxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(valid_labels, dtype=torch.int64)
            target["masks"] = torch.as_tensor(np.array(valid_masks), dtype=torch.uint8)
            target["image_id"] = torch.tensor([idx])

            # Расчет area
            areas = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                    target["boxes"][:, 2] - target["boxes"][:, 0]
            )
            target["area"] = areas
            target["iscrowd"] = torch.zeros((len(valid_bboxes),), dtype=torch.int64)

            return image_tensor, target

        # Этот блок ДОЛЖЕН выбрасывать исключение, чтобы Trainer.py мог его поймать
        except (IOError, json.JSONDecodeError, ValueError, TypeError, cv2.error) as e:
            print(
                f"\nCRITICAL (Dataset): Не удалось загрузить элемент {idx}\n"
                f"IMG: {image_path}\n"
                f"JSON: {json_path}\n"
                f"Ошибка: {e}"
            )
            print(traceback.format_exc())
            # Повторно выбрасываем исключение, чтобы Trainer мог его поймать
            raise e

    def _get_empty_target(self, image_np: np.ndarray, use_transformed: bool = False,
                          original_tensor: torch.Tensor = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Возвращает изображение и пустой словарь target (для изображений без объектов).
        """
        if use_transformed and original_tensor is not None:
            image_tensor = original_tensor
        else:
            # Применяем только дефолтные (без аугментаций)
            transformed = self.default_transforms(image=image_np, masks=[], bboxes=[], labels=[])
            image_tensor = transformed["image"]

        target = {}
        target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
        target["labels"] = torch.empty((0,), dtype=torch.int64)
        target["masks"] = torch.empty((0, self.image_size[0], self.image_size[1]), dtype=torch.uint8)
        target["image_id"] = torch.tensor([-1])  # -1 для обозначения "фона"
        target["area"] = torch.empty((0,), dtype=torch.float32)
        target["iscrowd"] = torch.empty((0,), dtype=torch.int64)

        return image_tensor, target

    def _apply_transforms(self, image_np: np.ndarray, masks: list[np.ndarray], bboxes: list[list[float]],
                          labels: list[int]) -> dict[str, Any]:
        """Применяет трансформации (аугментации или дефолтные)."""

        # Определяем, какой пайплайн использовать
        transform_pipeline = self.transforms if self.transforms else self.default_transforms

        # BboxParams необходимы *всегда*, если есть bboxes,
        # иначе Albumentations не знает, как их обрабатывать.
        # default_transforms их не имеет, поэтому мы "оборачиваем" его.

        # Создаем BboxParams (формат pascal_voc [xmin, ymin, xmax, ymax] корректен)
        bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1)

        # Если у нас кастомные аугментации, они уже должны иметь BboxParams
        if self.transforms:
            # Albumentations может падать (ValueError), если bbox некорректен
            try:
                transformed = transform_pipeline(image=image_np, masks=masks, bboxes=bboxes, labels=labels)
            except (ValueError, Exception) as aug_e:
                print(f"Warning: Ошибка аугментации ({aug_e}), возврат к дефолтной трансформации.")
                # Применяем дефолтный (только Resize/Normalize)
                default_pipeline = get_default_transforms(self.image_size)
                transformed = A.Compose(
                    default_pipeline.transforms,
                    bbox_params=bbox_params
                )(image=image_np, masks=masks, bboxes=bboxes, labels=labels)
        else:
            # Если у нас только default_transforms, у него нет bbox_params,
            # поэтому мы создаем Compose "на лету" с ними.
            default_with_bbox = A.Compose(
                transform_pipeline.transforms,
                bbox_params=bbox_params
            )
            transformed = default_with_bbox(image=image_np, masks=masks, bboxes=bboxes, labels=labels)

        # Фильтрация объектов, которые могли исчезнуть или стать невалидными
        valid_bboxes = []
        valid_masks = []
        valid_labels = []

        if "bboxes" in transformed and "masks" in transformed:
            for i, bbox in enumerate(transformed["bboxes"]):
                x_min, y_min, x_max, y_max = bbox

                # Добавлена проверка, что маска не пустая
                if (x_max > x_min and y_max > y_min and
                        i < len(transformed["masks"]) and transformed["masks"][i].sum() > 0):
                    valid_bboxes.append(bbox)
                    valid_masks.append(transformed["masks"][i])
                    valid_labels.append(transformed["labels"][i])

        transformed["bboxes"] = valid_bboxes
        transformed["masks"] = valid_masks
        transformed["labels"] = valid_labels

        return transformed

    @staticmethod
    def collate_fn(batch):
        """
        Пользовательская функция для сборки батча.
        """
        images, targets = list(zip(*batch, strict=True))
        # фильтруем батчи, где targets = None (может случиться при ошибках)
        # (Хотя теперь мы выбрасываем исключение, эта защита полезна)
        valid_batch = [(img, tgt) for img, tgt in zip(images, targets) if tgt is not None]
        if not valid_batch:
            return None, None  # Trainer должен будет это обработать

        images, targets = list(zip(*valid_batch, strict=True))
        return list(images), list(targets)
