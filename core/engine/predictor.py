import uuid
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch
import torchvision  # Добавлен для NMS
from PIL import Image, ImageOps

from core.data.transforms import get_default_transforms
from core.models.base_model import BaseModel


class Predictor:
    """
    Класс для выполнения инференса (анализа) с помощью обученной модели.
    """

    def __init__(self, model: BaseModel, config: dict[str, Any]):
        self.model = model
        self.config = config

        device_str = (
            self.config.get("training", {}).get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        self.device = torch.device(device_str)

        self.model.to(self.device)

        # 1. Переводим главную модель в режим оценки (.eval()).
        #    Это гарантирует, что CustomPlantModel.forward пойдет по ветке "else"
        #    и не будет требовать 'targets', устраняя ValueError.
        self.model.eval()

        # 2. "Хирургическое" включение Batch Normalization.
        #    Мы принудительно переводим только слои BN в режим .train().
        #    Это заставляет их использовать статистику текущего изображения,
        #    а не "отравленную" накопленную статистику.
        #    Это решает проблему отсутствия предсказаний.
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()

        cfg_image_size = self.config.get('data', {}).get('image_size', (640, 640))
        if isinstance(cfg_image_size, int):
            self.model_input_size = (cfg_image_size, cfg_image_size)
        else:
            self.model_input_size = tuple(cfg_image_size)

        self.transforms = get_default_transforms(self.model_input_size)

    def predict(self, image_path: str, threshold_min: float = 0.3, threshold_max: float = 0.6) -> tuple[
        dict[str, Any], list[dict]]:
        """
        Выполняет предсказание для одного изображения.
        """
        image_pil = Image.open(image_path).convert("RGB")
        image_pil = ImageOps.exif_transpose(image_pil)
        orig_w, orig_h = image_pil.size

        image_np = np.array(image_pil)

        transformed = self.transforms(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        model_h, model_w = image_tensor.shape[2:]

        with torch.no_grad():
            predictions = self.model([image_tensor.squeeze(0)])

        processed_json, detections_for_drawing = self._postprocess(
            predictions,
            orig_img_size=(orig_w, orig_h),
            model_img_size=(model_w, model_h),
            threshold_min=threshold_min,
            threshold_max=threshold_max
        )
        return processed_json, detections_for_drawing

    def _postprocess(self, predictions: Any,
                     orig_img_size: tuple[int, int],
                     model_img_size: tuple[int, int],
                     threshold_min: float,
                     threshold_max: float) -> tuple[dict[str, Any], list[dict]]:
        """
        Преобразует "сырые" выходы, применяет фильтрацию по порогу, NMS и сглаживание.
        """
        class_counts = defaultdict(int)
        result_json = {"class_counts": class_counts}
        detections_for_drawing = []

        orig_w, orig_h = orig_img_size
        model_w, model_h = model_img_size

        if not predictions or not isinstance(predictions, list) or len(predictions) == 0:
            return result_json, []

        preds = predictions[0]

        # --- 1. Фильтрация по уверенности ---
        scores_tensor = preds['scores']
        keep_indices = (scores_tensor >= threshold_min) & (scores_tensor <= threshold_max)

        labels = preds['labels'][keep_indices]
        scores = preds['scores'][keep_indices]
        masks = preds['masks'][keep_indices]
        boxes = preds['boxes'][keep_indices]

        if len(boxes) == 0:
            return result_json, []

        # --- 2. NMS (Удаление дублей) ---
        nms_keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.3)

        labels = labels[nms_keep_indices]
        scores = scores[nms_keep_indices]
        masks = masks[nms_keep_indices]

        # --- 3. Подготовка результатов ---
        task_config = self.config.get('task', {})
        detection_task_details = task_config.get('outputs', {}).get('main_detection', {})
        class_names = detection_task_details.get('classes', [])

        if model_w <= 0 or model_h <= 0:
            scale_x, scale_y = 1.0, 1.0
        else:
            scale_x = orig_w / model_w
            scale_y = orig_h / model_h

        print("\n--- РЕЗУЛЬТАТЫ ПОСЛЕ ФИЛЬТРАЦИИ ---")
        for label_idx, score, mask_tensor in zip(labels, scores, masks, strict=True):
            label_idx_int = label_idx.item()

            if label_idx_int >= 0 and label_idx_int < len(class_names):
                class_name = class_names[label_idx_int]
            else:
                class_name = f"Unknown_ID_{label_idx_int}"

            print(f"Модель ID: {label_idx_int} ({score:.2f}) -> Mapped: {class_name}")

            if class_name != "background":
                class_counts[class_name] += 1

                mask_np = (mask_tensor.squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                poly_scaled = []
                if contours:
                    c = max(contours, key=cv2.contourArea)

                    # --- СГЛАЖИВАНИЕ (Fix "Лесенки") ---
                    # Применяем аппроксимацию полигона, чтобы убрать пиксельные артефакты (лесенки)
                    # epsilon регулирует степень сглаживания (чем больше, тем проще фигура)
                    epsilon = 0.004 * cv2.arcLength(c, True)
                    c_smooth = cv2.approxPolyDP(c, epsilon, True)

                    if c_smooth.shape[0] > 2:
                        c_squeezed = c_smooth.squeeze(1)
                        c_scaled = c_squeezed.astype(np.float32)
                        c_scaled[:, 0] *= scale_x
                        c_scaled[:, 1] *= scale_y
                        poly_scaled = c_scaled.astype(np.int32).tolist()

                detection_for_drawing = {
                    'id': str(uuid.uuid4()),
                    'class_name': f"{class_name} ({score:.2f})",
                    'type': 'poly',
                    'coords': poly_scaled,
                }
                detections_for_drawing.append(detection_for_drawing)
        print("-----------------------------------\n")

        result_json["class_counts"] = dict(class_counts)
        return result_json, detections_for_drawing