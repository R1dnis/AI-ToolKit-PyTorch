from typing import Any

from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .base_model import BaseModel


class CustomPlantModel(BaseModel):
    """
    Кастомная модель для сегментации объектов на основе Mask R-CNN с ResNet50 FPN V2.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.backbone = self._create_model()

    def _create_model(self):
        """
        Создает и настраивает модель Mask R-CNN.
        """
        task_config = self.config["task"]["outputs"]["main_detection"]
        num_classes = len(task_config["classes"])

        image_size = self.config.get("model", {}).get("image_size", 640)
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(
            weights=weights, progress=True, min_size=image_size, max_size=int(image_size * 2)
        )

        model.transform.image_mean = [0.0, 0.0, 0.0]
        model.transform.image_std = [1.0, 1.0, 1.0]

        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        return model

    def forward(self, images, targets=None):
        """
        Прямой проход модели.
        В режиме обучения принимает изображения и цели, возвращает словарь потерь.
        В режиме предсказания принимает только изображения, возвращает предсказания.
        """
        if self.training:
            if targets is None:
                raise ValueError("В режиме обучения должны быть переданы цели (targets).")
            loss_dict = self.backbone(images, targets)
            return loss_dict
        else:
            detections = self.backbone(images)
            return detections

