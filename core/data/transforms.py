import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_default_transforms(image_size: tuple[int, int]) -> A.Compose:
    """
    Возвращает набор трансформаций по умолчанию (только ресайз и нормализация).
    """
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_augmentation_levels():
    """Возвращает список доступных уровней аугментации."""
    return [ "Легкая", "Средняя", "Интенсивная"]


def get_augmentation_transforms(image_size: tuple[int, int], level: str) -> A.Compose:
    """
    Собирает конвейер аугментаций Albumentations на основе выбранного уровня.
    """
    # Общие трансформации, которые применяются всегда (кроме ресайза и ToTensor)
    base_transforms = [
        A.HorizontalFlip(p=0.5),
    ]

    if level == "Легкая":
        level_transforms = [
            A.RandomBrightnessContrast(p=0.2),
        ]
    elif level == "Средняя":
        level_transforms = [
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Rotate(limit=15, p=0.5),
        ]
    elif level == "Интенсивная":
        level_transforms = [
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.ColorJitter(p=0.3),
            A.GaussNoise(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.ChannelShuffle(p=0.1),
            A.CLAHE(p=0.2),
        ]
    else:
        level_transforms = []

    # Собираем полный пайплайн
    transforms = [
                     A.Resize(height=image_size[0], width=image_size[1]),
                 ] + base_transforms + level_transforms + [
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ToTensorV2(),
                 ]

    # Этот bbox_params здесь корректен, так как это пайплайн аугментации для обучения
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=None,
    )
