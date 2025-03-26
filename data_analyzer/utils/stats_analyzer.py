import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm


# ---------- BraTS 2020 -----------


def calculate_class_distribution(mask_volume):
    """Вычисление распределения классов в маске"""
    class_counts = {
        'necrosis': np.sum(mask_volume[..., 0, :] > 0),
        'edema': np.sum(mask_volume[..., 1, :] > 0),
        'enhancing_tumor': np.sum(mask_volume[..., 2, :] > 0)
    }
    total = sum(class_counts.values())
    return {k: v/total for k, v in class_counts.items()}


def calculate_tumor_geometry(mask_slice):
    """Анализ геометрических характеристик опухоли для одного среза"""
    contours = [cv2.findContours(
        mask_slice[:, :, 1].astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )[0]]
    return {
        'num_regions': sum(len(c) for c in contours),
        'total_area': sum(cv2.contourArea(c) for cnt in contours for c in cnt)
    }


# ---------- BCSD -----------


def calculate_stats_bcsd(image_paths, mask_paths):
    """Вычисление статистик датасета"""
    stats = defaultdict(list)

    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        # Статистика изображений
        img = cv2.imread(img_path)
        stats['image_height'].append(img.shape[0])
        stats['image_width'].append(img.shape[1])
        stats['image_channels'].append(img.shape[2])

        # Статистика масок
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        stats['mask_classes'].append(np.unique(mask))
        stats['mask_pixel_counts'].append(np.sum(mask > 0))

        # Цветовые характеристики
        stats['mean_intensity'].append(np.mean(img, axis=(0, 1)))
        stats['std_intensity'].append(np.std(img, axis=(0, 1)))

    return stats


def show_stats(stats):
    """Визуализация статистик"""
    print(f"Всего изображений: {len(stats['image_height'])}")
    print(f"Средний размер изображений: {np.mean(stats['image_height']):.1f}x{np.mean(stats['image_width']):.1f}")
    print(f"Средняя интенсивность по каналам (R, G, B): {np.mean(stats['mean_intensity'], axis=0)}")
    print(f"Среднее количество пикселей объектов на маску: {np.mean(stats['mask_pixel_counts'])}")