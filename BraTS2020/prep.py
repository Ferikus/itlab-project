import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt


data_path = r"D:\Datasets\BraTS2020_training_data\content\data"


"""Функции для работы с данными"""


def load_and_sort_files(directory):
    """Создаёт отсортированный список из файлов .h5"""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
    files.sort()
    return files


def normalize(img):
    """Нормализация данных"""
    img_norm = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1) + 1e-8)
    img_norm = np.clip(img_norm, 0, 1)
    return img_norm


def get_slice(file_path):
    """Получить изображение и маску из тома"""
    with h5py.File(file_path, 'r') as hf:
        image_data = hf['image'][:]
        mask_data = hf['mask'][:]
    norm_image_data = normalize(image_data)
    norm_image_data = (norm_image_data * 255).astype(np.uint8)
    return norm_image_data, mask_data


def get_all_slices(files):
    """Создает массивы всех изображений и масок датасета"""
    images = []
    masks = []
    for file in files:
        image_data, mask_data = get_slice(file)
        images.append(image_data)
        masks.append(mask_data)
    return images, masks


def combine_mask(mask):
    """Преобразование 3-канальной маски в одноканальную"""
    combined_mask = np.any(mask > 0, axis=-1).astype(np.uint8)
    return combined_mask


def combine_rgb_mask(mask):
    """RGB-маска"""
    colors = [
        [0, 0, 0],  # 0: Фон
        [255, 0, 0],  # 1: Некроз
        [0, 255, 0],  # 2: Отек
        [0, 0, 255]  # 4: Активная опухоль
    ]
    mask_rgb = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for channel in range(3):
        mask_rgb[mask[..., channel] > 0] = colors[channel + 1]
    return mask_rgb
