import os
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


# ---------- BraTS 2020 -----------


COLOR_MAP = {
    'necrosis': [255, 0, 0],        # Некроз
    'edema': [0, 255, 0],           # Отек
    'enhancing_tumor': [0, 0, 255]  # Активная опухоль
}


def normalize(img):
    """Нормализация изображения"""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def combine_rgb_mask(mask):
    """Преобразование маски в цветовую схему"""
    rgb_mask = np.zeros((240, 240, 3), dtype=np.uint8)
    rgb_mask[mask[:, :, 0] > 0] = COLOR_MAP['necrosis']
    rgb_mask[mask[:, :, 1] > 0] = COLOR_MAP['edema']
    rgb_mask[mask[:, :, 2] > 0] = COLOR_MAP['enhancing_tumor']
    return rgb_mask


def load_volume_slices(volume_dir, volume_id):
    """Загрузка всех срезов одного объема"""
    volume_files = sorted([f for f in os.listdir(volume_dir) if f.startswith(f'volume_{volume_id}_')],
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))

    images = []
    masks = []

    for vf in volume_files:
        with h5py.File(os.path.join(volume_dir, vf), 'r') as hvf:
            images.append(hvf['image'][:])
            masks.append(hvf['mask'][:])

    image_volume = np.stack(images, axis=-1)  # (240, 240, 4, 155)
    mask_volume = np.stack(masks, axis=-1)  # (240, 240, 3, 155)

    return image_volume, mask_volume


def visualize_volume_slices(data_dir, volume_id, start_slice=0, end_slice=154, num_slices=3):
    """Визуализация срезов из собранного объема"""
    image_volume, mask_volume = load_volume_slices(data_dir, volume_id)

    assert image_volume.shape == (240, 240, 4, 155), "Некорректная форма объема изображений"
    assert mask_volume.shape == (240, 240, 3, 155), "Некорректная форма объема масок"

    slice_indices = np.linspace(start_slice, end_slice, num_slices, dtype=int)

    plt.figure(figsize=(10, 2 * num_slices))

    for i, slice_idx in enumerate(slice_indices):
        img_slice = image_volume[..., slice_idx]
        mask_slice = mask_volume[..., slice_idx]

        img_norm = normalize(img_slice)
        rgb_mask = combine_rgb_mask(mask_slice)

        plt.subplot(num_slices, 3, i * 3 + 1)
        plt.imshow(img_norm, cmap='gray')
        plt.title(f'Срез {slice_idx}')
        plt.axis('off')

        plt.subplot(num_slices, 3, i * 3 + 2)
        plt.imshow(rgb_mask)
        plt.title(f'Маска среза {slice_idx}')
        plt.axis('off')

        plt.subplot(num_slices, 3, i * 3 + 3)
        plt.imshow(img_norm, cmap='gray')
        plt.imshow(rgb_mask, alpha=0.4)
        plt.title(f'Наложение {slice_idx}')
        plt.axis('off')

    plt.suptitle(f"Визуализация срезов тома №{volume_id}")
    plt.tight_layout()
    plt.show()


# ---------- BCSD -----------


def load_image(image_path, normalize=True):
    """Загрузка изображения"""
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if normalize:
        img = img.astype(np.float32) / 255.0
    return img


def load_mask(mask_path, normalize=True):
    """Загрузка маски"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 255
    if normalize:
        mask = mask.astype(np.float32) / 255.0
    return mask


def get_random_sample(image_paths, mask_paths):
    """Получение случайного образца"""
    idx = np.random.randint(0, len(image_paths))
    return load_image(image_paths[idx]), load_mask(mask_paths[idx])


def visualize_bcsd_samples(image_paths, mask_paths, num_samples=4):
    """Визуализация нескольких образцов на одном холсте"""
    indices = np.random.choice(len(image_paths), num_samples, replace=False)

    plt.figure(figsize=(12, 2 * num_samples))

    rows = num_samples

    for i, idx in enumerate(indices):
        image = load_image(image_paths[idx])
        mask = load_mask(mask_paths[idx])

        plt.subplot(rows, 3, i * 3 + 1)
        plt.imshow(image)
        plt.title(f'Образец {idx}\nИсходник')
        plt.axis('off')

        plt.subplot(rows, 3, i * 3 + 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Маска')
        plt.axis('off')

        plt.subplot(rows, 3, i * 3 + 3)
        overlay_img = image.copy()
        overlay_img[mask > 0] = [1, 0, 0]
        plt.imshow(overlay_img)
        plt.title('Наложение')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# def visualize_bcsd_sample(image_path, mask_path):
#     """Визуализация образца"""
#     image = load_image(image_path)
#     mask = load_mask(mask_path)
#
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 3, 1)
#     plt.imshow(image)
#     plt.title('Исходное изображение')
#     plt.axis('off')
#
#     plt.subplot(1, 3, 2)
#     plt.imshow(mask, cmap='gray')
#     plt.title('Маска')
#     plt.axis('off')
#
#     plt.subplot(1, 3, 3)
#     overlay_img = image.copy()
#     overlay_img[mask > 0] = [255, 0, 0]
#     plt.imshow(overlay_img)
#     plt.title('Наложение маски')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()