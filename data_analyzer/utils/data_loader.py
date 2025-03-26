import os
import h5py


# ---------- BraTS 2020 -----------


def load_and_sort_files_brats(directory):
    """Загрузка и сортировка всех .h5 файлов"""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    return files


def get_slice_data(file_path, slice_idx=0):
    """Получение конкретного среза"""
    image, mask = load_volume(file_path)
    return image[..., slice_idx], mask[..., slice_idx]


# ---------- BCSD -----------


def load_and_sort_files_bcsd(files, directory):
    """Загрузка файлов в массив"""
    files += [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png') or f.endswith('jpg')]
    files.sort()
    return files


def load_image_and_mask_paths_bcsd(data_dir):
    """Получение путей к изображениям и маскам"""

    image_paths = []
    image_paths = load_and_sort_files_bcsd(image_paths, os.path.join(data_dir, "test", "original"))
    image_paths = load_and_sort_files_bcsd(image_paths, os.path.join(data_dir, "train", "original"))

    mask_paths = []
    mask_paths = load_and_sort_files_bcsd(mask_paths, os.path.join(data_dir, "test", "mask"))
    mask_paths = load_and_sort_files_bcsd(mask_paths, os.path.join(data_dir, "train", "mask"))

    return image_paths, mask_paths


def validate_dataset(image_paths, mask_paths):
    """Проверка соответствия изображение-маска"""
    assert len(image_paths) == len(mask_paths), "Несоответствие количества изображений и масок"

    for img_path, mask_path in zip(image_paths, mask_paths):
        img_name = os.path.basename(img_path.split('.')[0])
        mask_name = os.path.basename(mask_path.split('.')[0])
        assert img_name == mask_name, f"Несоответствие имен файлов: {img_name} vs {mask_name}"