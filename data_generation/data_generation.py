# blood_cell_generator.py
import cv2
import os
import numpy as np
import random
from glob import glob
from collections import defaultdict

# Конфигурация
PATCH_SIZE = (64, 64)
MIN_CELL_SIZE = 30
NUM_CELLS_RANGE = (2, 10)
IMAGE_SIZE = (512, 512)
OUTPUT_DIR = "generated_data"
CLASS_WEIGHTS = {'uniform': None, 'weighted': [0.6, 0.3, 0.1]}  # Пример весов для 3 классов


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


def create_dirs():
    """Создание необходимых директорий"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("temp/cells", exist_ok=True)
    os.makedirs("temp/background", exist_ok=True)


def balanced_cell_patches(image_paths, mask_paths):
    """Нарезка патчей клеток с балансировкой классов"""
    class_counts = defaultdict(int)

    for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= MIN_CELL_SIZE and h >= MIN_CELL_SIZE:
                cells.append((x, y, w, h))

        # Сохранение патчей
        for i, (x, y, w, h) in enumerate(cells):
            patch = img[y:y + h, x:x + w]
            class_id = i % 3  # Пример: 3 класса
            cv2.imwrite(f"temp/cells/class_{class_id}_cell_{idx}_{i}.png", patch)
            class_counts[class_id] += 1

    print("Нарезано патчей клеток:")
    for cls, count in class_counts.items():
        print(f"Класс {cls}: {count} патчей")


def crop_background_patches(image_paths):
    """Нарезка фоновых патчей"""
    count = 0
    for img_path in image_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        for i in range(0, h - PATCH_SIZE[1], PATCH_SIZE[1] // 2):
            for j in range(0, w - PATCH_SIZE[0], PATCH_SIZE[0] // 2):
                patch = img[i:i + PATCH_SIZE[1], j:j + PATCH_SIZE[0]]
                cv2.imwrite(f"temp/background/bg_{count}.png", patch)
                count += 1
    print(f"Нарезано {count} фоновых патчей")


def optimized_background_patches(image_paths, max_patches=1000):
    count = 0
    bg_patches = []

    for img_path in image_paths:
        if count >= max_patches:
            break

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Случайные координаты вместо полного перебора
        for _ in range(50):  # 50 попыток на изображение
            y = random.randint(0, h - PATCH_SIZE[1])
            x = random.randint(0, w - PATCH_SIZE[0])
            patch = img[y:y + PATCH_SIZE[1], x:x + PATCH_SIZE[0]]

            if patch.mean() > 30:  # Отсеиваем слишком тёмные области
                bg_patches.append(patch)
                count += 1

                if count >= max_patches:
                    break

    # Сохраняем только уникальные патчи
    unique_patches = list({p.tobytes(): p for p in bg_patches}.values())

    for i, patch in enumerate(unique_patches):
        cv2.imwrite(f"temp/background/bg_{i}.png", patch)

    print(f"Сохранено {len(unique_patches)} оптимизированных фоновых патчей")


def create_background():
    """Создание фонового изображения"""
    bg_files = glob("temp/background/*.png")
    background = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)

    for i in range(IMAGE_SIZE[1] // PATCH_SIZE[1]):
        for j in range(IMAGE_SIZE[0] // PATCH_SIZE[0]):
            patch = cv2.imread(random.choice(bg_files))
            background[i * PATCH_SIZE[1]:(i + 1) * PATCH_SIZE[1],
            j * PATCH_SIZE[0]:(j + 1) * PATCH_SIZE[0]] = patch
    return background


def place_cells(background, num_cells, distribution):
    """Размещение клеток на фоне"""
    cell_files = glob("temp/cells/*.png")
    if distribution == 'weighted':
        selected = random.choices(cell_files, weights=CLASS_WEIGHTS['weighted'], k=num_cells)
    else:
        selected = random.sample(cell_files, num_cells)

    for cell_path in selected:
        cell = cv2.imread(cell_path)
        h, w = cell.shape[:2]
        x = random.randint(0, IMAGE_SIZE[0] - w)
        y = random.randint(0, IMAGE_SIZE[1] - h)
        background[y:y + h, x:x + w] = cell
    return background


def generate_images(num_images=10, distribution='uniform'):
    """Генерация финальных изображений"""
    for i in range(num_images):
        bg = create_background()
        num_cells = random.randint(*NUM_CELLS_RANGE)
        final_image = place_cells(bg, num_cells, distribution)

        # Добавление шумов
        noisy_gaus = final_image + np.random.normal(0, 25, final_image.shape).astype(np.uint8)
        noisy_const = cv2.add(final_image, 30)

        # Сохранение
        cv2.imwrite(f"{OUTPUT_DIR}/clean_{i}.png", final_image)
        cv2.imwrite(f"{OUTPUT_DIR}/noisy_gaus_{i}.png", noisy_gaus)
        cv2.imwrite(f"{OUTPUT_DIR}/noisy_const_{i}.png", noisy_const)

    print(f"Сгенерировано {num_images} изображений в папке {OUTPUT_DIR}")


if __name__ == "__main__":
    # Инициализация
    create_dirs()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "BCCD Dataset with mask")
    # image_paths = glob("blood_cells_dataset/original/*.png")
    # mask_paths = glob("blood_cells_dataset/mask/*.png")
    image_paths, mask_paths = load_image_and_mask_paths_bcsd(data_dir)

    # Нарезка патчей
    balanced_cell_patches(image_paths, mask_paths)
    optimized_background_patches(image_paths)

    # Генерация изображений
    generate_images(num_images=20, distribution='uniform')  # 'uniform' или 'weighted'