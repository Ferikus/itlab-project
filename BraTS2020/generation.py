import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from prep import *

IMG_SIZE = (240, 240)
PATCH_SIZE = (int(0.4 * IMG_SIZE), int(0.4 * IMG_SIZE))


def crop_brain_patches_with_masks(data_path, output_dir, patch_size=PATCH_SIZE):
    count = 0
    for volume_id in range(10):
        file_path = os.path.join(data_path, f"volume_{volume_id + 1}_slice_60.h5")
        img, mask = get_slice(file_path)

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        mask = combine_mask(mask)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            patch = img[y:y + h, x:x + w]
            if patch.size == 0:
                continue
            if patch.shape[0] < patch_size[1] or patch.shape[1] < patch_size[0]:
                patch = cv2.resize(patch, patch_size, interpolation=cv2.INTER_AREA)
            output_path = os.path.join(output_dir, f"brain_patch_{count}.png")
            cv2.imwrite(output_path, patch)
            count += 1
        print(f"Сохранено {count} патчей")


data_path = r"BraTS2020_training_data\content\data"
output_brain_patches_dir = r"BraTS2020_training_data\content\brain_patches"

print("Нарезка изображений опухоли мозга")
crop_brain_patches_with_masks(data_path, output_brain_patches_dir)


# NUM_CELLS_RANGE = (2, 10)
#
# def create_background_from_patches(image_size=IMAGE_SIZE, patch_dir="data/background"):
#     background = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
#     patch_size = (64, 64)
#     patch_files = glob(os.path.join(patch_dir, "*.png"))
#     num_patches_x = image_size[0] // patch_size[0]
#     num_patches_y = image_size[1] // patch_size[1]
#
#     for i in range(num_patches_y):
#         for j in range(num_patches_x):
#             patch = cv2.imread(random.choice(patch_files), cv2.IMREAD_GRAYSCALE)
#             if patch is None:
#                 print(f"Error loading patch from {patch_files[0]}")
#                 continue
#             patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
#             background[i * patch_size[1]:(i + 1) * patch_size[1], j * patch_size[0]:(j + 1) * patch_size[0]] = patch
#
#     return background
#
#
# def place_cell(background, cell):
#     bg_h, bg_w, _ = background.shape
#     cell_h, cell_w, _ = cell.shape
#     mask = 255 * np.ones((cell_h, cell_w), dtype=np.uint8)
#     x = random.randint(0, bg_w - cell_w)
#     y = random.randint(0, bg_h - cell_h)
#     center = (x + cell_w // 2, y + cell_h // 2)
#     background = cv2.seamlessClone(cell, background, mask, center, cv2.NORMAL_CLONE)
#     return background
#
#
# def add_gaussian_noise(image, mean=0, sigma=1):
#     noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
#     noisy_image = cv2.add(image, noise)
#     return noisy_image
#
#
# def add_constant_noise(image, value=20):
#     noisy_image = cv2.add(image, value)
#     return noisy_image
#
#
# def data_generator(cell_patch_dir="data/cells", use_generated_circles=True):
#     while True:
#         background = create_background_from_patches()
#         clean_image = background.copy()
#         num_cells = random.randint(NUM_CELLS_RANGE[0], NUM_CELLS_RANGE[1])
#         cell_files = glob(os.path.join(cell_patch_dir, "*.png"))
#         for _ in range(num_cells):
#             cell = cv2.imread(random.choice(cell_files), cv2.IMREAD_GRAYSCALE)
#             cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
#             bg_h, bg_w, _ = clean_image.shape
#             cell_h, cell_w, _ = cell.shape
#             x = random.randint(0, bg_w - cell_w)
#             y = random.randint(0, bg_h - cell_h)
#             clean_image = place_cell(clean_image, cell)
#         noisy_gaus_image = add_gaussian_noise(clean_image.copy())
#         noisy_constant_image = add_constant_noise(clean_image.copy())
#         yield noisy_gaus_image, noisy_constant_image, clean_image
#
#
# generator = data_generator()
# for i in range(3):
#     noisy_gaus_image, noisy_constant_image, clean_image = next(generator)
#     cv2.imwrite(f"noisy_gaus_image_{i}.png", noisy_gaus_image)
#     cv2.imwrite(f"noisy_constant_image{i}.png", noisy_constant_image)
#     cv2.imwrite(f"clean_image_{i}.png", clean_image)
#
#     plt.figure(figsize=(12, 4))  # Создаем фигуру, задаем размер
#     plt.subplot(1, 3, 1)  # 1 строка, 3 столбца, первый график
#     plt.imshow(clean_image, cmap='gray')
#     plt.title("clean")
#     plt.axis('off')
#
#     plt.subplot(1, 3, 2)  # 1 строка, 3 столбца, второй график
#     plt.imshow(noisy_gaus_image, cmap='gray')
#     plt.title("gaus")
#     plt.axis('off')
#
#     plt.subplot(1, 3, 3)  # 1 строка, 3 столбца, третий график
#     plt.imshow(noisy_constant_image, cmap='gray')
#     plt.title("constant")
#     plt.axis('off')
#
#     plt.tight_layout()  # Автоматическая настройка отступов между графиками
#     plt.show()