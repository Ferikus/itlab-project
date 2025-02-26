from prep import *


"""Одноканальное представление маски"""

volume_id = 1
slice_id = 60
file_path = os.path.join(data_path, f"volume_{volume_id}_slice_{slice_id}.h5")

image, mask = get_slice(file_path)

mask_combined = combine_mask(mask)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Исходное изображение
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Исходное изображение")
axes[0].grid(False)

# Наложение маски
axes[1].imshow(image, cmap='gray')
axes[1].imshow(mask_combined, cmap='gray', alpha=0.5)
axes[1].set_title("Наложение масок")
axes[1].grid(False)

# Объединённая маска
axes[2].imshow(mask_combined, cmap='gray')
axes[2].set_title("Объединенная маска")
axes[2].grid(False)

plt.show()


"""RGB-представление маски"""

mask_rgb = combine_rgb_mask(mask)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Исходное изображение
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Исходное изображение")
axes[0].grid(False)

# Наложение маски
axes[1].imshow(image, cmap='gray')
axes[1].imshow(mask_rgb, alpha=0.5)
axes[1].set_title("Наложение масок")
axes[1].grid(False)


# Маска
axes[2].imshow(mask_rgb)
axes[2].set_title("Трехканальная маска")
axes[2].grid(False)

plt.show()