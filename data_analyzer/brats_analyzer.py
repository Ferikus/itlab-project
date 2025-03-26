import os
import random
from utils.data_loader import *
from utils.stats_analyzer import *
from utils.visualization import *


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "BraTS2020_training_data", "content", "data")
    files = load_and_sort_files_brats(data_dir)

    volume_id = random.randint(0, 154)

    start_slice = 60
    end_slice = 80
    num_slices = 4

    # Вывод статистик

    # Распределение по классам
    image_volume, mask_volume = load_volume_slices(data_dir, volume_id)
    class_dist = calculate_class_distribution(mask_volume)

    print("\nСтатистика распределения классов:")
    for cls, perc in class_dist.items():
        print(f"{cls}: {perc * 100:.2f}%")

    # Геометрия опухоли на примере конкретных срезов

    slice_indices = np.linspace(start_slice, end_slice, num_slices, dtype=int)

    for slice_index in slice_indices:
        mask_slice = mask_volume[:, :, :, slice_index]

        geometry = calculate_tumor_geometry(mask_slice)

        print(f"\nАнализ среза {slice_index}:")
        print(f"Количество регионов опухоли: {geometry['num_regions']}")
        print(f"Общая площадь опухоли: {geometry['total_area']:.2f} px²")
        print(f"Средняя площадь региона: {geometry['total_area'] / geometry['num_regions']:.2f} px²"
              if geometry['num_regions'] > 0 else "Нет опухолевых регионов")

    # Пример визуализации срезов в одном томе
    visualize_volume_slices(data_dir, volume_id, start_slice, end_slice, num_slices)