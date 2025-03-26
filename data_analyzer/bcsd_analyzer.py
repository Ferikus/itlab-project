import os
from utils.data_loader import *
from utils.stats_analyzer import *
from utils.visualization import *


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "BCCD Dataset with mask")
    print(data_dir)

    image_paths, mask_paths = load_image_and_mask_paths_bcsd(data_dir)
    validate_dataset(image_paths, mask_paths)

    # Вывод статистик
    # stats = calculate_stats_bcsd(image_paths, mask_paths)
    # show_stats(stats)

    # Визуализация случайного образца
    visualize_bcsd_samples(image_paths, mask_paths, num_samples=4)
    # idx = np.random.randint(0, len(image_paths))
    # visualize_bcsd_sample(image_paths[idx], mask_paths[idx])