from data_analyzer.utils.data_loader import *
from data_analyzer.utils.visualization import *
from pipeline import *

data_path = r"BraTS2020_training_data\content\data"
output_path = r"BraTS2020_training_data\content\data_augmented"
os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'masks'), exist_ok=True)

for volume_id in range(3):
    for slice_id in range(155):
        file_path = os.path.join(data_path, f"volume_{volume_id + 1}_slice_{slice_id}.h5")
        img, mask = get_slice_data(file_path)
        mask = combine_rgb_mask(mask)
        # plt.imshow(img, cmap='gray')

        augmented = transform(image=img, mask=mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']
        # plt.imshow(mask_aug, cmap='gray')

        img_name = f"volume_{volume_id + 1}_slice_{slice_id}_aug.png"
        mask_name = f"volume_{volume_id + 1}_slice_{slice_id}_aug_mask.png"

        cv2.imwrite(os.path.join(output_path, 'images', img_name), img_aug)
        print(img_name)
        cv2.imwrite(os.path.join(output_path, 'masks', mask_name), mask_aug)
        print(mask_name)

print("Аугментация завершена")