from prep import *
import albumentations as A


transform = A.Compose([
    A.RandomScale(scale_limit=(0.8, 1.2), p=0.5),
    A.Rotate(limit=180, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.GridDistortion(
        num_steps=5,
        distort_limit=0.2,
        p=0.3
    ),
    A.Resize(height=240, width=240),

    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, 0.4),
        contrast_limit=(-0.3, 0.3),
        p=0.8
    ),
    A.RandomGamma(gamma_limit=(70, 130), p=0.3),

    # A.GaussNoise(std_range=(0.02, 0.1), p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
])

data_path = r"D:\Datasets\BraTS2020_training_data\content\data"
output_path = r"D:\Datasets\BraTS2020_training_data\content\data_augmented"
os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'masks'), exist_ok=True)


for volume_id in range(3):
    for slice_id in range(155):
        file_path = os.path.join(data_path, f"volume_{volume_id + 1}_slice_{slice_id}.h5")
        print(file_path)
        img, mask = get_slice(file_path)
        mask = combine_rgb_mask(mask)

        augmented = transform(image=img, mask=mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']

        img_name = f"volume_{volume_id + 1}_slice_{slice_id}_aug.png"
        mask_name = f"volume_{volume_id + 1}_slice_{slice_id}_aug_mask.png"

        cv2.imwrite(os.path.join(output_path, 'images', img_name), img_aug)
        print(img_name)
        cv2.imwrite(os.path.join(output_path, 'masks', mask_name), mask_aug)
        print(mask_name)

print("Аугментация завершена")