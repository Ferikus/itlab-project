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