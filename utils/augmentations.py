import albumentations as A
from albumentations.pytorch import ToTensorV2

size = 64
in_chans = 64

# ============== augmentation =============
train_aug_list = [
    # A.RandomResizedCrop(
    #     size, size, scale=(0.85, 1.0)),
    A.Resize(size, size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.RandomRotate90(p=0.6),

    #A.RandomBrightnessContrast(p=0.15),
    A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.15,p=0.75),
    
    A.OneOf([
            A.GaussNoise(var_limit=[0.0, 0.01]),
            A.GaussianBlur(),
            #A.MotionBlur(),
            ], p=0.5),
    
    # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.CoarseDropout(max_holes=5, max_width=int(size * 0.02), max_height=int(size * 0.02), 
                    mask_fill_value=0, p=0.5),
    # A.Cutout(max_h_size=int(size * 0.6),
    #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
    A.Normalize(
        mean= [0] * in_chans,
        std= [1] * in_chans
    ),
    ToTensorV2(transpose_mask=True),
]

valid_aug_list = [
    A.Resize(size, size),
    A.Normalize(
        mean= [0] * in_chans,
        std= [1] * in_chans
    ),
    ToTensorV2(transpose_mask=True),
]


train_transformations = A.Compose(train_aug_list)
valid_transformations = A.Compose(valid_aug_list)