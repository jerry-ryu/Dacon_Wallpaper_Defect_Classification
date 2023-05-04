from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

## Dataset

class BaseDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
    def __len__(self):
        return len(self.img_path_list)


## Train augmentation

class BaseTrainAugmentation:
    def __init__(self, resize, **args):
        self.transform = A.Compose([
    A.Resize(resize, resize),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
    ])

    def __call__(self, image):
        return self.transform(image=image)


## Test augmentation

class BaseTestAugmentation:
    def __init__(self, resize, **args):
        self.transform = A.Compose([
    A.Resize(resize, resize),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
    ])

    def __call__(self, image):
        return self.transform(image=image)




# datset, train/test augmentaition list
_dataset_entrypoints = {
    'BaseDataset': BaseDataset
}

_train_augmentation_entrypoints = {
    'BaseTrainAugmentation': BaseTrainAugmentation
}

_test_augmentation_entrypoints= {
    'BaseTestAugmentation': BaseTestAugmentation
}

# dataset function
def dataset_entrypoint(dataset_name):
    return _dataset_entrypoints[dataset_name]


def is_dataset(dataset_name):
    return dataset_name in _dataset_entrypoints


def create_dataset(dataset_name, **kwargs):
    if is_dataset(dataset_name):
        create_fn = dataset_entrypoint(dataset_name)
        dataset = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return dataset

# train_augmentation function
def train_augmentation_entrypoint(train_augmentation_name):
    return _train_augmentation_entrypoints[train_augmentation_name]


def is_train_augmentation(train_augmentation_name):
    return train_augmentation_name in _train_augmentation_entrypoints


def create_train_augmentation(train_augmentation_name, **kwargs):
    if is_train_augmentation(train_augmentation_name):
        create_fn = train_augmentation_entrypoint(train_augmentation_name)
        train_augmentation = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return train_augmentation

# test_augmentation function
def test_augmentation_entrypoint(test_augmentation_name):
    return _test_augmentation_entrypoints[test_augmentation_name]


def is_test_augmentation(test_augmentation_name):
    return test_augmentation_name in _test_augmentation_entrypoints


def create_test_augmentation(test_augmentation_name, **kwargs):
    if is_test_augmentation(test_augmentation_name):
        create_fn = test_augmentation_entrypoint(test_augmentation_name)
        test_augmentation = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return test_augmentation