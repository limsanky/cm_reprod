import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
import os
import pickle
import torchvision.transforms.v2 as transforms
import torch 

# class CIFAR10(Dataset):
#     """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

#     Args:
#         root (string): Root directory of dataset where directory
#             ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
#         train (bool, optional): If True, creates dataset from training set, otherwise
#             creates from test set.
#         transform (callable, optional): A function/transform that takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.

#     """

#     base_folder = "cifar-10-batches-py"
#     url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
#     filename = "cifar-10-python.tar.gz"
#     tgz_md5 = "c58f30108f718f92721af3b95e74349a"
#     train_list = [
#         ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
#         ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
#         ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
#         ["data_batch_4", "634d18415352ddfa80567beed471001a"],
#         ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
#     ]

#     test_list = [
#         ["test_batch", "40351d587109b95175f43aff81a1287e"],
#     ]
#     meta = {
#         "filename": "batches.meta",
#         "key": "label_names",
#         "md5": "5ff9c542aee3614f3951f8cda6e48888",
#     }

#     def __init__(
#         self,
#         root: str,
#         train: bool = True,
#         transform = None,
#         shard : int = 0,
#         num_shards: int = 1,
#     ) -> None:

#         super().__init__()

#         self.train = train  # training set or test set
#         self.root = root
#         self.transform = transform
#         if self.train:
#             downloaded_list = self.train_list
#         else:
#             downloaded_list = self.test_list

#         self.data: Any = []
#         self.targets = []

#         # now load the picked numpy arrays
#         for file_name, checksum in downloaded_list:
#             file_path = os.path.join(self.root, self.base_folder, file_name)
#             with open(file_path, "rb") as f:
#                 entry = pickle.load(f, encoding="latin1")
#                 self.data.append(entry["data"])
#                 if "labels" in entry:
#                     self.targets.extend(entry["labels"])
#                 else:
#                     self.targets.extend(entry["fine_labels"])

#         self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
#         self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

#         self.data = self.data[shard:][::num_shards]
#         self.targets = self.targets[shard:][::num_shards]

#         # self._load_meta()

#     # def _load_meta(self) -> None:
#     #     path = os.path.join(self.root, self.base_folder, self.meta["filename"])
#     #     if not check_integrity(path, self.meta["md5"]):
#     #         raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
#     #     with open(path, "rb") as infile:
#     #         data = pickle.load(infile, encoding="latin1")
#     #         self.classes = data[self.meta["key"]]
#     #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

#     def __getitem__(self, index: int):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.targets[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, target

#     def __len__(self) -> int:
#         return len(self.data)

def load_data(
    *,
    datasetname,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    num_workers=2,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    assert datasetname == "cifar"
    if datasetname == "cifar":
        transforms_list = []
        if random_crop:
            transforms_list.append(transforms.RandomCrop(image_size))
        if random_flip:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToImage())
        transforms_list.append(transforms.ToDtype(torch.float32, scale=True))
        # dataset = CIFAR10(
        #     root=data_dir,
        #     train=True,
        #     transform=transforms.Compose(transforms_list),
        # )
        dataset = CIFAR10(
            root=data_dir,
            train=True,
            transform=transforms.Compose(transforms_list),
        )
    else:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # arr = arr.astype(np.float32) / 127.5 - 1
        arr = arr.astype(np.float32) / 255.0

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
