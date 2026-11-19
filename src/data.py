from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import v2

from .config import load_json, save_json
from .paths import DATA_DIR, SSL_SPLIT_PATH, TARGET_SPLIT_PATH

CIFAR_SUPERCLASSES = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit and veg": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "electricals": ["clock", "keyboard", "lamp", "telephone", "television"],
    "furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large buildings": ["bridge", "castle", "house", "road", "skyscraper"],
    "natural scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "weird vehicles": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}


class IndexedImageDataset(Dataset):
    """Wraps a torchvision dataset so frozen index manifests remain valid."""

    def __init__(self, base: Dataset, indices: list[int], transform, label_map: dict[int, int] | None = None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        image, label = self.base[self.indices[position]]
        if self.label_map is not None:
            label = self.label_map[int(label)]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class RotationDataset(Dataset):
    """SSL rotation-prediction dataset from Practical 3 / shared Colab bootstrap."""

    def __init__(self, images: np.ndarray, indices: list[int], transform, training: bool = True):
        self.images = images
        self.indices = list(indices)
        self.transform = transform
        self.training = training

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        image = Image.fromarray(self.images[self.indices[position]])
        rotation_label = int(torch.randint(0, 4, (1,)).item()) if self.training else position % 4
        angle = rotation_label * 90
        if angle:
            image = image.rotate(angle, expand=False)
        return self.transform(image), rotation_label


def _normalize(cfg: dict) -> transforms.Normalize:
    stats = cfg["preprocessing"]
    return transforms.Normalize(mean=stats["normalize_mean"], std=stats["normalize_std"])


def ssl_transforms(cfg: dict) -> tuple[Any, Any]:
    img_size = cfg["image_size"]
    train_tf = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.5),
            v2.RandomGrayscale(p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg["preprocessing"]["normalize_mean"], std=cfg["preprocessing"]["normalize_std"]),
        ]
    )
    val_tf = transforms.Compose([transforms.ToTensor(), _normalize(cfg)])
    return train_tf, val_tf


def supervised_transforms(cfg: dict) -> tuple[Any, Any]:
    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize(cfg),
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), _normalize(cfg)])
    return train_tf, eval_tf


def load_source_dataset(cfg: dict, train: bool = True, download: bool = True):
    name = cfg["source_dataset"]
    if name != "CIFAR10":
        raise NotImplementedError(
            f"Source dataset {name!r} is not wired yet. Keep CIFAR10 for the prototype "
            "and add a loader here when switching to the larger dataset."
        )
    return CIFAR10(root=str(DATA_DIR), train=train, download=download)


def load_target_dataset(cfg: dict, train: bool = True, download: bool = True):
    name = cfg["target_dataset"]
    if name != "CIFAR100":
        raise NotImplementedError(
            f"Target dataset {name!r} is not wired yet. Keep CIFAR100 for the prototype "
            "and add a loader here when switching to the larger dataset."
        )
    return CIFAR100(root=str(DATA_DIR), train=train, download=download)


def build_target_split_manifest(cfg: dict) -> dict[str, Any]:
    target_train = load_target_dataset(cfg, train=True)
    target_test = load_target_dataset(cfg, train=False)
    classes = cfg["target_classes"]
    if len(classes) != 5 or len(set(classes)) != 5:
        raise ValueError("Prototype target subset must be exactly five unique classes.")
    missing = [c for c in classes if c not in target_train.classes]
    if missing:
        raise ValueError(f"Unknown target classes: {missing}")

    class_ids = [target_train.class_to_idx[c] for c in classes]
    indices = np.flatnonzero(np.isin(np.asarray(target_train.targets), class_ids))
    labels = np.asarray(target_train.targets)[indices]
    train_idx, val_idx = train_test_split(
        indices,
        test_size=1 - cfg["target_train_fraction"],
        random_state=cfg["seed"],
        stratify=labels,
    )
    test_idx = np.flatnonzero(np.isin(np.asarray(target_test.targets), class_ids))
    return {
        "version": cfg["version"],
        "seed": cfg["seed"],
        "dataset": cfg["target_dataset"],
        "classes": classes,
        "original_class_ids": class_ids,
        "train_indices": sorted(map(int, train_idx)),
        "val_indices": sorted(map(int, val_idx)),
        "test_indices": sorted(map(int, test_idx)),
        "label_map": {str(cid): i for i, cid in enumerate(class_ids)},
    }


def build_ssl_split_manifest(cfg: dict) -> dict[str, Any]:
    source_train = load_source_dataset(cfg, train=True)
    all_indices = np.arange(len(source_train))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=cfg["ssl_val_fraction"],
        random_state=cfg["seed"],
        shuffle=True,
    )
    return {
        "version": cfg["version"],
        "dataset": cfg["source_dataset"],
        "seed": cfg["seed"],
        "pretext": cfg["ssl_pretext"],
        "train_indices": sorted(map(int, train_idx)),
        "val_indices": sorted(map(int, val_idx)),
    }


def write_frozen_splits(cfg: dict) -> tuple[Path, Path]:
    save_json(TARGET_SPLIT_PATH, build_target_split_manifest(cfg))
    save_json(SSL_SPLIT_PATH, build_ssl_split_manifest(cfg))
    return TARGET_SPLIT_PATH, SSL_SPLIT_PATH


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers else None,
        generator=generator if shuffle else None,
    )


def ssl_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    source = load_source_dataset(cfg, train=True)
    manifest = load_json(SSL_SPLIT_PATH)
    train_tf, val_tf = ssl_transforms(cfg)
    train_ds = RotationDataset(source.data, manifest["train_indices"], train_tf, training=True)
    val_ds = RotationDataset(source.data, manifest["val_indices"], val_tf, training=False)
    bs = cfg["training"]["ssl_batch_size"]
    return (
        make_loader(train_ds, bs, True, cfg["seed"]),
        make_loader(val_ds, bs, False, cfg["seed"]),
    )


def target_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_base = load_target_dataset(cfg, train=True)
    test_base = load_target_dataset(cfg, train=False)
    manifest = load_json(TARGET_SPLIT_PATH)
    label_map = {int(k): int(v) for k, v in manifest["label_map"].items()}
    train_tf, eval_tf = supervised_transforms(cfg)
    train_ds = IndexedImageDataset(train_base, manifest["train_indices"], train_tf, label_map)
    val_ds = IndexedImageDataset(train_base, manifest["val_indices"], eval_tf, label_map)
    test_ds = IndexedImageDataset(test_base, manifest["test_indices"], eval_tf, label_map)
    bs = cfg["training"]["batch_size"]
    return (
        make_loader(train_ds, bs, True, cfg["seed"]),
        make_loader(val_ds, bs, False, cfg["seed"]),
        make_loader(test_ds, bs, False, cfg["seed"]),
    )
