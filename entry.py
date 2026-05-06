import torch
import torch.nn as NN
import torch.nn.functional as func
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as tfunc2
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from torchvision.ops import masks_to_boxes
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert, generalized_box_iou
import matplotlib.pyplot as plt

import argparse
from enum import Enum
from itertools import zip_longest, chain
import json
import random
import time
import os
import math
from collections import Counter

from utils import encode_mask as rle_encode
from utils import decode_maskobj as rle_decode_maskobj
from utils import read_maskfile as rle_read_maskfile
from utils import get_maskobj_instances as rle_maskobj_get_instances



class RunMode(str, Enum):
    TRAIN = "train"
    INFER = "infer"

def init_device():
    device_string = "cuda" if torch.cuda.is_available() else\
        "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_string)
    
    return device, device_string 

def parse_cmd():
    parser = argparse.ArgumentParser(
        description="Select run mode"
    )

    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        choices=[mode.value for mode in RunMode],
        help="Select which operation you'd like to perform: " + ", ".join(
            [mode.value for mode in RunMode])
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Select image root directory"
    )

    parser.add_argument(
        "--validation_ratio",
        default=0.2,
        type=float,
        help="Specify the ratio of validation/training images"
    )

    return parser.parse_args()


class ImageDataset(Dataset):
    def __init__(self, root_dir=None, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        if root_dir != None:
            self.sample_paths = sorted(
                [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
            )

        # 0 is background for torchvision detection models
        self.class_to_label = {"class1": 1,"class2": 2,"class3": 3,"class4": 4}
        self.label_to_class = ["bg","class1","class2","class3","class4"]

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, self.sample_paths[idx])

        file_paths = os.listdir(sample_path)
        image_path = [os.path.join(sample_path, f) for f in file_paths if f.startswith("image")][0]
        class_paths = [os.path.join(sample_path, f) for f in file_paths if f.startswith("class")]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size

            if self.transforms is not None:
                image = self.transforms(img)
            else:
                image = tfunc2.to_dtype(
                    tfunc2.to_image(img),
                    dtype=torch.float32,
                    scale=True
                )

        all_masks = []
        all_labels = []

        for instance_mask_path in class_paths:
            class_name = os.path.basename(instance_mask_path).split(".")[0]
            class_label = self.class_to_label[class_name]

            mask_img = rle_read_maskfile(instance_mask_path)

            instance_masks = rle_maskobj_get_instances(mask_img)

            for m_idx in instance_masks:
                m = torch.as_tensor(mask_img == m_idx, dtype=torch.uint8)
                if m.sum() == 0:
                    continue
                all_masks.append(m)
                all_labels.append(class_label)

        if len(all_masks) == 0:
            masks = torch.zeros((0, orig_h, orig_w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.stack(all_masks, dim=0)                  # [N, H, W]
            labels = torch.tensor(all_labels, dtype=torch.int64)   # [N]
            boxes = masks_to_boxes(masks)          # [N, 4]
            area = masks.flatten(1).sum(dim=1).to(torch.float32)   # [N]
            iscrowd = torch.zeros((masks.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "sample_id": self.sample_paths[idx]
        }

        return image, target
    
    def split(self, split_items):
        other = ImageDataset()
        other.root_dir = self.root_dir
        other.transforms = self.transforms
        
        other.sample_paths = [item[0] for item in split_items]
        self.sample_paths = [f for f in self.sample_paths if f not in [name for name, data in split_items]]

        for path in other.sample_paths:
            if path in self.sample_paths:
                print("split bug")
        for path in self.sample_paths:
            if path in other.sample_paths:
                print("split bug other way")
        
        return other

import torch


def move_target_to_device(target, device):
    moved = {}

    for key, value in target.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()

    running_loss = 0.0

    for step, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [move_target_to_device(target, device) for target in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_value for loss_value in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % 10 == 0:
            loss_items = {
                key: float(value.detach().cpu())
                for key, value in loss_dict.items()
            }

            print(
                f"epoch={epoch}, step={step}, "
                f"loss={loss.item():.4f}, losses={loss_items}"
            )

    return running_loss / len(data_loader)


def get_dataset_statistics(dataset):
    print(f"Gathering dataset statistics for dataset of size {len(dataset)}")

    # Dataset uses:
    # 0 = background
    # 1 = class1
    # 2 = class2
    # 3 = class3
    # 4 = class4
    label_to_class = dataset.label_to_class
    class_names = label_to_class[1:]  # ignore "bg"

    class_basis_map = {
        class_name: i for i, class_name in enumerate(class_names)
    }

    max_width = 0
    max_height = 0
    max_instances = 0

    min_width = float("inf")
    min_height = float("inf")
    min_instances = float("inf")

    mean_width = 0
    mean_height = 0
    mean_instances = 0

    instance_counts = {class_name: 0 for class_name in class_names}
    image_counts = {class_name: 0 for class_name in class_names}

    image_vectors = {}

    for image, target in dataset:
        sample_id = target["sample_id"]

        height, width = target["orig_size"].tolist()

        labels = target["labels"]  # shape [N]
        instances = len(labels)

        image_vectors[sample_id] = [0 for _ in class_names]

        label_counts = Counter(labels.tolist())

        for label, count in label_counts.items():
            if label == 0:
                continue

            class_name = label_to_class[label]

            instance_counts[class_name] += count
            image_counts[class_name] += 1

            image_vectors[sample_id][class_basis_map[class_name]] = 1

        max_width = max(max_width, width)
        max_height = max(max_height, height)
        max_instances = max(max_instances, instances)

        min_width = min(min_width, width)
        min_height = min(min_height, height)
        min_instances = min(min_instances, instances)

        mean_width += width
        mean_height += height
        mean_instances += instances

    dataset_len = len(dataset)

    if dataset_len > 0:
        mean_width /= dataset_len
        mean_height /= dataset_len
        mean_instances /= dataset_len
    else:
        min_width = 0
        min_height = 0
        min_instances = 0

    print("Dataset Statistics:")
    print(f"\tmax_width: {max_width}") 
    print(f"\tmax_height: {max_height}") 
    print(f"\tmax_instances: {max_instances}") 
    print(f"\tmin_width: {min_width}") 
    print(f"\tmin_height: {min_height}") 
    print(f"\tmin_instances: {min_instances}") 
    print(f"\tmean_width: {mean_width}")
    print(f"\tmean_height: {mean_height}")
    print(f"\tmean_instances: {mean_instances}")

    print("\tClass Statistics:")
    print("\tInstance counts:")
    print(instance_counts)
    print("\tImage counts:")
    print(image_counts)

    return [(k, v) for k, v in image_vectors.items()]

def sum_dataset(dataset):
    sums = [0,0,0,0]
    for sample_name, l0_norm in dataset:
        for i in range(4):
            sums[i] += l0_norm[i]
    return sums

def get_progress(current, target):
    return [current[i] / target[i] for i in range(4)]


def get_instance_segmenter(num_classes=5):
    model = maskrcnn_resnet50_fpn_v2(
        weights=None, #coco pretrained weights not allowed, so will keep this as None
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
        trainable_backbone_layers=3,

        box_detections_per_img=1000,

        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
    )

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__=='__main__':
    args = parse_cmd()

    device, device_string = init_device()
    print(f"Using {device_string} backend ")
    if args.data_path == None:
        print("No data path specified, exiting")
        quit()
    data_path = Path("/".join([args.data_path, 'train']))
    
    training_dataset = ImageDataset(str(data_path))
    image, target = training_dataset[0]
    print(type(image), image.shape, image.dtype)
    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape, v.dtype)
        else:
            print(k, type(v), v)
    
    
    image_vectors = get_dataset_statistics(training_dataset)
    sums = sum_dataset(image_vectors)
    even_split_goal = []
    for i in range(4):
        even_split_goal.append(sums[i] * args.validation_ratio)
    
    val_image_count = len(training_dataset) * args.validation_ratio
    print(f"Want { \
        math.ceil(val_image_count)} validation images having [class1,class2,class3,class4] totals exceeding {[math.floor(t) for t in even_split_goal]}")

    training_items = image_vectors
    print(len(training_items))
    validation_items = []
    for i in range(10):
        next_choice = training_items.pop(random.randint(0,len(training_items)-1))
        validation_items.append(next_choice)
        for x in range(4):
            sums[x] -= next_choice[1][x]
    
    valid_sum = sum_dataset(validation_items)
    
    progress = get_progress(valid_sum, even_split_goal)
    remaining = val_image_count - 10
    while min(progress) < 1.0 or remaining > 0:
        direction = progress.index(min(progress))
        
        training_items = sorted(training_items, key=lambda x: x[1][direction], reverse=True)
        next_choice = training_items.pop(random.randint(0,sums[direction]-1))
        validation_items.append(next_choice)
        for x in range(4):
            sums[x] -= next_choice[1][x]
            valid_sum[x] += next_choice[1][x]
        remaining -= 1
        progress = get_progress(valid_sum, even_split_goal)

    print("Random Dataset Split Totals:")
    print(f"\tvalidation dataset {len(validation_items)} items, class totals: {sum_dataset(validation_items)}")
    print(f"\ttraining dataset {len(training_items)} items, class totals: {sum_dataset(training_items)}")
    validation_dataset = training_dataset.split(validation_items)

    # print("Validation Dataset Statistics")
    # get_dataset_statistics(validation_dataset)
    # print("Training Dataset Statistics")
    # get_dataset_statistics(training_dataset)

    #print(len(training_dataset) + len(validation_dataset))


    model = get_instance_segmenter(5)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    for epoch in range(1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        print(f"epoch={epoch}, train_loss={train_loss:.4f}")