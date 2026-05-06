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


import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

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

    parser.add_argument(
        "--training_epochs",
        default=1,
        type=int,
        help="number of training epochs, default to 1"
    )

    parser.add_argument(
        "--crop_size",
        default=512,
        type=int,
        help="image crop size (memory constraint)"
    )

    return parser.parse_args()


class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        transforms=None,
        crop_size=512,
        random_crop=True,
        crop_trials=10,
        min_instances_in_crop=1,
    ):
        self.root_dir = root_dir
        self.transforms = transforms

        self.crop_size = crop_size
        self.random_crop = random_crop
        self.crop_trials = crop_trials
        self.min_instances_in_crop = min_instances_in_crop

        if root_dir is not None:
            self.sample_paths = sorted(
                [
                    f for f in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, f))
                ]
            )
        else:
            self.sample_paths = []

        # 0 is background for torchvision detection models
        self.class_to_label = {
            "class1": 1,
            "class2": 2,
            "class3": 3,
            "class4": 4,
        }
        self.label_to_class = ["bg", "class1", "class2", "class3", "class4"]

    def __len__(self):
        return len(self.sample_paths)

    def _load_class_maps(self, class_paths):
        """
        Load each class mask image as an instance-ID map, e.g. pixels with value
        1 belong to one instance, pixels with value 2 belong to another, etc.
        """
        class_mask_maps = {}

        for class_mask_path in class_paths:
            class_name = os.path.basename(class_mask_path).split(".")[0]
            class_mask_maps[class_name] = rle_read_maskfile(class_mask_path)

        return class_mask_maps

    def _get_crop_hw(self, orig_h, orig_w):
        crop_h = min(self.crop_size, orig_h)
        crop_w = min(self.crop_size, orig_w)
        return crop_h, crop_w

    def _sample_crop_coords(self, orig_h, orig_w):
        crop_h, crop_w = self._get_crop_hw(orig_h, orig_w)

        if self.random_crop:
            y0 = 0 if orig_h == crop_h else random.randint(0, orig_h - crop_h)
            x0 = 0 if orig_w == crop_w else random.randint(0, orig_w - crop_w)
        else:
            y0 = (orig_h - crop_h) // 2
            x0 = (orig_w - crop_w) // 2

        y1 = y0 + crop_h
        x1 = x0 + crop_w

        return x0, y0, x1, y1

    def _count_instances_in_crop(self, class_mask_maps, crop_box):
        x0, y0, x1, y1 = crop_box
        total_instances = 0

        for class_name, mask_img in class_mask_maps.items():
            crop_mask = mask_img[y0:y1, x0:x1]
            instance_ids = rle_maskobj_get_instances(crop_mask)
            total_instances += len(instance_ids)

        return total_instances

    def _choose_crop_box(self, orig_h, orig_w, class_mask_maps):
        """
        Try a few crop candidates and prefer one containing at least
        min_instances_in_crop instances.
        """
        if self.crop_size is None:
            return (0, 0, orig_w, orig_h)

        best_crop = None
        best_count = -1

        for _ in range(self.crop_trials):
            crop_box = self._sample_crop_coords(orig_h, orig_w)
            instance_count = self._count_instances_in_crop(class_mask_maps, crop_box)

            if instance_count >= self.min_instances_in_crop:
                return crop_box

            if instance_count > best_count:
                best_count = instance_count
                best_crop = crop_box

        # fallback: return the best crop we saw, even if empty
        return best_crop

    def __getitem__(self, idx):
        sample_id = self.sample_paths[idx]
        sample_path = os.path.join(self.root_dir, sample_id)

        file_paths = os.listdir(sample_path)
        image_path = [
            os.path.join(sample_path, f)
            for f in file_paths
            if f.startswith("image")
        ][0]

        class_paths = [
            os.path.join(sample_path, f)
            for f in file_paths
            if f.startswith("class")
        ]

        # Load image metadata first
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size

            # Load full per-class instance-ID maps
            class_mask_maps = self._load_class_maps(class_paths)

            # Choose crop
            crop_box = self._choose_crop_box(orig_h, orig_w, class_mask_maps)
            x0, y0, x1, y1 = crop_box

            # Crop image first
            img = img.crop((x0, y0, x1, y1))
            crop_w, crop_h = img.size

            # Convert image to tensor
            if self.transforms is not None:
                # Keep this for photometric transforms only unless you explicitly
                # make the same geometric transform happen to masks.
                image = self.transforms(img)
            else:
                image = tfunc2.to_dtype(
                    tfunc2.to_image(img),
                    dtype=torch.float32,
                    scale=True
                )

        # Now build instance masks ONLY from the crop
        all_masks = []
        all_labels = []

        for class_name, full_mask_img in class_mask_maps.items():
            class_label = self.class_to_label[class_name]

            crop_mask_img = full_mask_img[y0:y1, x0:x1]
            instance_ids = rle_maskobj_get_instances(crop_mask_img)

            for inst_id in instance_ids:
                m = torch.as_tensor(crop_mask_img == inst_id, dtype=torch.uint8)

                if m.sum() == 0:
                    continue

                all_masks.append(m)
                all_labels.append(class_label)

        if len(all_masks) == 0:
            masks = torch.zeros((0, crop_h, crop_w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.stack(all_masks, dim=0)
            labels = torch.tensor(all_labels, dtype=torch.int64)

            boxes = masks_to_boxes(masks)
            area = masks.flatten(1).sum(dim=1).to(torch.float32)

            min_mask_area = 4

            keep = (
                (boxes[:, 2] > boxes[:, 0]) &
                (boxes[:, 3] > boxes[:, 1]) &
                (area >= min_mask_area)
            )

            masks = masks[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            area = area[keep]

            if masks.shape[0] == 0:
                masks = torch.zeros((0, crop_h, crop_w), dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                area = torch.zeros((0,), dtype=torch.float32)

            iscrowd = torch.zeros((masks.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx], dtype=torch.int64),

            # IMPORTANT: these should match the CURRENT tensor/image size
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([crop_h, crop_w], dtype=torch.int64),

            # extra metadata
            "sample_id": sample_id,
            "crop_box": torch.tensor([x0, y0, x1, y1], dtype=torch.int64),
            "source_orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
        }

        return image, target

    def split(self, split_items):
        other = ImageDataset()
        other.root_dir = self.root_dir
        other.transforms = self.transforms

        other.crop_size = self.crop_size
        other.random_crop = self.random_crop
        other.crop_trials = self.crop_trials
        other.min_instances_in_crop = self.min_instances_in_crop

        other.class_to_label = self.class_to_label
        other.label_to_class = self.label_to_class

        other.sample_paths = [item[0] for item in split_items]

        split_names = {name for name, data in split_items}
        self.sample_paths = [f for f in self.sample_paths if f not in split_names]

        return other


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
    print(f"Gathering LIGHT dataset statistics for dataset of size {len(dataset)}")

    class_names = dataset.label_to_class[1:]

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

    for sample_id in dataset.sample_paths:
        sample_path = os.path.join(dataset.root_dir, sample_id)
        file_paths = os.listdir(sample_path)

        image_path = [
            os.path.join(sample_path, f)
            for f in file_paths
            if f.startswith("image")
        ][0]

        class_paths = [
            os.path.join(sample_path, f)
            for f in file_paths
            if f.startswith("class")
        ]

        with Image.open(image_path) as img:
            width, height = img.size

        image_vectors[sample_id] = [0, 0, 0, 0]

        instances_in_image = 0

        for class_path in class_paths:
            class_name = os.path.basename(class_path).split(".")[0]
            class_idx = dataset.class_to_label[class_name] - 1

            mask_img = rle_read_maskfile(class_path)
            instance_ids = rle_maskobj_get_instances(mask_img)

            num_instances = len(instance_ids)

            if num_instances > 0:
                image_vectors[sample_id][class_idx] = 1
                image_counts[class_name] += 1
                instance_counts[class_name] += num_instances
                instances_in_image += num_instances

        max_width = max(max_width, width)
        max_height = max(max_height, height)
        max_instances = max(max_instances, instances_in_image)

        min_width = min(min_width, width)
        min_height = min(min_height, height)
        min_instances = min(min_instances, instances_in_image)

        mean_width += width
        mean_height += height
        mean_instances += instances_in_image

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


def get_instance_segmenter(num_classes=5, image_size=512):
    model = maskrcnn_resnet50_fpn_v2(
        weights=None, #coco pretrained weights not allowed, so will keep this as None
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
        trainable_backbone_layers=3,
        min_size=1,
        max_size=image_size,
        
        box_detections_per_img=300,
        
        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=300,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=300,
    )

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def get_image_class_vectors(root_dir):
    class_names = ["class1", "class2", "class3", "class4"]
    class_basis_map = {
        class_name: i for i, class_name in enumerate(class_names)
    }

    image_vectors = {}

    sample_paths = sorted(
        [
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ]
    )

    for sample_id in sample_paths:
        sample_path = os.path.join(root_dir, sample_id)
        file_paths = os.listdir(sample_path)

        image_vectors[sample_id] = [0, 0, 0, 0]

        class_paths = [
            os.path.join(sample_path, f)
            for f in file_paths
            if f.startswith("class")
        ]

        for class_path in class_paths:
            class_name = os.path.basename(class_path).split(".")[0]

            mask_img = rle_read_maskfile(class_path)
            instance_ids = rle_maskobj_get_instances(mask_img)

            if len(instance_ids) > 0:
                class_idx = class_basis_map[class_name]
                image_vectors[sample_id][class_idx] = 1

    return [(sample_id, vector) for sample_id, vector in image_vectors.items()]

def get_balanced_dataset_split(dataset_path, split_percent = 0.2):
    image_vectors = get_image_class_vectors(str(dataset_path))
    sums = sum_dataset(image_vectors)
    even_split_goal = []
    for i in range(4):
        even_split_goal.append(sums[i] * split_percent)
    
    split_image_count = len(training_dataset) * split_percent
    target_counts = [math.floor(t) for t in even_split_goal]
    print(
        f"Want {math.ceil(split_image_count)} validation images having "
        f"[class1,class2,class3,class4] totals exceeding {target_counts}"
    )

    itmes_A = image_vectors
    print(len(itmes_A))
    items_B = []
    for i in range(10):
        next_choice = itmes_A.pop(random.randint(0,len(itmes_A)-1))
        items_B.append(next_choice)
        for x in range(4):
            sums[x] -= next_choice[1][x]
    
    valid_sum = sum_dataset(items_B)
    
    progress = get_progress(valid_sum, even_split_goal)
    remaining = split_image_count - 10
    while min(progress) < 1.0 or remaining > 0:
        direction = progress.index(min(progress))
        
        itmes_A = sorted(itmes_A, key=lambda x: x[1][direction], reverse=True)
        next_choice = itmes_A.pop(random.randint(0,sums[direction]-1))
        items_B.append(next_choice)
        for x in range(4):
            sums[x] -= next_choice[1][x]
            valid_sum[x] += next_choice[1][x]
        remaining -= 1
        progress = get_progress(valid_sum, even_split_goal)

    print("Random Dataset Split Totals:")
    print(f"\tvalidation dataset {len(items_B)} items, class totals: {sum_dataset(items_B)}")
    print(f"\ttraining dataset {len(itmes_A)} items, class totals: {sum_dataset(itmes_A)}")

    return items_B

def sanity_test_dataset(dataset):

    for i in range(20):
        image, target = dataset[i]

        print(
            i,
            image.shape,
            target["masks"].shape,
            target["boxes"].shape,
            target["labels"].shape,
            target["sample_id"],
            target["crop_box"].tolist(),
        )

        assert image.dtype == torch.float32
        assert image.ndim == 3
        assert target["boxes"].dtype == torch.float32
        assert target["labels"].dtype == torch.int64
        assert target["masks"].dtype == torch.uint8

        if target["boxes"].shape[0] > 0:
            boxes = target["boxes"]
            assert torch.all(boxes[:, 2] > boxes[:, 0])
            assert torch.all(boxes[:, 3] > boxes[:, 1])
            assert target["masks"].shape[0] == target["labels"].shape[0]
            assert target["masks"].shape[0] == target["boxes"].shape[0]


def binary_mask_to_coco_rle(mask):
    """
    mask: torch.Tensor or np.ndarray [H, W], bool/uint8
    returns COCO compressed RLE
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = mask.astype(np.uint8)
    mask = np.asfortranarray(mask)

    rle = mask_utils.encode(mask)

    # pycocotools returns bytes; JSON-style COCO results need string counts.
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def build_coco_gt_from_targets(all_targets, num_classes=5):
    """
    Build a COCO ground-truth object from torchvision-style targets.

    all_targets: list of target dicts, one per validation image/crop.
    """
    coco_gt_dict = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id, "name": f"class{class_id}"}
            for class_id in range(1, num_classes)
        ],
    }

    ann_id = 1

    for image_id, target in enumerate(all_targets):
        masks = target["masks"].cpu()
        labels = target["labels"].cpu()
        boxes = target["boxes"].cpu()
        areas = target["area"].cpu()

        height, width = target["orig_size"].tolist()

        coco_gt_dict["images"].append(
            {
                "id": image_id,
                "width": int(width),
                "height": int(height),
                "file_name": str(target["sample_id"]),
            }
        )

        for obj_idx in range(masks.shape[0]):
            mask = masks[obj_idx].to(torch.bool)
            rle = binary_mask_to_coco_rle(mask)

            x1, y1, x2, y2 = boxes[obj_idx].tolist()
            bbox = [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ]

            coco_gt_dict["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(labels[obj_idx]),
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": float(areas[obj_idx]),
                    "iscrowd": 0,
                }
            )

            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    return coco_gt

@torch.inference_mode()
def evaluate_coco_ap50(
    model,
    data_loader,
    device,
    num_classes=5,
    mask_threshold=0.5,
    score_threshold=0.05,
    max_detections_per_image=300,
):
    """
    Official pycocotools COCOeval for instance segmentation AP50.

    Model inference runs on GPU.
    COCO RLE encoding + COCOeval run on CPU.
    """
    model.eval()

    all_targets = []
    coco_results = []

    image_id = 0

    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            # Store GT target for COCO GT construction.
            all_targets.append(target)

            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()
            boxes = output["boxes"].detach().cpu()
            masks = output["masks"].detach().cpu()[:, 0] >= mask_threshold

            keep = scores >= score_threshold

            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            masks = masks[keep]

            if scores.numel() > max_detections_per_image:
                order = torch.argsort(scores, descending=True)
                order = order[:max_detections_per_image]

                scores = scores[order]
                labels = labels[order]
                boxes = boxes[order]
                masks = masks[order]

            for pred_idx in range(scores.shape[0]):
                mask = masks[pred_idx]
                rle = binary_mask_to_coco_rle(mask)

                x1, y1, x2, y2 = boxes[pred_idx].tolist()
                bbox = [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                ]

                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(labels[pred_idx]),
                        "segmentation": rle,
                        "bbox": bbox,
                        "score": float(scores[pred_idx]),
                    }
                )

            image_id += 1

    coco_gt = build_coco_gt_from_targets(
        all_targets,
        num_classes=num_classes,
    )

    if len(coco_results) == 0:
        print("No predictions survived score threshold.")
        return 0.0, None

    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")

    # AP50 only.
    coco_eval.params.iouThrs = np.array([0.50])

    # Important for crowded cell images.
    coco_eval.params.maxDets = [1, 10, max_detections_per_image]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # With iouThrs=[0.50], stats[0] is AP averaged over the available IoU
    # thresholds, meaning AP50 here.
    ap50 = float(coco_eval.stats[0])

    print(f"Official COCOeval segm AP50: {ap50:.4f}")

    return ap50, coco_eval

if __name__=='__main__':
    args = parse_cmd()

    device, device_string = init_device()
    print(f"Using {device_string} backend ")
    if args.data_path == None:
        print("No data path specified, exiting")
        quit()
    data_path = Path("/".join([args.data_path, 'train']))
    print(f"training with {args.training_epochs} epochs")
    print(f"training with crop size of {args.crop_size}^2 px ")

    training_dataset = ImageDataset(
        str(data_path),
        crop_size=args.crop_size,
        random_crop=True,
        crop_trials=10,
        min_instances_in_crop=1,
    )
    validation_items = get_balanced_dataset_split(data_path, args.validation_ratio)
    validation_dataset = training_dataset.split(validation_items)
    validation_dataset.random_crop = False
    validation_dataset.crop_trials = 1
    validation_dataset.min_instances_in_crop = 0

    print("creating model")
    model = get_instance_segmenter(5, 512)
    model.to(device)

    print("creating optimizer")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )

    print("creating training loader")
    train_loader = DataLoader(
        training_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print("creating validation loader")
    val_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print("beginning training")
    for epoch in range(args.training_epochs):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        print(f"epoch={epoch}, train_loss={train_loss:.4f}")

        val_ap50, coco_eval = evaluate_coco_ap50(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=5,
            mask_threshold=0.5,
            score_threshold=0.05,
            max_detections_per_image=300,
        )

        print(f"epoch={epoch}, official_val_AP50={val_ap50:.4f}")