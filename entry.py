import torch
import torch.nn as NN
import torch.nn.functional as func
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as tfunc2
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torchvision.models import resnet50
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

from utils import encode_mask as rle_encode, decode_maskobj as rle_decode_maskobj, read_maskfile as rle_read_maskfile, get_maskobj_instances as rle_maskobj_get_instances



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
        default=None,
        type=float,
        help="Specify the ratio of validation/training images"
    )

    return parser.parse_args()


class ImageDataset(Dataset):
    def __init__(self, root_dir=None, transforms=None):
        if root_dir != None:
            self.root_dir = root_dir
            self.transforms = transforms

            self.sample_paths = sorted([f for f in os.listdir(root_dir) if os.path.isdir(self.root_dir + '/' + f)])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.root_dir + "/" + self.sample_paths[idx]

        file_paths = [f for f in os.listdir(sample_path)]
        image_path = [sample_path + '/' + f for f in file_paths if f.startswith("image")][0]
        class_paths = [sample_path + '/' + f for f in file_paths if f.startswith("class")]

        sample_id = self.sample_paths[idx]
        
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

            if self.transforms is not None:
                image = self.transforms(img)
            else:
                image = img

        targets = []
        

        for instance_mask_path in class_paths:
            class_name = instance_mask_path.split('/')[-1].split('.')[0]
            if self.transforms is not None:
                mask_img = self.transforms(rle_read_maskfile(instance_mask_path))
            else:
                mask_img = rle_read_maskfile(instance_mask_path)

            mask_instances = rle_maskobj_get_instances(mask_img)
            target_class = {
                "class_id": class_name,
                "mask_img": mask_img,
                "instance_arr": mask_instances
            }
            targets.append(target_class)

        
        target = {
            "sample_id": sample_id,
            "orig_size": (orig_w, orig_h),
            "masks": targets
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

def collate_infer(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_dataset_statistics(dataset):
    print(len(dataset))
    max_width = 0
    max_height = 0
    max_instances = 0
    min_width = 10000
    min_height = 10000
    min_instances = 10000
    mean_width = 0
    mean_height = 0
    mean_instances = 0
    instance_counts = {'class1':0, 'class2':0, 'class3': 0, 'class4': 0}
    image_counts = {'class1':0, 'class2':0, 'class3': 0, 'class4': 0}
    class_basis_map = {'class1':0, 'class2':1, 'class3': 2, 'class4': 3}
    image_vectors = {}
    for image, target in dataset:
        width, height = target['orig_size']
        instances = 0
        image_vectors[target['sample_id']] = [0,0,0,0]
        for mask in target['masks']:
            mask_instances = len(mask['instance_arr'])
            image_vectors[target['sample_id']][class_basis_map[mask['class_id']]] = 1
            instance_counts[mask['class_id']] += mask_instances
            instances += mask_instances
            image_counts[mask['class_id']] += 1

        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
        if instances > max_instances:
            max_instances = instances

        if width < min_width:
            min_width = width
        if height < min_height:
            min_height = height
        if instances < min_instances:
            min_instances = instances

        mean_width += width
        mean_height += height
        mean_instances += instances

    mean_width /= len(dataset)
    mean_height /= len(dataset)
    mean_instances /= len(dataset)

    print("Dataset Statistics:")
    print(f"\tmax_width: {str(max_width)}") 
    print(f"\tmax_height: {str(max_height)}") 
    print(f"\tmax_instances: {str(max_instances)}") 
    print(f"\tmin_width: {str(min_width)}") 
    print(f"\tmin_height: {str(min_height)}") 
    print(f"\tmin_instances: {str(min_instances)}") 
    print(f"\tmean_width: {str(mean_width)}") 
    print(f"\tmean_height: {str(mean_height)}") 
    print(f"\tmean_instances: {str(mean_instances)}")
    print("\tClass Statistics: ")
    print(instance_counts)
    print(image_counts)

    return [(k,v) for k,v in image_vectors.items()]

def sum_dataset(dataset):
    sums = [0,0,0,0]
    for sample_name, l0_norm in dataset:
        for i in range(4):
            sums[i] += l0_norm[i]
    return sums

def get_progress(current, target):
    return [current[i] / target[i] for i in range(4)]



if __name__=='__main__':
    args = parse_cmd()

    device, device_string = init_device()
    print(f"Using {device_string} backend ")

    data_path = Path("/".join([args.data_path, 'train']))
    
    training_dataset = ImageDataset(str(data_path))

    
    
    image_vectors = get_dataset_statistics(training_dataset)
    sums = sum_dataset(image_vectors)
    even_split_goal = []
    for i in range(4):
        even_split_goal.append(sums[i] * args.validation_ratio)
    
    val_image_count = len(training_dataset) * args.validation_ratio
    print(f"Want { \
        math.ceil(val_image_count)} validation images having [class1,class2,class3,class4] totals exceeding {[math.floor(t) for t in even_split_goal]}")

    training_items = image_vectors
    validation_items = []
    for i in range(10):
        next_choice = training_items.pop(random.randint(0,len(training_items)))
        validation_items.append(next_choice)
        for x in range(4):
            sums[x] -= next_choice[1][x]
    
    valid_sum = sum_dataset(validation_items)
    
    progress = get_progress(valid_sum, even_split_goal)
    remaining = val_image_count - 10
    while min(progress) < 1.0 or remaining > 0:
        direction = progress.index(min(progress))
        
        training_items = sorted(training_items, key=lambda x: x[1][direction], reverse=True)
        next_choice = training_items.pop(random.randint(0,sums[direction]))
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

    print("Validation Dataset Statistics")
    get_dataset_statistics(validation_dataset)
    print("Training Dataset Statistics")
    get_dataset_statistics(training_dataset)

    print(len(training_dataset) + len(validation_dataset))