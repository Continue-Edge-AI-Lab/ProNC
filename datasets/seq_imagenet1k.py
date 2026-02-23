import io
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from utils.conf import base_path
from utils.prompt_templates import templates

class ParquetImageNetDataset(Dataset):
    """ImageNet-1K dataset loaded from Parquet files with task filtering"""
    
    def __init__(self, root: str, train: bool = True, 
                 transform: Optional[transforms.Compose] = None,
                 task_classes: Tuple[int, int] = (0, 1000)) -> None:
        """
        Args:
            root: Path to root directory containing train/val folders
            train: Whether to load training or validation data
            transform: Optional transforms to apply
            task_classes: Inclusive range of classes (start, end) to load
        """
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.task_classes = task_classes
        
        # Validate directory structure
        self.split_dir = self.root / ("train" if train else "test")
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
            
        # Load and validate Parquet files
        self.files = list(self.split_dir.glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No Parquet files found in {self.split_dir}")
        
        # Load filtered dataset
        self.dataset = self._load_filtered_data()
        
        # Convert to numpy arrays - with proper handling of struct scalars
        self.data = []
        self.targets = self.dataset['label'].to_numpy().astype(np.int64)
        
        # Process images one by one to avoid memory issues
        for img in self.dataset['image']:
            array = self._bytes_to_array(img)
            if array is not None:
                self.data.append(array)
        
        # Stack the processed images
        self.data = np.stack(self.data)
        
        # Ensure targets match the processed data length
        if len(self.data) != len(self.targets):
            self.targets = self.targets[:len(self.data)]
            
        self.classes = np.arange(1000)

    def _load_filtered_data(self) -> pa.Table:
        """Load data with task class filtering"""
        try:
            # Create dataset with class filter
            dataset = pq.ParquetDataset(
                self.files,
                filters=[
                    ('label', '>=', self.task_classes[0]),
                    ('label', '<', self.task_classes[1])
                ]
            )
            table = dataset.read()

            # Validate loaded data
            if len(table) == 0:
                available_labels = set()
                for f in self.files:
                    t = pq.read_table(f, columns=['label'])
                    available_labels.update(t['label'].unique().to_pylist())
                raise ValueError(
                    f"No data for classes {self.task_classes} in {self.split_dir}\n"
                    f"Available classes: {sorted(available_labels)}"
                )
                
            return table
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Parquet data from {self.split_dir}\n"
                f"Files attempted: {[f.name for f in self.files]}\n"
                f"Error: {str(e)}"
            ) from e

    def _bytes_to_array(self, img_data) -> np.ndarray:
        """Convert PyArrow data to RGB numpy array"""
        try:
            # Handle PyArrow StructScalar
            if hasattr(img_data, 'as_py'):
                img_data = img_data.as_py()
            
            # If it's a dictionary or similar structure, extract the image bytes
            if isinstance(img_data, dict) and 'bytes' in img_data:
                img_bytes = img_data['bytes']
            elif isinstance(img_data, dict) and 'data' in img_data:
                img_bytes = img_data['data']
            else:
                img_bytes = img_data
                
            # Ensure we have bytes
            if not isinstance(img_bytes, bytes):
                raise TypeError(f"Expected bytes, got {type(img_bytes)}: {img_bytes}")
                
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return np.array(img)
        except Exception as e:
            print(f"Warning: Failed to decode image: {str(e)}")
            print(f"Data type: {type(img_data)}, Value: {str(img_data)[:100]}...")
            return None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img = Image.fromarray(self.data[index])
        target = self.targets[index]
        
        original_img = img.copy()
        not_aug_img = transforms.ToTensor()(original_img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, target, not_aug_img

    def __len__(self) -> int:
        return len(self.data)

class SequentialImageNet1K(ContinualDataset):
    """Sequential ImageNet-1K Dataset with Parquet backend"""
    
    NAME = 'seq-imagenet1k'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 50
    SIZE = (64, 64)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])
        train_dataset = ParquetImageNetDataset(
            base_path() + 'IMG1K',
            train=True,
            transform=self.TRANSFORM,
            task_classes=((self.c_task+1)*self.N_CLASSES_PER_TASK,(self.c_task+2)*self.N_CLASSES_PER_TASK)
        )
        
        test_dataset = ParquetImageNetDataset(
            base_path() + 'IMG1K', 
            train=False,
            transform=test_transform,
            task_classes=((self.c_task+1)*self.N_CLASSES_PER_TASK,(self.c_task+2)*self.N_CLASSES_PER_TASK)
        )

        return store_masked_loaders(train_dataset, test_dataset, self)

    @staticmethod
    def get_backbone(backbone: str = "resnet50"):
        return backbone

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialImageNet1K.MEAN, SequentialImageNet1K.STD)

    @staticmethod
    def get_denormalization_transform():
        return transforms.Compose([
            transforms.Normalize((0., 0., 0.), (1/0.229, 1/0.224, 1/0.225)),
            transforms.Normalize((-0.485, -0.456, -0.406), (1., 1., 1.))
        ])

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialImageNet1K.TRANSFORM])

    def get_epochs(self, n_epochs: int = 90):
        return n_epochs

    def get_batch_size(self, batch_size: int = 256):
        return batch_size

    @staticmethod
    def get_prompt_templates():
        return templates['imagenet']