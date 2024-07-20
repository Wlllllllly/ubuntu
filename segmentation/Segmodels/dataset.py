from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    RandScaleIntensityd,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)
import sys
from copy import copy, deepcopy
import h5py, os
import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import pandas as pd

sys.path.append("..") 

from torch.utils.data import Subset
import nrrd
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()
import SimpleITK as sitk
from monai.visualize import matshow3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from icecream import ic



class LoadImaged_BodyMap(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        # print("d:::",d)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            if key=="image":#self._loader.image_only:
            # if True:
                data = self._loader(d[key], reader)
                d[key] = data[0]
                d[meta_key] = data[1]
            else:
                d[key] = d['label']
            
        d['image']=np.array(d['image'])
        d['label']= self.label_transfer(d['label'], d['image'].shape)

        # print("d:::",d)
        
        return d
    
    
    def label_transfer(self, lbl_dir, shape):
        # print(f"lbl_dir:{lbl_dir}")
        # print(f"shape:{shape}")
        # organ_lbl = np.zeros([3,shape[0], shape[1]])
        # organ_lbl = np.zeros(3,96,96,96)
        organ_lbl = np.zeros([5,shape[0], shape[1], shape[2]])

        # if os.path.exists(lbl_dir + 'foreground_mask' + '.png'):
        #     mask_data=Image.open(lbl_dir + 'foreground_mask' + '.png')
        #     mask_data=np.array(mask_data)
            
            # organ_lbl[0][mask_data > 0] = 0
            # organ_lbl[1][mask_data > 0] = 0
            # organ_lbl[2][mask_data > 0] = 1
            # print("foreground")
            # mata_infomation='foreground'

        if os.path.exists(lbl_dir + 'heart' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'heart' + '.nii.gz')
            array=np.array(array)
            organ_lbl[0][array > 0] = 1
            # print("heart")
            # mata_infomation='heart'

        if os.path.exists(lbl_dir + 'desc1' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'desc1' + '.nii.gz')
            array=np.array(array)
            organ_lbl[1][array > 0] = 1
            # print("desc1")
            # mata_infomation='desc1'
        if os.path.exists(lbl_dir + 'desc2' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'desc2' + '.nii.gz')
            array=np.array(array)
            organ_lbl[2][array > 0] = 1
            # print("desc2")
            # mata_infomation='desc2'

        if os.path.exists(lbl_dir + 'asc' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'asc' + '.nii.gz')
            array=np.array(array)
            organ_lbl[3][array > 0] = 1
            # print("asc")
            # mata_infomation='asc'

        if os.path.exists(lbl_dir + 'arch' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'arch' + '.nii.gz')
        
            array=np.array(array)
            organ_lbl[4][array > 0] = 1
            # print("arch")

            # mata_infomation='arch'

        return organ_lbl

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)
    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        # print('file_name', d['name'])
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]

        return d
    
def get_loader(dataroot):
    train_transforms = Compose(
        [
            LoadImaged_BodyMap(keys=["image","label"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1, 1, 1),
                mode=("bilinear", "nearest"),
            ), 
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            # EnsureChannelFirstd(keys=["image", "label"]),
            # Resized(keys=["image", "label"],spatial_size=(128, 128, 128)),
            # RandRotate90d(
            #     keys=["image", "label"],
            #     prob=0.10,
            #     max_k=3,
            # ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms=Compose(
                [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0, b_max=1, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
      
    train_img=[]
    train_lbl=[]
    train_name=[]
    file_path = os.path.join(dataroot, "heart+aorta影像及標註名單_train.csv")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, try 'big5' encoding
        df = pd.read_csv(file_path, encoding='big5')
    
    df=np.array(df)
    for i in range(len(df)):
        name = df[i,0]#line.strip().split('\t')[0]
        # img_name_seg=str(name.split('-')[0])+'-'+str(name.split('-')[1])+'-'+str(name.split('-')[2])+'-seg'
        train_img.append(os.path.join(dataroot, name +'/'+name+ '.nii.gz'))
        train_lbl.append(os.path.join(dataroot, name + '/segmentations/'))
        train_name.append(name)

    data_dicts_train = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(train_img, train_lbl, train_name)]

    train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
    train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=(train_sampler is None), num_workers=1, 
                                collate_fn=list_data_collate, sampler=train_sampler)
    print("len(train):",len(train_dataset))

    

    val_img=[]
    val_lbl=[]
    val_name=[]

    file_path = os.path.join(dataroot, "heart+aorta影像及標註名單_test.csv")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, try 'big5' encoding
        df = pd.read_csv(file_path, encoding='big5')
    
    df=np.array(df)
    for i in range(len(df)):
        name = df[i,0]#line.strip().split('\t')[0]
        # img_name_seg=str(name.split('-')[0])+'-'+str(name.split('-')[1])+'-'+str(name.split('-')[2])+'-seg'
        val_img.append(os.path.join(dataroot, name +'/'+name+ '.nii.gz'))
        val_lbl.append(os.path.join(dataroot, name + '/segmentations/'))
        val_name.append(name)

    data_dicts_val = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(val_img, val_lbl, val_name)]

    val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
    val_sampler = None
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=(val_sampler is None), num_workers=1, 
                                collate_fn=list_data_collate, sampler=val_sampler)
    print("len(val):",len(val_dataset))

    return train_loader, train_dataset,val_loader, val_dataset,   

    