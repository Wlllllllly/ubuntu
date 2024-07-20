# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import numpy as np
import torch
# from trainer import dice
from dataset import get_loader
from model.unet3d import UNet3D
from monai.inferers import sliding_window_inference

DATAROOT='/mnt/nas/CVAI_WLY/CVAI_350_sclices/CVAI_V5'

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

def main():
    
    pretrained_pth='/home/cvlab4090wly/HD/Segmodels/Segresnet/checkpoints/epocd_143_metric_model.pth'
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, train_dataset, val_dataloader, val_dataset = get_loader(DATAROOT)

    model_dict = torch.load(pretrained_pth)
    model = UNet3D(in_channels=1 , num_classes= 5).to(device)
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_dataloader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            # print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.5)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            dice_list_sub = []
            for i in range(1, 6):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))


if __name__ == "__main__":
    main()