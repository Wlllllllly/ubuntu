import sys
sys.path.append('.')
sys.path.append('..')
import math
import torch
from torch.nn import CrossEntropyLoss
# from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
import os
from tqdm import tqdm
from monai.transforms import AsDiscrete
from model.unet3d import UNet3D
from monai.losses import DiceCELoss
from dataset import get_loader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import decollate_batch
import numpy as np

from icecream import ic
from monai.losses import DiceLoss



IMG_SIZE=(512,96,96)
TRAINING_EPOCH=100
NUM_CLASSES=5
IN_CHANNELS=1
OUT_CHANNELS=5
BCE_WEIGHTS=[1,1,1,1,1]
EVAL_NUM=10
NUM_SAMPLES=1
DATAROOT='/mnt/nas/CVAI_WLY/CVAI_350_sclices/CVAI_V5'
SAVE_WEIGHT='/home/cvlab4090wly/HD/Segmodels/3dUnet/checkpoints'
BACKGROUND_AS_CLASS=False



if __name__=='__main__':
    # if BACKGROUND_AS_CLASS: NUM_CLASSES += 1
    num_samples = NUM_SAMPLES
    max_iterations = TRAINING_EPOCH
    eval_num = EVAL_NUM
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, train_dataset, val_dataloader, val_dataset = get_loader(DATAROOT)
    model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(device)
    # loss_function = DiceCELoss(
    #     to_onehot_y=False, softmax=True, squared_pred=True, smooth_nr=1e-6, smooth_dr=0
    # )
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=True, sigmoid=True)
    # loss_function = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS)).cuda()
    optimizer = Adam(params=model.parameters(),lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    min_valid_loss = math.inf

    for epoch in range(TRAINING_EPOCH):
        train_loss = 0.0
        model=model.to(device)
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_dataloader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].to(device), batch["label"].to(device))
            ic(x.shape)
            ic(y.shape)
            optimizer.zero_grad()
            # ic(np.umique)
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                loss = loss_function(logit_map, y)
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            epoch_loss += loss.item()
            # scaler.unscale_(optimizer)
            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad()
            epoch_iterator.set_description(  # noqa: B038
                f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
            )

            torch.save(model.state_dict(), os.path.join(SAVE_WEIGHT, f"epocd_{step}_3dUnet_loss{loss:2.3}.pth"))
            # if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            #     epoch_iterator_val = tqdm(val_dataloader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            #     model.eval()
            #     with torch.no_grad():
            #         for data in val_dataloader:
            #             image, ground_truth = data['image'].cuda(), data['label'].cuda()
            #             target = model(image)
            #             loss = loss_function(target,ground_truth)
            #             valid_loss = loss.item()
            #         if min_valid_loss > valid_loss:
            #             min_valid_loss = valid_loss
            #             global_step_best = global_step
            #             torch.save(model.state_dict(), os.path.join(SAVE_WEIGHT, f"best_metric_model.pth"))
            #             print(
            #                 "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, valid_loss)
            #             )
            #         else:
            #             print(
            #                 "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
            #                     min_valid_loss, valid_loss
            #                 )
            #             )
                
                    
            global_step += 1


