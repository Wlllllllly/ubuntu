import sys
sys.path.append('.')
sys.path.append('..')
import math
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from tqdm import tqdm
from monai.transforms import AsDiscrete
from dataset import get_loader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.networks.nets import SegResNet
from icecream import ic
from monai.losses import DiceLoss,DiceCELoss


IMG_SIZE=(96,96,96)
TRAINING_EPOCH=100
NUM_CLASSES=5
IN_CHANNELS=1
OUT_CHANNELS=5
BCE_WEIGHTS=[0.004, 0.996]
EVAL_NUM=10
NUM_SAMPLES=1
DATAROOT='/mnt/nas/CVAI_WLY/CVAI_350_sclices/CVAI_V5'
SAVE_WEIGHT='/home/cvlab4090wly/HD/Segmodels/Segresnet/checkpoints'
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
    model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            dropout_prob=0.2,
        ).to(device)
# criterion = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    # loss_function = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS)).cuda()
    # loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    loss_function = DiceCELoss(
        to_onehot_y=False, softmax=True, squared_pred=True, smooth_nr=1e-6, smooth_dr=0
    )
    optimizer = Adam(params=model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    min_valid_loss = math.inf

    for epoch in range(TRAINING_EPOCH):
        train_loss = 0.0
        model=model.cuda()
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_dataloader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            # print(x.shape)
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                loss = loss_function(logit_map, y)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_iterator.set_description(  # noqa: B038
                f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
            )
            torch.save(model.state_dict(), os.path.join(SAVE_WEIGHT, f"epocd_{step}_metric_model.pth"))
            # if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            #     epoch_iterator_val = tqdm(val_dataloader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            #     model.eval()
            #     with torch.no_grad():
            #         for batch in epoch_iterator_val:
            #             val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            #             with torch.cuda.amp.autocast():
            #                 val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            #             val_labels_list = decollate_batch(val_labels)
            #             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            #             val_outputs_list = decollate_batch(val_outputs)
            #             val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            #             dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            #             epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
            #         dice_val = dice_metric.aggregate().item()
            #         dice_metric.reset()
            #         epoch_loss /= step
            #         epoch_loss_values.append(epoch_loss)
            #         metric_values.append(dice_val)
                    
            #         if dice_val > dice_val_best:
            #             dice_val_best = dice_val
            #             global_step_best = global_step
            #             torch.save(model.state_dict(), os.path.join(SAVE_WEIGHT, f"best_metric_model.pth"))
            #             print(
            #                 "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
            #             )
            #         else:
            #             print(
            #                 "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
            #                     dice_val_best, dice_val
            #                 )
            #             )
                
            
            global_step += 1


