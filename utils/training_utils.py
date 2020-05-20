import os

import torch

from dataloader import CustomDataset, SaliconDataset


def get_data_loaders(dataset_dir, use_custom_loader, batch_size, no_workers):
    train_img_dir = os.path.join(dataset_dir, "images/train/")
    train_gt_dir = os.path.join(dataset_dir, "maps/train/")  # black white
    train_fix_dir = os.path.join(dataset_dir, "fixations/train/")  # color with maps overlayed
    val_img_dir = os.path.join(dataset_dir, "images/val/")
    val_gt_dir = os.path.join(dataset_dir, "maps/val/")
    val_fix_dir = os.path.join(dataset_dir, "fixations/val/")
    train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
    val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
    if use_custom_loader:
        train_dataset = CustomDataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids)
        val_dataset = CustomDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)
    else:
        train_dataset = SaliconDataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids)
        val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=no_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=no_workers)
    return train_loader, val_loader
