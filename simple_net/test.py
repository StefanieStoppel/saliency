import argparse
import sys
import time

import matplotlib

from checkpoint_utils import load_checkpoint

matplotlib.use('Agg')
from torch.utils.data import DataLoader
import torch.nn as nn
from simple_net.dataloader import TestLoader, SaliconDataset, CustomDataset
from simple_net.loss import *
from simple_net.utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--custom_loader', default=True, type=bool)
parser.add_argument('--model_val_path', default="./saved_models/salicon_pnas.pt", type=str)
parser.add_argument('--test_img_dir', default="./images/", type=str)
parser.add_argument('--no_workers', default=4, type=int)
parser.add_argument('--enc_model', default="pnas", type=str)
parser.add_argument('--results_dir', default="./results/", type=str)
parser.add_argument('--validate', default=0, type=int)
parser.add_argument('--save_results', default=1, type=int)
parser.add_argument('--dataset_dir', default="/home/samyak/old_saliency/saliency/SALICON_NEW/", type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.enc_model == "pnas":
    print("PNAS Model")
    from simple_net.model import PNASModel
    model = PNASModel()

elif args.enc_model == "densenet" or args.enc_model == "salicon_densenet":
    print("DenseNet Model")
    from simple_net.model import DenseModel
    model = DenseModel()

elif args.enc_model == "resnet":
    print("ResNet Model")
    from simple_net.model import ResNetModel
    model = ResNetModel()

elif args.enc_model == "vgg":
    print("VGG Model")
    from simple_net.model import VGGModel
    model = VGGModel()

elif args.enc_model == "mobilenet":
    print("Mobile NetV2")
    from simple_net.model import MobileNetV2
    model = MobileNetV2()

model = nn.DataParallel(model)

print("Loading model.")
model_state_dict, _, _, _, _ = load_checkpoint(args.model_val_path)
if isinstance(model_state_dict, torch.nn.DataParallel):
    state_dict = model_state_dict.module
model.load_state_dict(model_state_dict)

model = model.to(device)


def validate(model, loader, device, args):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()

    for (img, gt, fixations) in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)

        pred_map = model(img)

        # Blurring
        blur_map = pred_map.cpu().squeeze(0).clone().numpy()
        blur_map = blur(blur_map).unsqueeze(0).to(device)

        cc_loss.update(cc(blur_map, gt))
        kldiv_loss.update(kldiv(blur_map, gt))
        nss_loss.update(nss(blur_map, gt))
        sim_loss.update(similarity(blur_map, gt))

    print('CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(cc_loss.avg,
                                                                                               kldiv_loss.avg,
                                                                                               nss_loss.avg,
                                                                                               sim_loss.avg,
                                                                                               (time.time() - tic) / 60))
    sys.stdout.flush()

    return cc_loss.avg


if args.validate:
    val_img_dir = os.path.join(args.dataset_dir, "images/val/")
    val_gt_dir = os.path.join(args.dataset_dir, "maps/val/")
    val_fix_dir = os.path.join(args.dataset_dir, "fixations/val/")

    val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
    if args.custom_loader:
        val_dataset = CustomDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)
    else:
        val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)
    with torch.no_grad():
        validate(model, val_loader, device, args)
        
if args.save_results:
    test_img_dir = os.path.abspath(args.test_img_dir)
    test_img_ids = os.listdir(test_img_dir)
    test_dataset = TestLoader(test_img_dir, test_img_ids)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

    visualize_model(model, test_loader, device, args)
