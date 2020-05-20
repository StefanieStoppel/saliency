import argparse
import os
import sys
import time
import matplotlib
import torch
import torch.nn as nn
import mlflow
import logging

from torch.utils.data import DataLoader
from simple_net.dataloader import SaliconDataset, CustomDataset
from simple_net.loss import *
from simple_net.utils import blur, AverageMeter
from checkpoint_utils import load_checkpoint, create_checkpoint
from utils.mlflow_utils import log_val_metrics, log_training_params, setup_mlflow_experiment, get_artifact_path, \
    get_log_path

matplotlib.use('Agg')


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', default="", type=str)
parser.add_argument('--custom_loader', default=True, type=bool)
parser.add_argument('--checkpoint_path', default="", type=str)
parser.add_argument('--fine_tune', default=True, type=bool)
parser.add_argument('--no_epochs', default=40, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--kldiv', default=True, type=bool)
parser.add_argument('--cc', default=False, type=bool)
parser.add_argument('--nss', default=False, type=bool)
parser.add_argument('--sim', default=False, type=bool)
parser.add_argument('--nss_emlnet', default=False, type=bool)
parser.add_argument('--nss_norm', default=False, type=bool)
parser.add_argument('--l1', default=False, type=bool)
parser.add_argument('--lr_sched', default=False, type=bool)
parser.add_argument('--dilation', default=False, type=bool)
parser.add_argument('--enc_model', default="pnas", type=str)
parser.add_argument('--optim', default="Adam", type=str)

parser.add_argument('--load_weight', default=1, type=int)
parser.add_argument('--kldiv_coeff', default=1.0, type=float)
parser.add_argument('--step_size', default=5, type=int)
parser.add_argument('--cc_coeff', default=-1.0, type=float)
parser.add_argument('--sim_coeff', default=-1.0, type=float)
parser.add_argument('--nss_coeff', default=1.0, type=float)
parser.add_argument('--nss_emlnet_coeff', default=1.0, type=float)
parser.add_argument('--nss_norm_coeff', default=1.0, type=float)
parser.add_argument('--l1_coeff', default=1.0, type=float)
parser.add_argument('--train_enc', default=1, type=int)

parser.add_argument('--dataset_dir', default="/home/samyak/old_saliency/saliency/SALICON_NEW/", type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--log_interval', default=60, type=int)
parser.add_argument('--no_workers', default=4, type=int)
parser.add_argument('--model_val_path', default="model.pt", type=str)
parser.add_argument('--pretrained_model_path', default="/home/steffi/dev/CV2/saliency/saved_models/salicon_densenet.pt", type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.enc_model == "pnas":
    print("PNAS Model")
    from simple_net.model import PNASModel

    model = PNASModel(train_enc=bool(args.train_enc), load_weight=args.load_weight)

elif args.enc_model == "densenet":
    print("DenseNet Model")
    from simple_net.model import DenseModel

    model = DenseModel(train_enc=bool(args.train_enc), load_weight=args.load_weight)

elif args.enc_model == "salicon_densenet":
    print("Training existing Salicon DenseNet Model")
    from simple_net.model import DenseModel
    model = DenseModel()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.pretrained_model_path))
    if args.fine_tune:
        print("Finetuning only deconv_layer5.")
        for param in model.parameters():
            param.requires_grad = False
        model.module.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
    model = model.to(device)

elif args.enc_model == "resnet":
    print("ResNet Model")
    from simple_net.model import ResNetModel

    model = ResNetModel(train_enc=bool(args.train_enc), load_weight=args.load_weight)

elif args.enc_model == "vgg":
    print("VGG Model")
    from simple_net.model import VGGModel

    model = VGGModel(train_enc=bool(args.train_enc), load_weight=args.load_weight)

elif args.enc_model == "mobilenet":
    print("Mobile NetV2")
    from simple_net.model import MobileNetV2

    model = MobileNetV2(train_enc=bool(args.train_enc), load_weight=args.load_weight)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

train_img_dir = os.path.join(args.dataset_dir, "images/train/")
train_gt_dir = os.path.join(args.dataset_dir, "maps/train/")  # black white
train_fix_dir = os.path.join(args.dataset_dir, "fixations/train/")  # color with maps overlayed

val_img_dir = os.path.join(args.dataset_dir, "images/val/")
val_gt_dir = os.path.join(args.dataset_dir, "maps/val/")
val_fix_dir = os.path.join(args.dataset_dir, "fixations/val/")


train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]

if args.custom_loader:
    train_dataset = CustomDataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids)
    val_dataset = CustomDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)
else:
    train_dataset = SaliconDataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids)
    val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)


def setup_logging(log_file_path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    root_logger = logging.getLogger()
    return root_logger


def _get_loss_type_str(args):
    loss_type = ""
    if args.kldiv:
        loss_type = "kldiv"
    if args.cc:
        loss_type = "cc"
    if args.nss:
        loss_type = "nss"
    if args.l1:
        loss_type = "l1"
    if args.sim:
        loss_type = "sim"
    return loss_type


def loss_func(pred_map, gt, fixations, args):
    loss = torch.FloatTensor([0.0]).cuda()
    criterion = nn.L1Loss()
    if args.kldiv:
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if args.cc:
        loss += args.cc_coeff * cc(pred_map, gt)
    if args.nss:
        loss += args.nss_coeff * nss(pred_map, fixations)
    if args.l1:
        loss += args.l1_coeff * criterion(pred_map, gt)
    if args.sim:
        loss += args.sim_coeff * similarity(pred_map, gt)
    return loss


def train(model, optimizer, loader, epoch, device,
          loss_type, args, log_file_path, logger, artifact_path):
    model.train()
    tic = time.time()

    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()
        pred_map = model(img)
        assert pred_map.size() == gt.size()
        loss = loss_func(pred_map, gt, fixations, args)
        loss.backward()
        total_loss += loss.item()
        cur_loss += loss.item()

        optimizer.step()
        if idx % args.log_interval == (args.log_interval - 1):
            avg_loss = cur_loss / args.log_interval
            mlflow.log_metric(f"train--avg_batch_loss--{loss_type}", avg_loss)
            logger.info(
                '[{:2d}, {:5d}] train--avg_batch_loss--{} : {:.5f}, time:{:3f} minutes'.format(epoch, idx, loss_type, avg_loss,
                                                                                  (time.time() - tic) / 60))
            mlflow.log_artifact(log_file_path)
            cur_loss = 0.0
            sys.stdout.flush()

    avg_epoch_loss = total_loss / len(loader)
    mlflow.log_metric(f"train--avg_epoch_loss--{loss_type}", avg_epoch_loss)
    logger.info('[{:2d}, train] train--avg_epoch_loss--{} : {:.5f}'.format(epoch, loss_type, avg_epoch_loss))
    mlflow.log_artifact(log_file_path)
    sys.stdout.flush()

    return avg_epoch_loss


def validate(model, loader, epoch, device, args):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()

    for (img, gt, fixations) in loader:
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        pred_map = model(img)

        # Blurring
        blur_map = pred_map.cpu().squeeze(0).clone().numpy()
        blur_map = blur(blur_map).unsqueeze(0).to(device)

        cc_loss.update(cc(blur_map, gt))
        kldiv_loss.update(kldiv(blur_map, gt))
        nss_loss.update(nss(blur_map, gt))
        sim_loss.update(similarity(blur_map, gt))

    execution_time_min = (time.time() - tic) / 60
    log_val_metrics(cc_loss, epoch, kldiv_loss, nss_loss, sim_loss, execution_time_min)
    logger.info('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'
          .format(epoch,
                  cc_loss.avg,
                  kldiv_loss.avg,
                  nss_loss.avg,
                  sim_loss.avg,
                  execution_time_min))
    mlflow.log_artifact(log_file_path)
    sys.stdout.flush()

    return cc_loss.avg


# create mlflow experiment
experiment_id, run_name = setup_mlflow_experiment(args)

with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id
    artifact_path = get_artifact_path(active_run)

    log_path = get_log_path(active_run)
    log_file_path = os.path.join(log_path, "train.log")

    logger = setup_logging(log_file_path)
    logger.info(f"Starting run {run_id} of experiment {experiment_id}.")

    loss_type = _get_loss_type_str(args)
    log_training_params(device, loss_type, args)

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    trainable_params = sum([np.prod(p.size()) for p in params])
    logging.info(f"Training {trainable_params} model parameters.")

    if args.optim == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.optim == "Adagrad":
        optimizer = torch.optim.Adagrad(params, lr=args.lr)
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    if args.lr_sched:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    start_epoch = 0
    # load checkpoint
    if len(args.checkpoint_path) > 0:
        model_state_dict, optimizer_state_dict, start_epoch, train_loss, val_loss = load_checkpoint(args.checkpoint_path)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    for epoch in range(start_epoch, args.no_epochs):
        loss_type = _get_loss_type_str(args)

        loss = train(model, optimizer, train_loader, epoch, device,
                     loss_type, args, log_file_path, logger, artifact_path)

        with torch.no_grad():
            cc_loss = validate(model, val_loader, epoch, device, args)
            if epoch == 0:
                best_loss = cc_loss
            if best_loss <= cc_loss:
                best_loss = cc_loss

            create_checkpoint(model, optimizer, loss, cc_loss, logger, artifact_path, epoch, args)

            logger.info("")
            mlflow.log_artifact(log_file_path)

        if args.lr_sched:
            scheduler.step()
