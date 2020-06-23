import argparse
import os
import sys
import time
import matplotlib
import torch
import torch.nn as nn
import mlflow
import logging
import optuna

from model import DECONV_LAYERS
from simple_net.loss import *
from simple_net.utils import blur, AverageMeter
from checkpoint_utils import load_checkpoint, create_checkpoint
from training_utils import get_data_loaders
from utils.mlflow_utils import log_val_metrics, log_training_params, setup_mlflow_experiment, get_artifact_path, \
    get_log_path

matplotlib.use('Agg')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default="", type=str)
    parser.add_argument('--custom_loader', default=True, type=bool)
    parser.add_argument('--checkpoint_path', default="", type=str)
    parser.add_argument('--fine_tune', default=False, type=bool)
    parser.add_argument('--fine_tune_override_layers', default=False, type=bool)
    parser.add_argument('--no_epochs', default=40, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
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
    parser.add_argument('--pretrained_model_path',
                        default="/home/steffi/dev/CV2/saliency/saved_models/salicon_densenet.pt", type=str)

    return parser.parse_args()


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


def loss_func(pred_map, gt, fixations, loss_type, args):
    loss = torch.FloatTensor([0.0]).cuda()
    criterion = nn.L1Loss()
    if loss_type == "kldiv":
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if loss_type == "cc":
        loss += args.cc_coeff * cc(pred_map, gt)
    if loss_type == "nss":
        loss += args.nss_coeff * nss(pred_map, fixations)
    if loss_type == "l1":
        loss += args.l1_coeff * criterion(pred_map, gt)
    if loss_type == "sim":
        loss += args.sim_coeff * similarity(pred_map, gt)
    return loss


def train(model, optimizer, loader, epoch, device,
          loss_type, args, log_file_path, logger):
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
        loss = loss_func(pred_map, gt, fixations, loss_type, args)
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


def validate(model, loader, epoch, device, args, logger, log_file_path):
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

    return cc_loss.avg, kldiv_loss.avg


def get_suggested_params(trial, logger):
    from pprint import pformat
    sugg_lr = trial.suggest_float("lr", 1e-4, 1e-4, log=True)
    sugg_dropout = trial.suggest_float("dropout", 0.0, 0.0)
    sugg_optimizer = trial.suggest_categorical("optim", ["Adam"])
    sugg_loss_type = trial.suggest_categorical("loss_type", ["kldiv"])
    sugg_finetune_layers = []
    if args.fine_tune:
        sugg_finetune_layers = trial.suggest_categorical("finetune_layers", [["deconv_layer5"],
                                                                             ["deconv_layer5", "deconv_layer4"],
                                                                             ["deconv_layer5", "deconv_layer4", "deconv_layer3"]])
    logger.info(f"Trial parameters: {pformat([trial.params])}")
    return sugg_lr, sugg_dropout, sugg_optimizer, sugg_loss_type, sugg_finetune_layers


def create_model(args, device, sugg_dropout, sugg_finetune_layers, logger):
    from simple_net.model import DenseModel
    model = DenseModel(dropout=sugg_dropout)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.pretrained_model_path))

    if args.fine_tune:
        layers = sugg_finetune_layers
        logger.info(f"Fine-tuning layers: {layers}")
        frozen = []
        for name, param in model.named_parameters():
            if not any(layer in name for layer in layers):
                frozen.append(name)
                param.requires_grad = False
        logger.info(f"Frozen model weights: {frozen}")

        # reset fine-tuning layers => reinit weights
        if args.fine_tune_override_layers:
            logger.info(f"Reinitializing layers: {layers}")
            for layer in layers:
                model_layer = eval(f"model.module.{layer}")
                model_layer = DECONV_LAYERS[layer]
    model = model.to(device)
    return model


def objective(trial, experiment, args=None):
    print("Training existing Salicon DenseNet Model")
    run_name = str(time.time()).split('.')[0]

    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # mlflow run infos & paths
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id
        artifact_path = get_artifact_path(active_run)

        # logging
        log_path = get_log_path(experiment)
        log_file_path = os.path.join(log_path, "train.log")
        logger = setup_logging(log_file_path)
        logger.info(f"Starting run {run_id} of experiment {experiment.experiment_id}.")
        # create parameters using optuna
        sugg_lr, sugg_dropout, sugg_optimizer, sugg_loss_type, sugg_finetune_layers = get_suggested_params(trial, logger)
        mlflow.log_params(trial.params)

        # create network model
        model = create_model(args, device, sugg_dropout, sugg_finetune_layers, logger)

        # log training params to mlflow
        loss_type = sugg_loss_type
        log_training_params(logger, device, loss_type, args)

        # dataset loaders
        train_loader, val_loader = get_data_loaders(args.dataset_dir, args.custom_loader, args.batch_size, args.no_workers)

        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        trainable_params = sum([np.prod(p.size()) for p in params])
        logging.info(f"Training {trainable_params} model parameters.")

        if sugg_optimizer == "Adam":
            optimizer = torch.optim.Adam(params, lr=sugg_lr, weight_decay=args.weight_decay)
        if sugg_optimizer == "Adagrad":
            optimizer = torch.optim.Adagrad(params, lr=sugg_lr, weight_decay=args.weight_decay)
        if sugg_optimizer == "SGD":
            optimizer = torch.optim.SGD(params, lr=sugg_lr, momentum=0.9, weight_decay=args.weight_decay)
        if args.lr_sched:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
            logging.info(f"Using LR Scheduler")

        start_epoch = 0
        # load checkpoint
        if len(args.checkpoint_path) > 0:
            model_state_dict, optimizer_state_dict, start_epoch, train_loss, val_loss = load_checkpoint(args.checkpoint_path)
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

        for epoch in range(start_epoch, args.no_epochs):
            try:
                loss = train(model, optimizer, train_loader, epoch, device,
                             loss_type, args, log_file_path, logger)

                with torch.no_grad():
                    cc_loss, kldiv_loss = validate(model, val_loader, epoch, device, args, logger, log_file_path)
                    logger.info(f"cc_loss avg: {cc_loss}")
                    logger.info(f"kldiv_loss avg: {kldiv_loss}")
                    total_loss = ((args.kldiv_coeff * kldiv_loss + args.cc_coeff * cc_loss) + 0.5) / 2
                    logger.info(f"Total loss: {total_loss}")
                    if epoch == 0:
                        best_loss = total_loss
                    if best_loss <= total_loss:
                        best_loss = total_loss
                        logger.info(f"Best combined loss updated({args.kldiv_coeff} * kldiv + "
                                    f"{args.cc_coeff} * cc_loss): {best_loss}")

                # report intermediate cc_loss
                trial.report(total_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                create_checkpoint(model, optimizer, loss, cc_loss, logger, artifact_path, epoch, args)

                logger.info("")
                mlflow.log_artifact(log_file_path)

                if args.lr_sched:
                    scheduler.step()
            except KeyboardInterrupt:
                logger.error(f"Interrupted trial manually.")
                raise optuna.exceptions.TrialPruned()
    # return best validation loss of model after X epochs
    return total_loss


if __name__ == "__main__":
    args = parse_arguments()

    # create mlflow experiment
    experiment = setup_mlflow_experiment(args)

    study = optuna.create_study(study_name=args.experiment_name, direction="minimize")
    study.optimize(lambda trial: objective(trial, experiment, args), n_trials=3)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
