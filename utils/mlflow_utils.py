import sys
import time
import os
from urllib.parse import urlparse

import mlflow


def sync_neptune():
    os.system("neptune mlflow")


def setup_mlflow_experiment(args):
    experiment_name = args.experiment_name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)
    return experiment


def get_artifact_path(active_run):
    p = urlparse(active_run.info.artifact_uri)
    final_path = os.path.abspath(os.path.join(p.netloc, p.path))
    return final_path


def get_log_path(experiment):
    experiment_path = experiment.artifact_location.replace("file://", "")
    log_path = os.path.abspath(os.path.join(experiment_path, "logs"))
    os.makedirs(log_path, exist_ok=True)
    return log_path


def get_run_path(active_run):
    return os.path.abspath(os.path.join(get_artifact_path(active_run), os.pardir))


def log_val_metrics(cc_loss, epoch, kldiv_loss, nss_loss, sim_loss, execution_time_min):
    metrics = {
        "val--avg--cc_loss": cc_loss.avg.cpu().item(),
        "val--avg--kldiv": kldiv_loss.avg.cpu().item(),
        "val--avg--nss_loss": nss_loss.avg.cpu().item(),
        "val--avg--sim_loss": sim_loss.avg.cpu().item(),
        "val--time": execution_time_min,
        "epoch": epoch,
    }
    mlflow.log_metrics(metrics, step=epoch)


def log_training_params(logger, device, loss_type_str, args):
    training_params = {
        "device": device,
        "loss_type": loss_type_str,
        "no_epochs": args.no_epochs,
        "nss_emlnet": args.nss_emlnet,
        "nss_norm": args.nss_norm,
        "l1": args.l1,
        "lr_sched": args.lr_sched,
        "dilation": args.dilation,
        "enc_model": args.enc_model,
        "optim": args.optim,
        "load_weight": args.load_weight,
        "kldiv_coeff": args.kldiv_coeff,
        "step_size": args.step_size,
        "cc_coeff": args.cc_coeff,
        "sim_coeff": args.sim_coeff,
        "nss_coeff": args.nss_coeff,
        "nss_norm_coeff": args.nss_norm_coeff,
        "l1_coeff": args.l1_coeff,
        "train_enc": bool(args.train_enc),
        "dataset_dir": args.dataset_dir,
        "batch_size": args.batch_size,
        "log_interval": args.log_interval,
        "no_workers": args.no_workers,
        "model_val_path": args.model_val_path
    }
    logger.info(f"Training args: {args}")
    mlflow.log_params(training_params)

