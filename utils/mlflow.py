import sys
import time

import mlflow


def setup_mlflow_experiment(args):
    run_id = str(time.time()).split('.')[0]
    if args.create_experiment and not args.experiment_name:
        print("Experiment name not provided. Exiting.")
        sys.exit(1)
    if not args.create_experiment and args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        experiment_id = experiment.experiment_id
    if args.create_experiment and args.experiment_name:
        experiment_id = mlflow.create_experiment(args.experiment_name)
    return experiment_id, run_id

def log_val_metrics(cc_loss, epoch, kldiv_loss, nss_loss, sim_loss, execution_time_min):
    metrics = {
        "val--avg--cc_loss": cc_loss.avg,
        "val--avg--kldiv": kldiv_loss.avg,
        "val--avg--nss_loss": nss_loss.avg,
        "val--avg--sim_loss": sim_loss.avg,
        "val--time": execution_time_min,
        "epoch": epoch,
    }
    mlflow.log_metrics(metrics, step=epoch)


def log_training_params(device, loss_type_str, args):
    training_params = {
        "device": device,
        "loss_type": loss_type_str,
        "no_epochs": args.no_epochs,
        "lr": args.lr,
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
    mlflow.log_params(training_params)

