import os

import torch


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    return model_state_dict, optimizer_state_dict, epoch, train_loss, val_loss


def create_checkpoint(model, optimizer, avg_train_loss_epoch, val_loss, logger, artifact_path, epoch, args):
    checkpoint_path = os.path.join(artifact_path, f"checkpoints/{epoch}")
    os.makedirs(checkpoint_path, exist_ok=True)
    model_path = os.path.join(checkpoint_path, f"{args.enc_model}.pt")
    log_message = f"Saving model to {model_path} normally."
    state_dict = model.state_dict()
    if torch.cuda.device_count() > 1:
        log_message = f"Saving checkpoint to {model_path} using .module."
        state_dict = model.module.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss_epoch,
        'val_loss': val_loss
    }
    logger.info(log_message)
    torch.save(checkpoint, model_path)


# unused currently
def save_model(model, logger, artifact_path, epoch, args):
    # model_name = f"{epoch}/{os.path.basename(args.model_val_path)}"
    epoch_path = os.path.join(artifact_path, str(epoch))
    os.makedirs(epoch_path, exist_ok=True)
    model_path = os.path.join(epoch_path, f"{args.enc_model}.pt")
    if torch.cuda.device_count() > 1:
        logger.info(f"Saving model to {model_path} using .module.")
        torch.save(model.module.state_dict(), model_path)
        # pytorch.log_model(model, model_name)
    else:
        logger.info(f"Saving model to {model_path} normally.")
        torch.save(model.state_dict(), model_path)
        # pytorch.log_model(model, model_name)
