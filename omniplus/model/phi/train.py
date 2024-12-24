import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from schedulefree import AdamWScheduleFree
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# optimizer


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_model(rank, world_size, model, train_dataset, val_dataset, n_epochs, config):
    setup(rank, world_size)

    # Initialize wandb only on main process
    if rank == 0:
        wandb.init(project="omni-plus", config=config)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Initialize optimizer
    optimizer = AdamWScheduleFree(model.parameters(), lr=config["learning_rate"])
    optimizer.train()  # Set optimizer to train mode

    model.train()
    start_time = time.time()

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"] // world_size,  # Local batch size
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True)

    global_step = 0
    for epoch in range(n_epochs):
        train_sampler.set_epoch(epoch)
        losses = []
        for bi, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch['image'] = batch['image'].to(rank)

            logits, loss = model.forward(batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.data.item())
            global_step += 1

            # Log metrics to wandb from main process only
            if bi % 100 == 0 and rank == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(losses) / len(losses)
                wandb.log({
                    "train/loss": avg_loss,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                    "train/elapsed_time": elapsed
                })
                print(
                    f'Epoch [{epoch+1}/{n_epochs}] Step [{global_step}] Loss: {avg_loss:.4f} Time: {elapsed:.2f}s'
                )

            # Save model from main process only
            if global_step % 5000 == 0 and rank == 0:
                os.makedirs('models', exist_ok=True)
                checkpoint_path = os.path.join('models', f'vlm_model_step_{global_step}.pt')
                torch.save(
                    {
                        'model_state_dict': model.module.state_dict(),  # Save the underlying model
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch,
                    },
                    checkpoint_path)
                wandb.save(checkpoint_path)
                print(f'\nCheckpoint saved to {checkpoint_path}')

        if rank == 0:
            val_metrics = val_step(rank, val_dataset, config, model, optimizer)
            wandb.log({"val/metrics": val_metrics, "val/epoch": epoch + 1})
            print(val_metrics)

    if rank == 0:
        wandb.finish()

    cleanup()


def val_step(rank, val_dataset, config, model, optimizer):
    model.eval()
    optimizer.eval()  # Set optimizer to eval mode for validation

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        sampler=DistributedSampler(val_dataset, num_replicas=torch.cuda.device_count(), rank=rank),
        num_workers=4,
        pin_memory=True)

    val_metrics = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch['image'] = batch['image'].to(rank)
            logits, _ = model.forward(batch)
            # Calculate validation metrics here

    model.train()
    optimizer.train()  # Set optimizer back to train mode

    return val_metrics


if __name__ == '__main__':
    """Train a model with multi-GPU support."""
    world_size = torch.cuda.device_count()

    # Model
    from vlm import build_vlm
    model = build_vlm()

    # Training configuration
    config = {
        "learning_rate": 3e-4,
        "epochs": 2,
        "batch_size": 1 * world_size,  # Global batch size
        "model_type": "vlm",
        "optimizer": "AdamWScheduleFree",
        "num_gpus": world_size
    }

    # Data
    from dataset import get_coco_dataset

    train_dataset = get_coco_dataset(mode='train')
    val_dataset = get_coco_dataset(mode='val')

    print(f"Training with {world_size} GPUs")
    print(train_dataset, val_dataset)
    n_epochs = config["epochs"]

    # Launch training processes
    mp.spawn(
        train_model,
        args=(world_size, model, train_dataset, val_dataset, n_epochs, config),
        nprocs=world_size,
        join=True)

    # Save final model
    if torch.distributed.get_rank() == 0:
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join('models', f'vlm_model_{timestamp}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_epochs': n_epochs,
        }, model_path)
        wandb.save(model_path)
        print(f'\nModel saved to {model_path}')
