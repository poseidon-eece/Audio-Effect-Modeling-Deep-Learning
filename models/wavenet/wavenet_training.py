import torch
import torch.optim as optim
import torch.utils.data
import time
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_logging import Logger


class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001,
                 weight_decay=0,
                 gradient_clipping=None,
                 logger=Logger(),
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 device=torch.device("cpu")):

        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scaler = GradScaler(device='cuda')
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.logger = logger
        self.logger.trainer = self
        self.validation_history = []
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.last_epoch_printed = -1
        self.device = device
        self.current_epoch = 0
        self.best_val_loss = float('inf')

    def train(self,
              batch_size=32,
              epochs=10,
              start_epoch=0,
              continue_training_at_step=0):

        # train_dataloader : shuffle=True, training samples
        # val_dataloader   : shuffle=False, val samples
        train_dataset = self.dataset
        train_dataset.train = True

        val_dataset = self.dataset.__class__(
            dataset_file=self.dataset.dataset_file,
            item_length=self.dataset._item_length,
            clean_dir='',           # δεν χρειάζεται — το .npz υπάρχει ήδη
            processed_dir='',       # δεν χρειάζεται — το .npz υπάρχει ήδη
            target_length=self.dataset.target_length,
            classes=self.dataset.classes,
            sampling_rate=self.dataset.sampling_rate,
            mono=self.dataset.mono,
            normalize=self.dataset.normalize,
            dtype=self.dataset.dtype,
            train=False,
            test_stride=self.dataset._test_stride,
            device=self.dataset.device
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size // 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        self.val_dataloader = val_dataloader

        print(f" Train samples : {len(train_dataset)}")
        print(f" Val samples   : {len(val_dataset)}")
        print(f" Batch size    : {batch_size}")
        print(f" Train batches : {len(train_dataloader)}")
        print(f" Val batches   : {len(val_dataloader)}")

        step = continue_training_at_step
        batch_print_limit = 3

        for current_epoch in range(start_epoch, epochs):
            print(f"\n─────────────────────────────")
            print(f"Epoch {current_epoch + 1}")
            print(f"─────────────────────────────")
            self.current_epoch = current_epoch
            self.model.train()
            tic = time.time()

            for (clean_condition, target) in iter(train_dataloader):
                start_batch_time = time.time()

                clean_condition = clean_condition.to(self.device).float()
                target = target.to(self.device).long()

                if step < batch_print_limit:
                    print(f"\nData Load Time: {time.time() - start_batch_time:.4f}s")

                start_forward_time = time.time()
                with autocast(device_type='cuda'):
                    output = self.model(clean_condition)

                    if step == 0:
                        print("🔹 INPUT SHAPE :", clean_condition.shape)
                        print("🔹 OUTPUT SHAPE:", output.shape)
                        print("🔹 TARGET SHAPE:", target.view(-1).shape)

                    loss = F.cross_entropy(output, target.view(-1))

                if step < batch_print_limit:
                    print(f"Forward Time: {time.time() - start_forward_time:.4f}s")

                self.optimizer.zero_grad()
                start_backward_time = time.time()

                self.scaler.scale(loss).backward()
                if self.clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if step < batch_print_limit:
                    print(f"Backward Time: {time.time() - start_backward_time:.4f}s")
                    print("-------------------------------------")

                loss_val = loss.item()
                step += 1

                # Progress bar
                total_steps = len(train_dataloader) * epochs
                current_progress = (step / total_steps) * 100
                print(f"\rTraining Progress: {current_progress:.2f}%", end="")

                if step == 100:
                    toc = time.time()
                    print(f"\nOne training step ≈ {(toc - tic) * 0.01:.4f}s")

                if step % self.snapshot_interval == 0:
                    if self.snapshot_path is not None:
                        time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                        torch.save({
                            'epoch': self.current_epoch,
                            'step': step,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'loss': loss_val
                        }, self.snapshot_path + '/' + self.snapshot_name + '_' + time_string + '.pth')

                self.logger.log(step, loss_val)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def validate(self):
        self.model.eval()
        total_loss = 0
        accurate_classifications = 0
        total_samples = 0

        with torch.no_grad():
            for (clean_condition, target) in iter(self.val_dataloader):
                clean_condition = clean_condition.to(self.device).float()
                target = target.to(self.device).long()

                with autocast(device_type='cuda'):
                    output = self.model(clean_condition)
                    target_reshaped = target.contiguous().view(-1)
                    loss = F.cross_entropy(output, target_reshaped)
                    total_loss += loss.item()

                predictions = torch.argmax(output, dim=1)
                accurate_classifications += (predictions == target_reshaped).sum().item()
                total_samples += target_reshaped.numel()

        avg_loss = total_loss / len(self.val_dataloader)
        avg_accuracy = accurate_classifications / total_samples

        self.validation_history.append({
            'step': self.logger.current_step,
            'loss': avg_loss,
            'accuracy': avg_accuracy
        })

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            if self.snapshot_path is not None:
                torch.save({
                    'epoch': self.current_epoch,
                    'step': self.logger.current_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': avg_loss,
                    'val_accuracy': avg_accuracy
                }, self.snapshot_path + '/' + self.snapshot_name + '_best.pth')
                print(f"\n✅ New best model saved! Val Loss: {avg_loss:.4f}")

        # Scheduler update
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(avg_loss)

        if self.last_epoch_printed != self.current_epoch:
            new_lr = self.optimizer.param_groups[0]['lr']
            print(f"[Scheduler] LR: {new_lr:.2e}", end="")
            if new_lr != old_lr:
                print(f"  (reduced from {old_lr:.2e})", end="")
            print()
            self.last_epoch_printed = self.current_epoch

        self.model.train()
        return avg_loss, avg_accuracy