import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from gru_logging import Logger 
from stft_loss import MRSTFTLoss 
from gru_model import GRUModel 

class GRUTrainer:
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
                 snapshot_interval=800,
                 device=torch.device("cpu")):
        
        self.model = model
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
        self.scaler = GradScaler(device='cuda')
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=1e-3, min_lr=1e-5)
        self.logger = logger    
        self.loss_function = MRSTFTLoss()
        self.validation_history = []
        self.logger.trainer = self
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.last_epoch_printed = -1
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.patience = 10
        self.device = device

    def train(self,
              batch_size=32,
              epochs=10,
              start_epoch=0,
              continue_training_at_step=0):
        self.model.train()
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=True,)
        
        self.dataset.train = False
        self.val_dataloader = torch.utils.data.DataLoader(self.dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=True)
        
        self.dataset.train = True
        
        print(f" Dataset size: {len(self.dataset)} samples")
        print(f" Batch size: {batch_size}")
        print(f" Batches per epoch: {len(self.dataloader)}")
        
        step = continue_training_at_step
        batch_print_limit = 3
        
        for current_epoch in range(start_epoch, epochs):
            print(f"\n─────────────────────────────")
            print(f"Epoch {current_epoch + 1}")
            print(f"─────────────────────────────")
            self.current_epoch = current_epoch
            tic = time.time()
            
       
            for (clean_input, target) in iter(self.dataloader): 
                start_batch_time = time.time()
                
               
                start_data_load_time = time.time()
                # clean_input shape: [B, L, 1]
                clean_input = clean_input.to(self.device).float() 
                # target shape: [B*L, 1]
                target = target.to(self.device).float() 
                             
                end_data_load_time = time.time()

                if step < batch_print_limit:
                    print(f"\nData Load/Transfer Time: {end_data_load_time - start_data_load_time:.4f}s")
                start_forward_time = time.time()

                with autocast(device_type='cuda'): 
                    output = self.model(clean_input) # output shape: [B*L, 1]
                    
                    if step == 0:
                        print("🔹 DEBUG INPUT SHAPE:", clean_input.shape)
                        print("🔹 DEBUG OUTPUT SHAPE:", output.shape)
                        print("🔹 DEBUG TARGET SHAPE:", target.shape)
                        print("🔹 DEBUG TARGET SHAPE BEFORE VIEW:", target.shape)
                        print("🔹 DEBUG TARGET SHAPE AFTER VIEW:", target.view(-1).shape)
                    
                    # Use MRSTFT Loss 
                    loss = self.loss_function(output, target) 
                end_forward_time = time.time()

                if step < batch_print_limit:
                    print(f"Forward Pass Time: {end_forward_time - start_forward_time:.4f}s") 

                self.optimizer.zero_grad()

                start_backward_time = time.time()
                self.scaler.scale(loss).backward() 
                
                if self.clip is not None:
                    self.scaler.unscale_(self.optimizer) 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.scaler.step(self.optimizer) 
                self.scaler.update() 
                end_backward_time = time.time()
                if step < batch_print_limit:
                    print(f"Backward/Optimizer Time: {end_backward_time - start_backward_time:.4f}s")
                    
                loss = loss.item()

                end_batch_time = time.time()
                if step < batch_print_limit:
                    print(f"Total Batch Time: {end_batch_time - start_batch_time:.4f}s")
                    print("-------------------------------------")

                step += 1

               
                total_steps = len(self.dataloader) * (epochs - self.current_epoch) + step
                current_progress = (step / total_steps) * 100
                print(f"\rTraining Progress: {current_progress:.2f}%", end="")
                
               
                if step == 100:
                    toc = time.time()
                    print("\none training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

                if step % self.snapshot_interval == 0:
                    if self.snapshot_path is None:
                        continue
                    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                    torch.save({
                        'epoch': self.current_epoch,
                        'step': step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': loss 
                    }, self.snapshot_path + '/' + self.snapshot_name + '_' + time_string + '.pth')
                    
                self.logger.log(step, loss)
            torch.cuda.empty_cache() 

    def validate(self):
        self.model.eval()
        self.dataset.train = False
        total_loss = 0
        
        with torch.no_grad():
            for (clean_input, target) in iter(self.val_dataloader):
                clean_input = clean_input.to(self.device).float()
                target = target.to(self.device).float()                
                with autocast(device_type='cuda'):                    
                    output = self.model(clean_input)   
                    loss = self.loss_function(output, target)
 
                    total_loss += loss.item()

        avg_loss = total_loss / len(self.val_dataloader)
        
        current_step = self.logger.current_step

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            if self.snapshot_path is not None:
                torch.save({
                    'epoch': self.current_epoch,
                    'step': current_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_loss
                }, self.snapshot_path + '/best_model.pth') 
                print(f"\n✅ New best model saved! Val loss: {avg_loss:.4f}")

        self.validation_history.append({
            'step': current_step,
            'loss': avg_loss,
            })
        
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(avg_loss)
        current_epoch = getattr(self, "current_epoch", None)
        if current_epoch is not None and self.last_epoch_printed != current_epoch:
            for param_group in self.optimizer.param_groups:
                print(f"[Scheduler Update] Current LR: {param_group['lr']}")
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f" Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
            self.last_epoch_printed = current_epoch
            
        self.dataset.train = True
        self.model.train()
        return avg_loss


