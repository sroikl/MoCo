import torch
import tqdm #module that wraps status bars
import os
from torch import nn

class DST_Trainer():

    def __init__(self,model,loss_fn, optimizer, scheduler, device):
        super(DST_Trainer,self).__init__()

        # Encoder models:
        self.model = model

        # Optimization processors:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Hyperparameters:
        self.loss_fn = loss_fn


    def train(self,dl_train,dl_val,epochs):

        # Initialize general losses:
        train_loss, val_loss = [], []

        for epoch in range(epochs):

            print (f' ============= Epoch {epoch +1}/{epochs} =============')

            # Initialize per-epoch train and val losses:
            epoch_train_loss, epoch_val_loss = [], []

            '''============= Training Set ============='''
            num_train_batches = len(dl_train.batch_sampler)

            # Set Query encoder to training mode:
            self.model.train()

            # Train over the batches with progress-bar description:
            with tqdm.tqdm(total= num_train_batches,desc=f'Training Epoch') as pbar:
                for i,batch in enumerate(dl_train):

                    loss = self.train_batch(batch)
                    epoch_train_loss.append(loss)

                    # Update progress bar:
                    pbar.set_description(f'Batch Loss: {loss:.3f}')
                    pbar.update()

            avg_loss = sum(epoch_train_loss)/num_train_batches
            print(f'Average Epoch Training Loss {avg_loss:.3f}')

            # Append current epoch loss:
            train_loss.append(avg_loss)

            ''' ============= Validation Set ============='''
            num_val_batches = len(dl_val.batch_sampler)

            # Set models to evaluation modes:
            self.model.eval()

            with tqdm.tqdm(total=num_val_batches, desc=f'Val Epoch') as pbar:
                for i,batch in enumerate(dl_val):
                    loss = self.val_batch(batch)
                    epoch_val_loss.append(loss)
                    pbar.set_description(f'Batch Val Loss: {loss:.3f}')
                    pbar.update()

            avg_loss = sum(epoch_val_loss) / num_val_batches
            print(f'Average Epoch Validation Loss {avg_loss:.3f}')

            # Append current epoch loss (val):
            val_loss.append(avg_loss)

            # Update lr scheduler:
            self.scheduler.step()

            # Save model weights every 10 epochs:
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(),f'{os.getcwd()}/DST_classifier.pt')

    def train_batch(self, batch):

        (_, Xq), gt_labels = batch

        batch_size, _, _, _ = Xq.shape
        Xq = Xq.to(device= self.device) # Query batch
        gt_labels = gt_labels.to(device= self.device) # No need for "gt" labels in this scheme

        self.optimizer.zero_grad() # Set previous gradients to zero

        logits= self.model(Xq)

        loss= self.loss_fn(logits,gt_labels)

        loss.backward()

        ''' Update encoders' param: '''

        self.optimizer.step()


        # Dispose GPU device of residual cache:
        torch.cuda.empty_cache()

        return loss.item() # Takes torch tensor and outputs a scalar


    def val_batch(self,batch):

        (_, Xq), gt_labels = batch

        batch_size, _, _, _ = Xq.shape
        Xq = Xq.to(device=self.device)  # Query batch
        gt_labels = gt_labels.to(device=self.device)  # No need for "gt" labels in this scheme

        self.optimizer.zero_grad()  # Set previous gradients to zero


        with torch.no_grad():
            logits = self.model(Xq)

        loss = self.loss_fn(logits, gt_labels)


        # Dispose GPU device of residual cache:
        torch.cuda.empty_cache()

        return loss.item()  # Takes torch tensor and outputs a scalar





