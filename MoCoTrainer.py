import torch
import tqdm #module that wraps status bars
import os
from torch import nn

class MoCoTrainer():

    def __init__(self, Qencoder, Kencoder, Dqueue, loss_fn, optimizer, scheduler, tau, cMomentum, flg, device):
        super(MoCoTrainer,self).__init__()

        # Encoder models:
        self.QE = Qencoder
        self.KE = Kencoder

        # Dictionary queue:
        self.Dqueue = Dqueue
        self.queue_ptr = 0

        # Optimization processors:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Hyperparameters:
        self.loss_fn = loss_fn
        self.tau = tau # Contrastive loss temperament
        self.momentum = cMomentum # Contrast update momentum


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
            self.QE.train()
            # Set Keys encoder to evaluation mode:
            # self.KE.eval()

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


            # Update lr scheduler:
            self.scheduler.step()

            # Save model weights every 10 epochs:
            if epoch % 10 == 0:
                torch.save(self.QE.state_dict(),f'{os.getcwd()}/Model_Weights.pt')

    def train_batch(self, batch):

        (Xk, Xq), gt_labels = batch

        batch_size, _, _, _ = Xk.shape
        Xq = Xq.to(device= self.device) # Query batch
        Xk = Xk.to(device= self.device) # Positive keys batch
        gt_labels = gt_labels.to(device= self.device) # No need for "gt" labels in this scheme

        self.optimizer.zero_grad() # Set previous gradients to zero

        # Produce model outputs via forward pass:
        queries = nn.functional.normalize(self.QE(Xq), p=2, dim=1)
        keys = nn.functional.normalize(self.QE(Xk), p=2, dim=1)

        # Calcualte positive-negative output multiplications:
        N, C = keys.shape  # Shape of input data (NxC)
        _, K = self.Dqueue.queue.shape # Dictionary queue length
        l_pos = torch.bmm(queries.view(N, 1, C), keys.view(N, C, 1)).squeeze(-1) # dim (Nx1)
        l_neg = torch.mm(queries.view(N, C), self.Dqueue.queue.clone().detach().view(C, K).to(device=self.device)) # dim (NxK)

        # Construct inputs for contrastive loss function:
        logits = torch.cat([l_pos, l_neg], dim=1)
        contrastive_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device=self.device)

        # # QE Backwards Pass:
        # loss.backward()
        loss = self.loss_fn(logits / self.tau, contrastive_labels)

        loss.backward()

        ''' Update encoders' param: '''
        # QE:
        self.optimizer.step()

        # KE:
        with torch.no_grad():
            for thetaQ, thetaK in zip(self.QE.parameters(), self.KE.parameters()):
               thetaK.data.copy_(thetaK.data * self.momentum + thetaQ.data * (1. - self.momentum))

        self.Dqueue.UpdateQueue(keys=keys)

        # Dispose GPU device of residual cache:
        torch.cuda.empty_cache()

        return loss.item() # Takes torch tensor and outputs a scalar


class DictionaryQueue():

    def __init__(self, output_size, queue_size, device):

        self.device = device
        self.output_size = output_size
        self.queue_size = queue_size
        self.queue = nn.functional.normalize(torch.randn(output_size, queue_size), p=2, dim=1)
        self.queue.to(device=self.device)
        self.queue_ptr = 0

    def UpdateQueue(self, keys):
        # Update the Key-dictionary by enqueuing current output keys and dequeuing the oldest ones:

        batch_size = keys.shape[0]

        ptr = self.queue_ptr
        assert self.output_size % batch_size == 0  # for simplicity

        self.queue[:, ptr:ptr + batch_size] = torch.transpose(keys, 1, 0)
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr = ptr






