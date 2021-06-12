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
        train_top1Acc, train_top5Acc = [], []
        val_top1Acc, val_top5Acc = [], []

        for epoch in range(epochs):

            print (f' ============= Epoch {epoch +1}/{epochs} =============')

            # Initialize per-epoch train and val losses:
            epoch_train_loss, epoch_val_loss = [], []
            epoch_train_top1Acc, epoch_val_top1Acc = [], []
            epoch_train_top5Acc, epoch_val_top5Acc = [], []

            '''============= Training Set ============='''
            num_train_batches = len(dl_train.batch_sampler)

            # Set Query encoder to training mode:
            self.model.train()

            # Train over the batches with progress-bar description:
            with tqdm.tqdm(total= num_train_batches,desc=f'Training Epoch') as pbar:
                for i,batch in enumerate(dl_train):

                    loss, top1Acc, top5Acc = self.train_batch(batch)

                    epoch_train_loss.append(loss)
                    epoch_train_top1Acc.append(top1Acc)
                    epoch_train_top5Acc.append(top5Acc)

                    # Update progress bar:
                    pbar.set_description(f'Train Batch Loss: {loss:.3f}, Top 1 Acc: {top1Acc:.3f}, Top 5 Acc: {top5Acc:.3f}')
                    pbar.update()

            avg_loss = sum(epoch_train_loss)/num_train_batches
            avg_top1Acc = sum(epoch_train_top1Acc)/num_train_batches
            avg_top5Acc = sum(epoch_train_top5Acc)/num_train_batches
            print(f'Train Epoch (avg) top 1 Acc: {avg_top1Acc:.3f}, Top 5 Acc: {avg_top5Acc:}')

            # Append current epoch loss:
            train_loss.append(avg_loss)
            train_top1Acc.append(avg_top1Acc)
            train_top5Acc.append(avg_top5Acc)

            ''' ============= Validation Set ============='''
            num_val_batches = len(dl_val.batch_sampler)

            # Set models to evaluation modes:
            self.model.eval()

            with tqdm.tqdm(total=num_val_batches, desc=f'Val Epoch') as pbar:
                for i,batch in enumerate(dl_val):

                    loss, top1Acc, top5Acc = self.val_batch(batch)

                    epoch_val_loss.append(loss)
                    epoch_val_top1Acc.append(top1Acc)
                    epoch_val_top5Acc.append(top5Acc)

                    # Update progress bar:
                    pbar.set_description(f'Val Batch Loss: {loss:.3f}, Top 1 Acc: {top1Acc:.3f}, Top 5 Acc: {top5Acc:.3f}')
                    pbar.update()

            avg_loss = sum(epoch_val_loss)/num_val_batches
            avg_top1Acc = sum(epoch_val_top1Acc)/num_val_batches
            avg_top5Acc = sum(epoch_val_top5Acc)/num_val_batches
            print(f'Val Epoch (avg) top 1 Acc: {avg_top1Acc:.3f}, Top 5 Acc: {avg_top5Acc:.3f}')

            # Append current epoch loss:
            val_loss.append(avg_loss)
            val_top1Acc.append(avg_top1Acc)
            val_top5Acc.append(avg_top5Acc)


            # Update lr scheduler:
            self.scheduler.step()

            # Save model weights every 10 epochs:
            if (epoch % 10 == 0) | (epoch == epochs):
                torch.save(self.model.state_dict(),f'{os.getcwd()}/DST_classifier_epoch_{epoch}.pt')
                loss_dict= dict(train_loss= {'loss':train_loss,'top1acc':train_top1Acc,'top5acc':train_top5Acc},
                                val_loss=  {'loss':val_loss,'top1acc':val_top1Acc,'top5acc':val_top5Acc})
                torch.save(loss_dict,f'{os.getcwd()}/DST_loss_dict_epoch_{epoch}.pt')

    def train_batch(self, batch):

        (Xq, _), gt_labels = batch

        batch_size, _, _, _ = Xq.shape
        Xq = Xq.to(device= self.device) # Query batch
        gt_labels = gt_labels.to(device= self.device) # No need for "gt" labels in this scheme

        self.optimizer.zero_grad() # Set previous gradients to zero

        logits= self.model(Xq)

        loss= self.loss_fn(logits,gt_labels)

        loss.backward()

        ''' Update encoders' param: '''

        self.optimizer.step()

        top1Acc = self.accuracy(output=logits.detach().cpu(), target=gt_labels.detach().cpu(), topk=(1,))
        top5Acc = self.accuracy(output=logits.detach().cpu(), target=gt_labels.detach().cpu(), topk=(5,))
        # Dispose GPU device of residual cache:
        torch.cuda.empty_cache()

        return loss.item(), top1Acc, top5Acc # Takes torch tensor and outputs a scalar


    def val_batch(self,batch):

        (_, Xq), gt_labels = batch

        batch_size, _, _, _ = Xq.shape
        Xq = Xq.to(device=self.device)  # Query batch
        gt_labels = gt_labels.to(device=self.device)  # No need for "gt" labels in this scheme

        self.optimizer.zero_grad()  # Set previous gradients to zero


        with torch.no_grad():
            logits = self.model(Xq)

        loss = self.loss_fn(logits, gt_labels)


        top1Acc = self.accuracy(output=logits.detach().cpu(), target=gt_labels.detach().cpu(), topk=(1,))
        top5Acc = self.accuracy(output=logits.detach().cpu(), target=gt_labels.detach().cpu(), topk=(5,))

        # Dispose GPU device of residual cache:
        torch.cuda.empty_cache()

        return loss.item(), top1Acc, top5Acc # Takes torch tensor and outputs a scalar


    def accuracy(self, output, target, topk=(1,)):
            """Computes the precision@k for the specified values of k"""

            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / batch_size))

            return res[0].item()





