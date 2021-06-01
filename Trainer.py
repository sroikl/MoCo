import torch
import tqdm #module that wraps status bars
import os

class MoCoTrainer():
    def __init__(self,model,loss_fn,optimizer,scheduler,tau,flg,device):
        super(MoCoTrainer,self).__init__()


        self.model= model
        self.optimizer= optimizer
        self.scheduler= scheduler
        self.device= device

        self.loss_fn= loss_fn
        self.tau= tau



    def fit(self,dl_train,dl_val,epochs):
        train_loss,val_loss= [],[]

        for epoch in range(epochs):
            print (f' =============================== Epoch {epoch +1}/{epochs} ===============================')
            epoch_train_loss,epoch_val_loss= [],[]

            '''========================= Train Part ========================='''
            num_train_batches= len(dl_train.batch_sampler)

            dl_iter= iter(dl_train)
            with tqdm.tqdm(total= num_train_batches,desc=f'Train Epoch') as pbar:
                for i in range(num_train_batches):
                    batch= next(dl_iter)
                    loss= self.train_batch(batch)
                    epoch_train_loss.append(loss)
                    pbar.set_description(f'Batch Train Loss: {loss:.3f}')
                    pbar.update()

            avg_loss= sum(epoch_train_loss)/num_train_batches
            pbar.set_description(f'Epoch Train Loss {avg_loss:.3f}')
            train_loss.append(avg_loss)

            ''' ========================= Validation Part ========================='''
            num_val_batches = len(dl_val.batch_sampler)

            dl_iter = iter(dl_val)
            with tqdm.tqdm(total=num_val_batches, desc=f'Val Epoch') as pbar:
                for i in range(num_val_batches):
                    batch = next(dl_iter)
                    loss = self.val_batch(batch)
                    epoch_val_loss.append(loss)
                    pbar.set_description(f'Batch Val Loss: {loss:.3f}')
                    pbar.update()

            avg_loss = sum(epoch_val_loss) / num_val_batches
            pbar.set_description(f'Epoch Val Loss {avg_loss:.3f}')
            val_loss.append(avg_loss)

            if epoch%10==0:
                torch.save(self.model.state_dict(),f'{os.getcwd()}/Model_Weights.pt')

    def train_batch(self,batch):
        (x_k,x_q),gt_labels= batch

        batch_size,_,_,_= x_k.shape
        x_q= x_q.to(device= self.device)
        x_k= x_k.to(device= self.device)
        gt_labels= gt_labels.to(device= self.device)


        self.optimizer.zero_grad()

        #Forward Pass
        logits,keys= self.model(x_k=x_k,x_q=x_q)#TODO:try adding method in model and see if need to call forward
        contrastive_labels= torch.zeros(logits.shape[0], dtype=torch.long).to(device=self.device)

        #calculate Loss
        loss= self.loss_fn(logits/self.tau,contrastive_labels)

        #backward Pass
        loss.backward()
        self.optimizer.step()

        torch.cuda.empty_cache()

        return loss.item() #takes torch tensor and outpots the scalar

    def val_batch(self,batch):

        (x_k, x_q), gt_labels = batch

        batch_size, _, _, _ = x_k.shape
        x_q = x_q.to(device=self.device)
        x_k = x_k.to(device=self.device)
        gt_labels = gt_labels.to(device=self.device)

        contrastive_labels= torch.zeros(batch_size,dtype=torch.long).to(device= self.device)
        self.optimizer.zero_grad()

        # Forward Pass
        with torch.no_grad():
            logits, keys = self.model(x_k=x_k, x_q=x_q)  # TODO:try adding method in model and see if need to call forward

        # calculate Loss
        loss= self.loss_fn(logits/self.tau,contrastive_labels)

        return loss.item()




