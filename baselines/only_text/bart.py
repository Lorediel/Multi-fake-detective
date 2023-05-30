from transformers import AutoModel, AutoTokenizer, ResNetModel, AutoImageProcessor, VisionTextDualEncoderModel, AutoProcessor, AdamW, get_scheduler, AutoModelForSeq2SeqLM

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from utils.FakeNewsDataset import collate_fn
import random
from utils import format_metrics, compute_metrics
import os
from tqdm.auto import tqdm
from utils.focal_loss import FocalLoss
from torchvision import transforms

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class BartModel(nn.Module):
    def __init__(self, device, pretrain = "morenolq/bart-it-ilpost"):
        super().__init__()
        self.bart = AutoModel.from_pretrained(pretrain)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain)
        self.device = device

        self.l1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, texts):
        tokens = self.tokenizer(texts, padding=True, max_length=1024, truncation=True, return_tensors="pt")
        for k, v in tokens.items():
            tokens[k] = v.to(self.device)
        embeds = self.bart(**tokens).encoder_last_hidden_state[:,0,:]
        embeds = self.l1(embeds)
        logits = self.classifier(embeds)
        return logits
    

class Trainer:
    def __init__(self):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BartModel(self.device).to(self.device)
        

    def eval(self, ds, tokenization_strategy = "first", batch_size=8):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, collate_fn = collate_fn
        )
        total_preds = []
        total_labels = []
        #progress_bar = tqdm(range(len(dataloader)))
        with torch.no_grad():
            for batch in dataloader:
                texts = batch["text"]
                labels = batch["label"]

                        
                preds = self.model(
                    texts
                )

                preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
                total_preds += list(preds)
                total_labels += list(labels)
                #progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        return metrics
    
    def train(self, train_ds, eval_ds, lr = 5e-5, batch_size= 8, num_epochs = 30, warmup_steps = 0, save_path = "./", loss= "cross_entropy"):
            
            dataloader = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
            )
            if loss == "cross_entropy":
                criterion = nn.CrossEntropyLoss()
            elif loss == "focal":
                criterion = FocalLoss(gamma=2, reduction='mean')
            else: 
                raise ValueError("Loss not supported")
            
            
            self.model.train()
            # Initialize the optimizer
            optimizer = AdamW(self.model.parameters(), lr=lr)
            num_training_steps=len(dataloader) * num_epochs
            # Initialize the scheduler
            scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
            #progress_bar = tqdm(range(num_training_steps))
            current_step = 0
            # save the best model
            best_metrics = [0, 0, 0]
            print("accuracy | precision | recall | f1 | f1_weighted | f1_for_each_class")
            for epoch in range(num_epochs):
                for batch in dataloader:
                    current_step += 1
                    texts = batch["text"]
                    labels = batch["label"]
                    
                    labels = torch.tensor(labels).to(self.device)

                    #nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=self.device)

                    logits = self.model(
                        texts = texts
                    )
                    
                    loss = criterion(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    #progress_bar.update(1)
                print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
                eval_metrics = self.eval(eval_ds, batch_size=batch_size)
                print("Eval metrics: ", format_metrics(eval_metrics))
                f1_score = eval_metrics["f1_weighted"]
                if f1_score > min(best_metrics):
                    best_metrics.remove(min(best_metrics))
                    # remove the old best model 
                    if os.path.exists(os.path.join(save_path, "best_model.pth" + str(min(best_metrics)))):
                        os.remove(os.path.join(save_path, "best_model.pth" + str(min(best_metrics))))
                    best_metrics.append(f1_score)
                    torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth" + str(f1_score)))
                print("Best metrics: ", best_metrics)
                self.model.train()
        
            return best_metrics
    