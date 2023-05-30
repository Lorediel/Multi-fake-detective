from transformers import ResNetModel, BertModel, AutoModel,  AutoTokenizer, AutoImageProcessor, AdamW, get_scheduler
from tqdm.auto import tqdm
from utils.utils import compute_metrics, format_metrics
from utils.FakeNewsDataset import collate_fn
import os
from utils.focal_loss import FocalLoss
import torch
from torch import nn


import math
from typing import Tuple


class BertParts(nn.Module):

  def __init__(self, pretrained_model_path = None):
    super().__init__()
    self.bert = AutoModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
    if (pretrained_model_path != None):
        s = torch.load(pretrained_model_path)
        new_s = {}
        for n in s:
            if (n.startswith("bert")):
                new_s[n[5:]] = s[n]
        self.bert.load_state_dict(new_s)
    self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
    self.max_len = 512
    for param in self.bert.parameters():
       param.requires_grad = False
    self.pooler = nn.Sequential(
      nn.Linear(768, 768),
      nn.Tanh(),
    )
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def forward(self, texts):
    tokens_list = self.tokenizer(texts).input_ids    
    output = []
    masks = []
    num_sublists = []
    divided_tokens = []
    longest = 0
    for tokens in tokens_list:
      tokens.pop(0) #remove cls
      tokens.pop(-1) #remove eos
      n = 0
      
      for x in range(0, len(tokens), self.max_len - 2):
        chunk = [102] + tokens[x:x+self.max_len-2] + [103]
        mask = [1] * self.max_len
        #pad the last chunk
        if (len(chunk) != self.max_len):
          mask = [1] * len(chunk) + [0] * (self.max_len - len(chunk))
          chunk = chunk + ([0] * (self.max_len - len(chunk)))
        divided_tokens.append(chunk)   
        masks.append(mask)
        n+=1
      num_sublists.append(n)
    
    input_ids = torch.tensor(divided_tokens).to(self.device)
    attention_masks = torch.tensor(masks).to(self.device)
    bertOutput = self.bert(input_ids, attention_masks).last_hidden_state[:,0,:]
    bertOutput = self.pooler(bertOutput)


    base = 0
    final = []
    for i in range(len(num_sublists)):
      tensors = bertOutput[base:base+num_sublists[i]]
      mean_tensor = torch.mean(tensors, dim = 0)

      final.append(mean_tensor)
      base += num_sublists[i]
    final = torch.stack(final, dim=0).to(self.device)
    
    return final
    

class Model(nn.Module):
    def __init__(self, pretrained_model_path = None):
        super(Model, self).__init__()
        self.bertParts = BertParts(pretrained_model_path)
        self.relu = nn.ReLU()
        self.linear1 = nn.Sequential(
          nn.Linear(768, 768),
          nn.LayerNorm(768),
          nn.Dropout(0.1),
          nn.ReLU(),
        )
        self.linear3 = nn.Linear(768, 4)

           
    def forward(self, texts):
        bert_output = self.bertParts(texts)

        cls_out = self.linear1(bert_output)
        cls_out = self.linear3(cls_out)
        logits = cls_out
        return logits

    

class LongBert():

    def __init__(self, pretrained_model_path = None):
        self.model = Model(pretrained_model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def eval(self, ds, batch_size = 8):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, collate_fn = collate_fn
        )
        total_preds = []
        total_labels = []
        progress_bar = tqdm(range(len(dataloader)))
        with torch.no_grad():
            for batch in dataloader:
                texts = batch["text"]
                labels = batch["label"]
                
                
                logits = self.model(texts)

                preds = torch.argmax(logits, dim=1).tolist()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        return metrics

    def train(self, train_ds, val_ds, lr = 5e-5, batch_size= 8, num_epochs = 25, save_path = "./", focal_loss = False):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.train()
        self.model.to(device)

        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
        )

        if focal_loss:
            criterion = FocalLoss(gamma = 2, reduction = "sum")
        else:
            criterion = nn.CrossEntropyLoss()
        # Initialize the optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps=len(dataloader) * num_epochs
        # Initialize the scheduler
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        current_step = 0
        # save the best model
        best_metrics = [0, 0, 0, 0, 0]
        print("accuracy | precision | recall | f1 | f1_weighted | f1_for_each_class")
        for epoch in range(num_epochs):
            for batch in dataloader:
                texts = batch["text"]
                labels = batch["label"]

                labels_tensor = torch.tensor(labels).to(device)

                logits = self.model(texts)

                loss = criterion(logits, labels_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                current_step += 1
                progress_bar.update(1)
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(val_ds, batch_size = batch_size)
            print("Eval metrics: ", format_metrics(eval_metrics))
            f1_score = eval_metrics["f1_weighted"]
            if f1_score > min(best_metrics):
                best_metrics.remove(min(best_metrics))
                best_metrics.append(f1_score)
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth" + str(f1_score)))
            print("Best metrics: ", best_metrics)
            self.model.train()
        return best_metrics