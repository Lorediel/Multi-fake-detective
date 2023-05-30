from transformers import AutoImageProcessor, ViTModel, AdamW, get_scheduler
import torch
from tqdm.auto import tqdm
from utils.utils import *
from utils.FakeNewsDataset import FakeNewsDataset, collate_fn
from math import floor
import os
import torch.nn as nn
import random
import sys

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.base_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.relu = nn.ReLU()
        self.layernorm = nn.BatchNorm1d(768)
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 768)
        self.linear3 = nn.Linear(768, 4)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, pixel_values):
        
        i_embeddings = self.base_model(pixel_values = pixel_values).pooler_output
        #computehe max of the emnbeddings
        embeddings_images = self.relu(i_embeddings)
        
        embeddings = self.linear1(embeddings_images)
        embeddings = self.relu(embeddings)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        logits = self.linear3(embeddings)
        
        probs = self.softmax(logits)
        return logits, probs




class VisualTransformer():

    def __init__(self, path=None):
        self.model = Model()
        if path is not None:
            load_model(self.model, path)
        

    def eval(self, ds, batch_size = 8):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, collate_fn = collate_fn
        )
        total_preds = []
        total_labels = []
        progress_bar = tqdm(range(len(dataloader)))
    
        with torch.no_grad():
            for batch in dataloader:
                images_list = batch["images"]
                labels = batch["label"]
                nums_images = batch["nums_images"]
                
                inputs = self.model.processor(images = images_list, return_tensors="pt")
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                
                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)
                logits, _ = self.model(inputs["pixel_values"])
                new_logits = []
                base = 0
                for i in range(len(nums_images)):
                    current_sample_logits = logits[base:base+nums_images[i]] 
                    current_sample_logits = torch.mean(current_sample_logits, dim=0)
                    new_logits.append(current_sample_logits)
                    base += nums_images[i]
                logits = torch.stack(new_logits)
                preds = logits.argmax(dim=-1).tolist()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        return metrics

    def train(self, train_ds, val_ds, num_epochs= 30, lr = 5e-5,  warmup_steps = 0, batch_size = 8, num_eval_steps = 10, save_path = "./"):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.train()
        self.model.to(device)
        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
        )
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps=len(dataloader) * num_epochs
        # Initialize the scheduler
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        current_step = 0
        criterion = nn.CrossEntropyLoss()
        best_metrics = [0, 0, 0, 0, 0]
        print("accuracy | precision | recall | f1 | f1_weighted | f1_for_each_class")
        for epoch in range(num_epochs):
            for batch in dataloader:
                #current_step += 1
                #batch = {k: v.to(device) for k, v in batch.items()}
               
                images_list = batch["images"]
                labels = batch["label"]
                nums_images = batch["nums_images"]


                random_images_list = []
                base = 0
                for i in range(len(nums_images)):
                    if nums_images[i] == 1:
                        random_images_list.append(images_list[base])
                        base += nums_images[i]
                        continue
                    random_index = random.randint(0, nums_images[i]-1)
                    sublist = images_list[base:base+nums_images[i]]
                    random_images_list.append(sublist[random_index])
                    base += nums_images[i]

                inputs = self.model.processor(images = random_images_list, return_tensors="pt")
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                
                labels_tensor = torch.tensor(labels).to(device)
                
                outputs = self.model(inputs["pixel_values"])

                logits = outputs[0]

                loss = criterion(logits, labels_tensor)

                optimizer.zero_grad()
                current_step += 1
                loss.backward()
                optimizer.step()
                scheduler.step()
               
                progress_bar.update(1)
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(val_ds)
            print("Eval metrics: ", format_metrics(eval_metrics))
            f1_score = eval_metrics["f1_weighted"]
            if f1_score > min(best_metrics):
                best_metrics.remove(min(best_metrics))
                best_metrics.append(f1_score)
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth" + str(f1_score)))
            print("Best metrics: ", best_metrics)
            self.model.train()
        return best_metrics
