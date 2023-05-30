from transformers import AutoImageProcessor, ResNetModel, AdamW, get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import ast
from PIL import Image
from utils.FakeNewsDataset import collate_fn
from tqdm.auto import tqdm
from utils.utils import compute_metrics, format_metrics, display_confusion_matrix
import random

class Model(nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()

        self.input_sizes = {
            "microsoft/resnet-18": 512,
            "microsoft/resnet-50": 2048,
            "microsoft/resnet-101": 2048,
            "microsoft/resnet-152": 2048,
        }
        
        self.base_model = ResNetModel.from_pretrained(backbone)
        self.processor = AutoImageProcessor.from_pretrained(backbone)
        self.flatten = nn.Flatten(1,-1)
        self.relu = nn.ReLU()

        base_model_sd = self.base_model.state_dict()
    
        self.linear1 = nn.Linear(self.input_sizes[backbone], self.input_sizes[backbone])
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.BatchNorm1d(self.input_sizes[backbone])

        self.linear3 = nn.Linear(self.input_sizes[backbone], 4)

        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, pixel_values, nums_images, is_train = True):
                
        i_embeddings = self.base_model(pixel_values = pixel_values).pooler_output
        embeddings_images = self.flatten(i_embeddings)
        #compute the max of the embeddings
        
        embeddings_images = self.relu(embeddings_images)

        embeddings = self.linear1(embeddings_images)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.relu(embeddings)

        logits = self.linear3(embeddings)

        if not is_train:
            new_logits = []
            base = 0
            for i in range(len(nums_images)):
                current_sample_logits = logits[base:base+nums_images[i]] 
                current_sample_logits = torch.mean(current_sample_logits, dim=0)
                new_logits.append(current_sample_logits)
                base += nums_images[i]
            logits = torch.stack(new_logits)
        
        
        probs = self.softmax(logits)
        return logits, probs


class ResnetModel():
    def __init__(self, backbone = "microsoft/resnet-50"):
       
        self.model = Model(backbone)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def eval(self, ds, batch_size = 8, print_confusion_matrix = False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, collate_fn = collate_fn
        )
        total_preds = []
        total_labels = []
        #progress_bar = tqdm(range(len(dataloader)))
    
        with torch.no_grad():
            for batch in dataloader:
                images_list = batch["images"]
                labels = batch["label"]
                nums_images = batch["nums_images"]
                

                inputs = self.model.processor(images = images_list, return_tensors="pt")
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                
                
                logits, _ = self.model(inputs["pixel_values"], nums_images, is_train = False)
                preds = logits.argmax(dim=-1).tolist()
                total_preds += list(preds)
                total_labels += list(labels)
                #progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        if print_confusion_matrix:
            display_confusion_matrix(total_preds, total_labels)
        return metrics

    
    def train(self, train_ds, val_ds, lr = 5e-5, batch_size= 8, num_epochs = 3, warmup_steps = 0, save_path = "./"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.train()
        self.model.to(device)

        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
        )

        criterion = nn.CrossEntropyLoss()
        
        
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
                
                images_list = batch["images"]
                nums_images = batch["nums_images"]
                labels = batch["label"]


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

                i_inputs = self.model.processor(images = random_images_list, return_tensors="pt")

                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)

                labels = torch.tensor(labels).to(device)

                
                outputs = self.model(
                    pixel_values=i_inputs["pixel_values"],
                    nums_images = nums_images,
                    is_train = True
                )

                logits = outputs[0]
                loss = criterion(logits, labels)


                optimizer.zero_grad()
                current_step += 1
                loss.backward()
                optimizer.step()
                scheduler.step()
                #progress_bar.update(1)
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(val_ds)
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
    