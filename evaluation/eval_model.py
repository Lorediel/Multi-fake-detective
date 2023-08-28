from transformers import AutoModel, AutoTokenizer, ResNetModel, AutoImageProcessor, VisionTextDualEncoderModel, AutoProcessor, AdamW, get_scheduler, AutoModelForSeq2SeqLM
from utils.utils import get_confusion_matrix
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from utils.FakeNewsDataset import collate_fn, FakeNewsDataset
from utils.utils import stratifiedSplit, compute_metrics
from statistics import mean
import sys
from models_for_ensemble.A2_B import FNDModel as FNDModel1
from models_for_ensemble.A1_E_D import FNDModel as FNDModel2 
from models_for_ensemble.A1_C_D import FNDModel as FNDModel3
from itertools import combinations
import pandas as pd
import sys

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def extract_logits(model, ds, batch_size=16, device="cuda:0"):
    model.eval()
    model.to(device)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn = collate_fn, shuffle = False)
    total_preds = []
    total_labels = []
    all_logits = []
    total_images = []
    #progress_bar = tqdm(range(len(dataloader)))
    with torch.no_grad():
        for batch in dataloader:
            texts = batch["text"]
            images_list = batch["images"]
            labels = batch["label"]
            nums_images = batch["nums_images"]

            new_texts = []
            for k in range(len(nums_images)):
                current_text = texts[k]
                replication = nums_images[k]
                for i in range(replication):
                    new_texts.append(current_text)
            texts = new_texts

            
                    
            logits = model(texts, images_list) # batch_size x 4

            all_logits.append(logits)
            total_labels += list(labels)
            total_images += list(nums_images)

    logits_tensor = torch.cat(all_logits, dim=0)
    return logits_tensor, total_labels, total_images

def make_eval_final(logits, nums_images, labels, strategy="probs"):
    if strategy == "probs":
        probs = F.softmax(logits, dim=1)
        preds = list(torch.argmax(probs, dim=1).detach().cpu().numpy())
        new_preds = []
        base = 0
        for i in range(len(nums_images)):
            current_preds = preds[base:base+nums_images[i]]
            if 0 in current_preds or 1  in current_preds:
                # take the 0s and 1s in current_preds
                fake_preds = [p for p in current_preds if p == 0 or p == 1]
                # take the average of the fake preds
                new_preds.append(round(mean(fake_preds)))
            else:
                new_preds.append(round(mean(current_preds)))
            base += nums_images[i]
        preds = new_preds
    elif strategy == "logits":
        # take the average of the logits
        total_preds = []
        total_labels = []
        new_logits = []
        base = 0
        for i in range(len(nums_images)):
            current_sample_logits = logits[base:base+nums_images[i]]
            current_sample_logits = torch.mean(current_sample_logits, dim=0)
            new_logits.append(current_sample_logits)
            base += nums_images[i]
        logits = torch.stack(new_logits)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
        total_preds += list(preds)
        preds = total_preds
    metrics = compute_metrics(preds, labels)
    cm = get_confusion_matrix(preds, labels)
    return metrics, cm

if __name__ == "__main__":
    args = sys.argv
    tsvfile = args[1]
    mediafile = args[2]
    
    # path of the model
    model_path = args[3]

    model_type = args[4]

    dataset = FakeNewsDataset(tsvfile, mediafile)
    train_ds, val_ds = stratifiedSplit(dataset)
    
    dataset = FakeNewsDataset('./MULTI-Fake-Detective_Task1_Data.tsv','../Media')
    train_ds, val_ds = stratifiedSplit(dataset)

    if model_type == "a2_b":
        model = FNDModel1(None, None, None)
    elif model_type == "a1_e_d":
        model = FNDModel2(None, None, None)
    elif model_type == "a1_c_d":
        model = FNDModel3(None, None, None)
    else:
        raise Exception("Invalid model type")

    load_model(model, model_path)

    logits, labels, nums_images = extract_logits(model, val_ds)

    metrics, cm = make_eval_final(logits, nums_images, labels, strategy="logits")

    print("Metrics: ", metrics)
    print("Confusion Matrix: ", cm)
    

    





