
import torch
import torch.nn as nn
from transformers import ResNetModel, BertModel, AutoModel,  AutoTokenizer, AutoImageProcessor, AdamW, get_scheduler
from tqdm.auto import tqdm
from utils.utils import compute_metrics, format_metrics
from utils.FakeNewsDataset import collate_fn
import os

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gilberto = AutoModel.from_pretrained("idb-ita/gilberto-uncased-from-camembert")
        self.tokenizer = AutoTokenizer.from_pretrained("idb-ita/gilberto-uncased-from-camembert")
        self.tokenizerLast = AutoTokenizer.from_pretrained("idb-ita/gilberto-uncased-from-camembert", padding_side = 'left', truncation_side = 'left')
        self.linear1 = nn.Linear(768, 768)
        self.layer_norm = nn.BatchNorm1d(768)
        self.dropout = nn.Dropout(0.1)
        self.linear3 = nn.Linear(768, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        
        #textual embeddings extraction from bert
        embeddings_text = self.gilberto(input_ids = input_ids, attention_mask = attention_mask).pooler_output

        embeddings = self.linear1(embeddings_text)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.relu(embeddings)

        logits = self.linear3(embeddings)
        
        probs = self.softmax(logits)
        return logits, probs
    
class GilbertoModel():

    def __init__(self):
        self.model = Model()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def get_tokens(self, texts, tokenization_strategy):
        if tokenization_strategy == "first":
            return self.model.tokenizer(texts, return_tensors="pt", padding = True, truncation=True, max_length=512)
        elif tokenization_strategy == "last":
            return self.model.tokenizerLast(texts, return_tensors="pt", padding = True, truncation=True, max_length=512)
        elif tokenization_strategy == "head-tail":
            max_len = 512
            tokens = self.model.tokenizer(texts)
            half_len = int(max_len/2)
            post_tokens = {
                "input_ids": [],
                "attention_mask": []
            }
            max_token_len = 0
            for token_list in tokens.input_ids:
                tl = len(token_list)
                if tl>max_token_len:
                    max_token_len = tl
            max_len = min(max_token_len, max_len)
            for token_list in tokens.input_ids:
                new_tokens = []
                tl = len(token_list)
                if tl>max_len:
                    new_tokens = token_list[:half_len] + token_list[-half_len:]
                    new_tokens[-1] = [103]
                    attention_mask = [1] * max_len
                elif tl<=max_len:
                    # add padding
                    new_tokens = token_list + [0] * (max_len - tl)
                    attention_mask = [1] * tl + [0] * (max_len - tl)
                post_tokens["input_ids"].append(new_tokens)
                post_tokens["attention_mask"].append(attention_mask)
            post_tokens["input_ids"] = torch.tensor(post_tokens["input_ids"])
            post_tokens["attention_mask"] = torch.tensor(post_tokens["attention_mask"])
            return post_tokens
        else:
            raise ValueError(f"tokenization_strategy {tokenization_strategy} not supported")
        
    
    def eval(self, ds, tokenization_strategy, batch_size = 8):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size,collate_fn = collate_fn
        )
        total_preds = []
        total_labels = []
        progress_bar = tqdm(range(len(dataloader)))
        with torch.no_grad():
            for batch in dataloader:
                texts = batch["text"]
                labels = batch["label"]
                
                texts = [t.lower() for t in texts]


                
                t_inputs = self.get_tokens(texts, tokenization_strategy)

                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                
                logits, probs = self.model(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"],
                )

                preds = torch.argmax(logits, dim=1).tolist()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        return metrics
    
    def train(self, train_ds, val_ds, lr = 5e-5, batch_size= 8, num_epochs = 3, eval_every_epoch = False, warmup_steps = 0, num_eval_steps = 10, save_path = "./", tokenization_strategy = "first"):
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
        progress_bar = tqdm(range(num_training_steps))
        current_step = 0
        # save the best model
        best_metrics = [0, 0, 0, 0, 0]
        print("accuracy | precision | recall | f1 | f1_weighted | f1_for_each_class")
        for epoch in range(num_epochs):
            for batch in dataloader:
                texts = batch["text"]
                labels = batch["label"]

                texts = [t.lower() for t in texts]

                t_inputs = self.get_tokens(texts, tokenization_strategy)

                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)

                labels_tensor = torch.tensor(labels).to(device)

                logits, probs = self.model(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"]
                )

                loss = criterion(logits, labels_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                current_step += 1
                progress_bar.update(1)
            print("Epoch: ", epoch, " | Step: ", current_step, " | Loss: ", loss.item())
            eval_metrics = self.eval(val_ds, tokenization_strategy, batch_size = batch_size)
            print("Eval metrics: ", format_metrics(eval_metrics))
            f1_score = eval_metrics["f1_weighted"]
            if f1_score > min(best_metrics):
                best_metrics.remove(min(best_metrics))
                best_metrics.append(f1_score)
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth" + str(f1_score)))
            print("Best metrics: ", best_metrics)
            self.model.train()
        return self.model