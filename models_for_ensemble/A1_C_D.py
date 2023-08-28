from transformers import AutoModel, AutoTokenizer, ResNetModel, AutoImageProcessor, VisionTextDualEncoderModel, AutoProcessor, AdamW, get_scheduler, AutoModelForSeq2SeqLM

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from utils.FakeNewsDataset import collate_fn, FakeNewsDataset
import random
from utils import format_metrics, compute_metrics
import os
from tqdm.auto import tqdm
from utils.focal_loss import FocalLoss
from torchvision import transforms
import sys
from utils.utils import stratifiedSplit

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class BertModel(nn.Module):
    def __init__(self, device, bert_model = None):
        super().__init__()
        self.bert = AutoModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        ds = self.bert.state_dict()
        if (bert_model != None):
            s = torch.load(bert_model)
            new_s = {}
            for n in s:
                if (n.startswith("bert")):
                    new_s[n[5:]] = s[n]
            self.bert.load_state_dict(new_s)
            # freeze only if pretrained
            for name, param in ds.items():
                param.required_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        self.device = device
    
    def forward(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        for k, v in tokens.items():
            tokens[k] = v.to(self.device)
        embeds = self.bert(**tokens).pooler_output
        return embeds
    
class ResnetModel(nn.Module):
    def __init__(self, device, resnet_model = None):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-152")
        ds = self.resnet.state_dict()
        if (resnet_model != None):
            s = torch.load(resnet_model)
            new_s = {}
            for n in s:
                if (n.startswith("base_model")):
                    new_s[n[11:]] = s[n]
            self.resnet.load_state_dict(new_s)
            # freeze only if pretrained
            for name, param in ds.items():
                param.required_grad = False
        
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        self.flatten = nn.Flatten(1,-1)
        self.device = device

    def forward(self, images):

        i_inputs = self.processor(images = images, return_tensors="pt")

        for k, v in i_inputs.items():
            i_inputs[k] = v.to(self.device)

        pixel_values = i_inputs["pixel_values"]
        embeds = self.resnet(pixel_values = pixel_values).pooler_output
        return self.flatten(embeds)

class BertSentiment(nn.Module):
    def __init__(self, device):
        super().__init__()
        model_path = "neuraly/bert-base-italian-cased-sentiment"
        self.model = AutoModel.from_pretrained(model_path)
        model_sd = self.model.state_dict()
        for param in self.model.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        
    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors = "pt", truncation = True, padding = True)
        for k, v in tokens.items():
            tokens[k] = v.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids = tokens.input_ids, attention_mask = tokens.attention_mask).pooler_output
        
        return out

class ClipModel(nn.Module):

    def __init__(self, device, clip_model = None):
        super().__init__()
        self.clip = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        if (clip_model != None):
            s = torch.load(clip_model)
            new_s = {}
            for n in s:
                if (n.startswith("base_model")):
                    new_s[n[11:]] = s[n]
            self.clip.load_state_dict(new_s)
        # freeze no matter what, to exploit the similarities
        for param in self.clip.parameters():
            param.requires_grad = False
        self.processor = AutoProcessor.from_pretrained("clip-italian/clip-italian")
        self.device = device

    def embed_texts(self, texts):
        inputs = self.processor(text=texts, padding="longest", truncation=True)

        input_ids = torch.tensor(inputs["input_ids"]).to(self.device)
        attention_mask = torch.tensor(inputs["attention_mask"]).to(self.device)
        with torch.no_grad():
            embeddings = self.clip.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return embeddings

    def embed_images(self, images):
        inputs = self.processor(images=images)
        pixel_values = torch.tensor(np.array(inputs["pixel_values"])).to(self.device)
        with torch.no_grad():
            embeddings = self.clip.get_image_features(pixel_values=pixel_values)
        return embeddings
    
    def forward(self, texts, images):
        text_embeddings = self.embed_texts(texts)
        image_embeddings = self.embed_images(images)
        cosine_similarity = F.cosine_similarity(image_embeddings, text_embeddings, dim=1).unsqueeze(1)
        return text_embeddings, image_embeddings, cosine_similarity
    
class SELayer(nn.Module):
    def __init__(self):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((3,1))
        self.fc1 = nn.Sequential(
            nn.Linear(3,3),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(3,3),
            nn.GELU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x size in batch size x 3 x 1024
        avg = self.avg_pool(x).squeeze(2) # batch size x 3 
        y = self.fc1(avg) # batch size x 3 
        y = self.fc2(y) # batch size x 3 
        return x * y.unsqueeze(2)
    
class FNDModel(nn.Module):
    def __init__(self, bert_model, resnet_model, clip_model, norm_weight = 0.5):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.bert = BertParts(self.device, bert_model)
        self.resnet = ResnetModel(self.device, resnet_model)
        self.clip = ClipModel(self.device, clip_model)
        self.squeeze = SELayer()
        self.sentiment = BertSentiment(self.device)
        self.first_pass = True
        self.running_sim_mean = 0
        self.running_sim_std = 1
        self.sigmoid = nn.Sigmoid()
        self.norm_weight = norm_weight

        self.bert_projection_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.resnet_projection_head = nn.Sequential(
            nn.Linear(2560, 2560),
            nn.BatchNorm1d(2560),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2560, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.clip_projection_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(),
            nn.Linear(3072, 4),
        )


    def forward(self, texts, images):
        bert_embeddings = self.bert(texts)
        sentiment_embeddings = self.sentiment(texts)
        resnet_embeddings = self.resnet(images)
        clip_text_embeddings, clip_image_embeddings, clip_similarity = self.clip(texts, images)
        
        if self.first_pass == True:
            self.running_sim_mean = clip_similarity.mean().item()
            self.running_sim_std = clip_similarity.std().item()
            self.first_pass = False
        else:
            #update running mean and std
            self.running_sim_mean = self.running_sim_mean * self.norm_weight + clip_similarity.mean().item() * (1 - self.norm_weight)
            self.running_sim_std = self.running_sim_std * self.norm_weight + clip_similarity.std().item() * (1 - self.norm_weight)

        #normalize the clip similarity
        normalized_clip_similarity = (clip_similarity - self.running_sim_mean) / self.running_sim_std

        # apply sigmoid to the normalized clip similarity
        normalized_clip_similarity = self.sigmoid(normalized_clip_similarity)

        textual_embeddings = self.bert_projection_head(torch.cat([bert_embeddings, sentiment_embeddings ,clip_text_embeddings], dim=1)).unsqueeze(1)
        clip_embeddings = self.clip_projection_head(torch.cat([clip_text_embeddings, clip_image_embeddings], dim=1))
        visual_embeddings = self.resnet_projection_head(torch.cat([resnet_embeddings, clip_image_embeddings], dim=1)).unsqueeze(1)

        clip_embeddings = (clip_embeddings * normalized_clip_similarity).unsqueeze(1)

        #concatenate the features 
        concatenated_embeddings = torch.cat([textual_embeddings, visual_embeddings, clip_embeddings], dim=1) # batch size x 3 x 1024
        embeds = self.squeeze(concatenated_embeddings) # batch size x 3 x 1024

        batch_size = embeds.shape[0]
        embeds = embeds.view(batch_size, -1) # batch size x 3072

        logits = self.classifier(embeds)
        return logits
    
class FNDTrainer:
    def __init__(self, bert_model, resnet_model, clip_model):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = FNDModel(bert_model, resnet_model, clip_model).to(self.device)
        self.backtranslation = BackTranslationModule("./dict_files/tweet_translations.pt")

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
                        
                logits = self.model(
                    texts, images_list
                )

                # take the average of the logits
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
                total_labels += list(labels)
                #progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        return metrics
    
    def train(self, train_ds, eval_ds, lr = 1e-3, batch_size= 8, num_epochs = 30, warmup_steps = 0,weight_decay = 0.001, save_path = "./", loss= "cross_entropy"):
            
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
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay = weight_decay)
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
                    images_list = batch["images"]
                    labels = batch["label"]
                    nums_images = batch["nums_images"]
                    types = batch["type"]
                    ids = batch["id"]
                    #backtranslate the texts if label is different than 2 and type is tweet with probability 0.5
                    new_texts = []
                    for k in range(len(texts)):
                        current_text = texts[k]
                        current_label = labels[k]
                        current_type = types[k]
                        current_id = ids[k]
                        if current_label != 2 and current_type == "tweet" and random.random() < 0.5:
                            current_text = self.backtranslation(current_id)
                        new_texts.append(current_text)
                    texts = new_texts
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
                    
                    labels = torch.tensor(labels).to(self.device)

                    #nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=self.device)

                    logits = self.model(
                        texts = texts,
                        images = random_images_list
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
            
class BertParts(nn.Module):
  def __init__(self, device, pretrained_model_path = None):
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
    self.device = device

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

class BackTranslationModule():
    def __init__(self, path):
        super().__init__()
        self.translation_dict = torch.load(path)

    def __call__(self, id):
        return self.translation_dict[id]

if __name__ == "__main__":

    args = sys.argv
    # loss value
    loss = args[1]
    # learning rate value
    lr = float(args[2])
    # file to save the best perfroamnce model
    save_path = args[3]
    # weight decay value
    wd = float(args[4])
    # path to the tsv file of the dataset
    tsvpath = args[5]
    # path to the media folder of the dataset
    mediapath = args[6]
    # path to the pretrained bert model
    pretrained_bert_path = args[7]

    dataset = FakeNewsDataset(tsvpath, mediapath)
    train_ds, val_ds = stratifiedSplit(dataset)
    save_path = save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model = FNDTrainer(pretrained_bert_path, None, None)
    model.train(train_ds, val_ds, lr=lr, num_epochs = 80, batch_size=16,  save_path = save_path, loss=loss, weight_decay = wd)
