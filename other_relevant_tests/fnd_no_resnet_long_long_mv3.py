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
import timm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class LongClipModule(nn.Module):
  def __init__(self, pretrained_model_path = None):
    super().__init__()
    self.base_model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
    if (pretrained_model_path != None):
        s = torch.load(pretrained_model_path)
        new_s = {}
        for n in s:
            if (n.startswith("base_model")):
                new_s[n[11:]] = s[n]
        self.base_model.load_state_dict(new_s)
    self.tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")
    self.max_len = 512
    for param in self.base_model.parameters():
       param.requires_grad = False
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def forward(self, texts):
    tokens_list = self.tokenizer(texts).input_ids    
    with torch.no_grad():
        masks = []
        num_sublists = []
        divided_tokens = []

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
        bertOutput = self.base_model.get_text_features(input_ids, attention_masks)


        base = 0
        final = []
        for i in range(len(num_sublists)):
            tensors = bertOutput[base:base+num_sublists[i]]
            mean_tensor = torch.mean(tensors, dim = 0)

            final.append(mean_tensor)
            base += num_sublists[i]
        final = torch.stack(final, dim=0).to(self.device)
        
        return final

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
    bertOutput = self.bert(input_ids, attention_masks).pooler_output


    base = 0
    final = []
    for i in range(len(num_sublists)):
      tensors = bertOutput[base:base+num_sublists[i]]
      mean_tensor = torch.mean(tensors, dim = 0)

      final.append(mean_tensor)
      base += num_sublists[i]
    final = torch.stack(final, dim=0).to(self.device)
    
    return final
    
class LongSentiment(nn.Module):
  def __init__(self, device):
      super().__init__()
      model_path = "neuraly/bert-base-italian-cased-sentiment"
      self.model = AutoModel.from_pretrained(model_path)
      for param in self.model.parameters():
          param.requires_grad = False
      self.tokenizer = AutoTokenizer.from_pretrained(model_path)
      self.device = device
      self.max_len = 512
      self.model.eval()

  def forward(self, texts):
    tokens_list = self.tokenizer(texts).input_ids
    masks = []
    num_sublists = []
    divided_tokens = []
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
    with torch.no_grad():
        bertOutput = self.model(input_ids, attention_masks).pooler_output

    base = 0
    final = []
    for i in range(len(num_sublists)):
      tensors = bertOutput[base:base+num_sublists[i]]
      mean_tensor = torch.mean(tensors, dim = 0)

      final.append(mean_tensor)
      base += num_sublists[i]
    final = torch.stack(final, dim=0).to(self.device)
    
    return final

class ClipModel(nn.Module):

    def __init__(self, device, clip_model = None):
        super().__init__()
        self.clip = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        self.longClip = LongClipModule(clip_model)
        if (clip_model != None):
            s = torch.load(clip_model)
            new_s = {}
            for n in s:
                if (n.startswith("base_model")):
                    new_s[n[11:]] = s[n]
            self.clip.load_state_dict(new_s)
        for name, param in self.clip.state_dict().items():
            param.required_grad = False
        
        self.processor = AutoProcessor.from_pretrained("clip-italian/clip-italian")
        self.device = device

    def embed_texts(self, texts):
        return self.longClip(texts)

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
        avg = self.avg_pool(x).squeeze(2) # batch size x 3 x 1
        y = self.fc1(avg) # batch size x 3 x 1
        y = self.fc2(y) # batch size x 3 x 1
        return x * y.unsqueeze(2)
    
class MobileNetModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=4)

    def forward(self, images):
        out = self.model(images)
        return out
    
class FNDModel(nn.Module):
    def __init__(self, bert_model, resnet_model, clip_model, norm_weight=0.5):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.bert = BertParts(bert_model)
        self.sentiment = LongSentiment(self.device)
        self.clip = ClipModel(self.device, clip_model)
        self.squeeze = SELayer()
        self.mobilenet = MobileNetModule(self.device)
        self.first_pass = True
        self.running_sim_mean = 0
        self.running_sim_std = 1
        self.sigmoid = nn.Sigmoid()
        self.norm_weight = norm_weight

        self.bert_projection_head = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.BatchNorm1d(1280),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
        )

        self.sentiment_projection_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
        )

        self.clip_projection_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
        )
        
        


    def forward(self, texts, images):
        bert_embeddings = self.bert(texts)
        sentiment_embeddings = self.sentiment(texts) # batch size x 768
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

        normalized_clip_similarity = self.sigmoid(normalized_clip_similarity)

        textual_embeddings = self.bert_projection_head(torch.cat([bert_embeddings, clip_text_embeddings], dim=1)).unsqueeze(1)
        clip_embeddings = self.clip_projection_head(torch.cat([clip_text_embeddings, clip_image_embeddings], dim=1))
        sentiment_embeddings = self.sentiment_projection_head(sentiment_embeddings).unsqueeze(1) # batch size x 1 x 1024

        clip_embeddings = (clip_embeddings * normalized_clip_similarity).unsqueeze(1)

        #concatenate the features 
        concatenated_embeddings = torch.cat([textual_embeddings, sentiment_embeddings, clip_embeddings], dim=1)
        embeds = self.squeeze(concatenated_embeddings) # batch size x 3 x 1024
        
        # reshape to input the mobilenet
        batch_size, _, _ = embeds.size()
        embeds = torch.reshape(embeds, (batch_size, 3, 32, 32))
        logits = self.mobilenet(embeds)
        return logits
    
class FNDTrainer:
    def __init__(self, bert_model, resnet_model, clip_model):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = FNDModel(bert_model, resnet_model, clip_model).to(self.device)

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
    
    def train(self, train_ds, eval_ds, lr = 1e-3, batch_size= 8, num_epochs = 30, weight_decay = 0.0001, warmup_steps = 0, save_path = "./", loss=None):
            
            dataloader = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
            )
            if loss == "None":
                criterion = nn.CrossEntropyLoss()
            elif loss == "focal":
                criterion = FocalLoss(gamma=2, reduction='mean')

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
            best_metrics = [0, 0, 0, 0, 0]
            print("accuracy | precision | recall | f1 | f1_weighted | f1_for_each_class")
            for epoch in range(num_epochs):
                for batch in dataloader:
                    current_step += 1
                    texts = batch["text"]
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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

