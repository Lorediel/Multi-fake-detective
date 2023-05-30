from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from FakeNewsDataset import FakeNewsDataset
import torch.nn as nn
import random
from tqdm.auto import tqdm
import sys
# Backtranslates in english and saves it to a dict
class BackTranslationModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer_it_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
        self.model_it_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en")

        self.tokenizer_en_it = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
        self.model_en_it = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")

        self.model_it_en.to(device)
        self.model_en_it.to(device)
        self.model_it_en.eval() 
        self.model_en_it.eval()
        for param in self.model_it_en.parameters():
            param.requires_grad = False
        for param in self.model_en_it.parameters():
            param.requires_grad = False

    def forward(self, text, type="tweet"):
        with torch.no_grad():
            if (type=="tweet"):
                # translate from it to en
                batch = self.tokenizer_it_en([text], return_tensors="pt", truncation = True, padding=True)
                for k, v in batch.items():
                    batch[k] = v.to("cuda")
                generated_ids = self.model_it_en.generate(**batch)
                eng_texts = self.tokenizer_it_en.batch_decode(generated_ids, skip_special_tokens=True)
                
                # back translate
                batch_eng = self.tokenizer_en_it(eng_texts, return_tensors="pt", truncation=True, padding=True)
                for k, v in batch_eng.items():
                    batch_eng[k] = v.to("cuda")
                generated_ids = self.model_en_it.generate(**batch_eng)
                it_text = self.tokenizer_en_it.batch_decode(generated_ids, skip_special_tokens=True)
                return it_text[0]
            elif (type=="article"):
                batch = self.tokenizer_it_en([text], return_tensors="pt")
                for k, v in batch.items():
                    batch[k] = v.to("cuda")
                input_ids = list(batch.input_ids[0])
                attention_mask = list(batch.attention_mask[0])
                new_input_ids = []
                new_attention_mask = []
                if (len(input_ids) > self.max_len):
                    i_chunks = list(chunks(input_ids, self.max_len-1))
                    a_chunks = list(chunks(attention_mask, self.max_len-1))
                    num_chunks = len(i_chunks)
                    # add 0 at the end and pad the last chunk
                    for i in range(num_chunks):
                        i_chunks[i].append(0)
                        a_chunks[i].append(1)
                        if i == num_chunks - 1:
                            i_chunks[i] = i_chunks[i] + ([0] * (self.max_len - len(i_chunks[i])))
                            a_chunks[i] = a_chunks[i] + ([0] * (self.max_len - len(a_chunks[i])))
                    new_input_ids = torch.tensor(i_chunks).to(self.device)
                    new_attention_mask = torch.tensor(a_chunks).to(self.device)
                else:
                    new_input_ids = batch.input_ids
                    new_attention_mask = batch.attention_mask

                generated_ids = self.model_it_en.generate(input_ids = new_input_ids, attention_mask = new_attention_mask)
                eng_texts = self.tokenizer_it_en.batch_decode(generated_ids, skip_special_tokens=True)

                # back translate
                batch_eng = self.tokenizer_en_it(eng_texts, return_tensors="pt", padding=True)
                for k, v in batch_eng.items():
                    batch_eng[k] = v.to("cuda")
                generated_ids = self.model_en_it.generate(**batch_eng)
                it_text = self.tokenizer_en_it.batch_decode(generated_ids, skip_special_tokens=True)
                final_article = " ".join(it_text)
                return final_article
                
class BackTranslationModule4Lang(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.max_len = 512
        self.model_dict = {
            "en": {
                "translate_tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en"),
                "translate_model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en"),
                "backtranslate_tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it"),
                "backtranslate_model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")
            },
            "lt": {
                "translate_tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-lt"),
                "translate_model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-lt"),
                "backtranslate_tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-lt-it"),
                "backtranslate_model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-lt-it")
            },
            "es": {
                "translate_tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-es"),
                "translate_model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-es"),
                "backtranslate_tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-it"),
                "backtranslate_model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-it")
            },
            "de": {
                "translate_tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-de"),
                "translate_model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-de"),
                "backtranslate_tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-it"),
                "backtranslate_model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-it")
            }
            
        }
        
    def forward(self, text, type="tweet"):
        with torch.no_grad():
            # choose randomly a language from EN, FR, DE, ES
            lang = random.choice(["en", "lt", "de", "es"])
            translate_tokenizer = self.model_dict[lang]["translate_tokenizer"]
            translate_model = self.model_dict[lang]["translate_model"].to(self.device).eval()
            backtranslate_tokenizer = self.model_dict[lang]["backtranslate_tokenizer"]
            backtranslate_model = self.model_dict[lang]["backtranslate_model"].to(self.device).eval()
            for param in translate_model.parameters():
                param.requires_grad = False
            for param in backtranslate_model.parameters():
                param.requires_grad = False

            # translate from it to en
            if (type == "tweet"):
                batch = translate_tokenizer([text], return_tensors="pt", truncation = True, padding=True)
                for k, v in batch.items():
                    batch[k] = v.to("cuda")
                generated_ids = translate_model.generate(**batch)
                eng_texts = translate_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # back translate
                batch_eng = backtranslate_tokenizer(eng_texts, return_tensors="pt", truncation=True, padding=True)
                for k, v in batch_eng.items():
                    batch_eng[k] = v.to("cuda")
                generated_ids = backtranslate_model.generate(**batch_eng)
                it_text = backtranslate_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                return it_text[0]
            elif (type=="article"):
                batch = translate_tokenizer([text], return_tensors="pt")
                for k, v in batch.items():
                    batch[k] = v.to("cuda")
                input_ids = list(batch.input_ids[0])
                attention_mask = list(batch.attention_mask[0])
                new_input_ids = []
                new_attention_mask = []
                if (len(input_ids) > self.max_len):
                    i_chunks = list(chunks(input_ids, self.max_len-1))
                    a_chunks = list(chunks(attention_mask, self.max_len-1))
                    num_chunks = len(i_chunks)
                    # add 0 at the end and pad the last chunk
                    for i in range(num_chunks):
                        i_chunks[i].append(0)
                        a_chunks[i].append(1)
                        if i == num_chunks - 1:
                            i_chunks[i] = i_chunks[i] + ([0] * (self.max_len - len(i_chunks[i])))
                            a_chunks[i] = a_chunks[i] + ([0] * (self.max_len - len(a_chunks[i])))
                    new_input_ids = torch.tensor(i_chunks).to(self.device)
                    new_attention_mask = torch.tensor(a_chunks).to(self.device)
                else:
                    new_input_ids = batch.input_ids
                    new_attention_mask = batch.attention_mask

                generated_ids = translate_model.generate(input_ids = new_input_ids, attention_mask = new_attention_mask)
                eng_texts = translate_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # back translate
                batch_eng = backtranslate_tokenizer(eng_texts, return_tensors="pt", padding=True)
                for k, v in batch_eng.items():
                    batch_eng[k] = v.to("cuda")
                generated_ids = backtranslate_model.generate(**batch_eng)
                it_text = backtranslate_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                final_article = " ".join(it_text)
                return final_article

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    args = sys.argv
    # path to save the translations
    save_path = args[1]
    # path to the dataset tsv file
    tsvpath = args[2]
    # path to the media folder
    mediapath = args[3]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = FakeNewsDataset(tsvpath, mediapath)
    translations = {}
    m = BackTranslationModule(device)
    i=0
    progress_bar = tqdm(len(dataset))
    for sample in dataset:
        label = sample["label"]
        id = sample["id"]
        text = sample["text"]
        t = sample["type"]    
        if t != "tweet":
            continue
        translation = m(text)
        #print(translation)
        translations[id] = translation
        i+=1
        progress_bar.update(1)
        if i==908:
            break
    torch.save(translations, save_path)
