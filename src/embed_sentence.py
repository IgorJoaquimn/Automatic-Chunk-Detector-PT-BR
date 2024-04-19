from transformers import BertTokenizer, BertModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased").to(device)

def embed_sentences(sentences):
    sentences_with_cls = ["[CLS] " + sentence for sentence in sentences]
    
    tokenized_inputs = tokenizer(sentences_with_cls, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    
    sentence_embeddings = outputs.last_hidden_state[:, 0, :]
    
    return sentence_embeddings.cpu()