import pandas as pd
import pickle
import os

# Paths
data_path = "../data/machado.csv"
last_index_path = "../data/last_index.txt"

# Chunk size
chunk_size = 750*5  # Adjust the chunk size as needed

# Read last index from file if it exists
if os.path.exists(last_index_path):
    with open(last_index_path, "r") as file:
        last_index = int(file.read())
else:
    last_index = 0

# Initialize tokenizer and model
from torch.utils.data import Dataset, DataLoader,TensorDataset,SequentialSampler
from transformers import BertTokenizerFast, BertModel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import pre_tokenizers
import torch

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained("neuralmind/bert-base-portuguese-cased", model_max_length=512)
pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
tokenizer.pre_tokenizer = pre_tokenizer
model = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased").to(device)

class ChunkDataset(Dataset):
    def __init__(self, context, next_sentence,Y):
        # Process context sentences and next sentences in batches
        self.context = context
        self.next = next_sentence
        self.Y = torch.tensor(Y.values)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.context[idx], self.next[idx],self.Y[idx]

    def __add__(self, other):
        self.context = torch.cat([self.context, other.context])
        self.next    = torch.cat([self.next, other.next])
        self.Y = torch.cat([self.Y, other.Y])
        
        return self

def vectorize(values):
    print("Tokenizing...")
    encoded_data = tokenizer.batch_encode_plus(values, 
                                               add_special_tokens=False, 
                                               return_attention_mask=True, 
                                               padding='longest',
                                               truncation=True,
                                               max_length=256, 
                                               return_tensors='pt')

    input_ids       = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    print("Tokenized!")
    
    dataset    = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=750)

    context_embeddings = torch.empty([0,768])

    for batch in tqdm(dataloader):

        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                'attention_mask': batch[1],
                }
        with torch.no_grad():        
            outputs = model(**inputs)

        logits = outputs.last_hidden_state[:, 0, :]
        context_embeddings = torch.vstack([context_embeddings, logits.detach().cpu()])
    return context_embeddings

# Function to process and save each chunk
def process_chunk(chunk, last_index):
    # Tokenize and vectorize the chunk
    context = vectorize(list(chunk["Context"].values))
    next_sentence = vectorize(list(chunk["Next_Sentence"].values))
    Y = chunk["is_same_paragraph"]

    # Create a ChunkDataset
    chunk_dataset = ChunkDataset(context, next_sentence, Y)

    # Define file paths for test and train data
    train_data_path = '../data/data.pkl'

    # Save the dataset to file
    with open(train_data_path, 'ab') as f:
        pickle.dump(chunk_dataset, f)

    # Update the last index
    last_index += len(chunk)

    # Save the last index to a file
    with open(last_index_path, "w") as file:
        file.write(str(last_index))

# Read the data in chunks starting from the last index
for chunk in pd.read_csv(data_path, chunksize=chunk_size, skiprows=range(1, last_index + 1)):
    print("Chunk",last_index)
    process_chunk(chunk, last_index)
    last_index += len(chunk)
