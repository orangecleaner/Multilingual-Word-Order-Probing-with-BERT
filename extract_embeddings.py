from transformers import AutoTokenizer, AutoModel
import torch
import json
import numpy as np
from tqdm import tqdm

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model= AutoModel.from_pretrained(model_name,output_hidden_states=True)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_file="word_order.json"
samples=[]
with open(input_file,'r') as f:
    for line in f:
        samples.append(json.loads(line))

results = {f"layer_{i}": [] for i in range(model.config.num_hidden_layers + 1)}
labels=[]

with torch.no_grad():
    for sample in tqdm(samples):
        text=sample["sentence"]
        input=tokenizer(text,return_tensors="pt")
        inputs = {k: v.to(device) for k, v in input.items()}
        outputs=model(**inputs)
        hidden_states=outputs.hidden_states
        for i, layer_output in enumerate(hidden_states):
            #get cls token
            cls_vector = layer_output[0, 0, :].cpu().numpy()
            results[f"layer_{i}"].append(cls_vector)
        
        labels.append(sample["label"])
            
np.savez("sentence_embeddings.npz", labels=labels, **results)