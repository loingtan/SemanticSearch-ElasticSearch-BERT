import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
index_name = "nf_corpus"
path_index = "config/index.json"
path_data = "data/nfcorpus/corpus.jsonl"
batch_size = 128

def embed_text(batch_text, tokenizer, model):
    tokenized = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        outputs = model(**tokenized)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.numpy().tolist()
def index_batch(docs, client, tokenizer, model):
    requests = []
    try:
        descriptions = [(doc["title"] +" "+ doc["content"]) for doc in docs]
        descriptions_vectors = embed_text(descriptions, tokenizer, model)
        for i, doc in enumerate(docs):
            request = {
                "_op_type": "index",
                "_index": index_name,
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "descriptions_vector": descriptions_vectors[i],
            }
            requests.append(request)
        bulk(client, requests)
    except Exception as e:
        print(f"Error while indexing batch: {e}")

if __name__ == "__main__":
    client = Elasticsearch(hosts="http://localhost:9200")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
    print(f"Creating the {index_name} index.")
    client.indices.delete(index=index_name, ignore=[404])
    with open(path_index) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=index_name, body=source)

    docs = []
    count = 0
    with open(path_data, 'r') as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame(data)
    for _, row in df.iterrows():
        count += 1
        item = {
            'id': row['_id'],
            'title': row['title'],
            'content': row['text']
        }
        docs.append(item)
        if count % batch_size == 0:
            index_batch(docs, client, tokenizer, model)
            docs = []
            print(f"Indexed {count} documents.")


    if docs:
        index_batch(docs, client, tokenizer, model)
        print(f"Indexed {count} documents.")


    client.indices.refresh(index=index_name)
    print("Done indexing.")
