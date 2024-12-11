from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch

# Constants
index_name = "demo_simcse"
path_index = "config/index.json"
path_data = "data/data_title.csv"
batch_size = 128

# Function to embed text
def embed_text(batch_text, tokenizer, model):
    tokenized = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        outputs = model(**tokenized)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.numpy().tolist()

# Function to index a batch of documents
def index_batch(docs, client, tokenizer, model):
    requests = []
    try:
        titles = [(doc["title"]) for doc in docs]
        title_vectors = embed_text(titles, tokenizer, model)
        for i, doc in enumerate(docs):
            request = {
                "_op_type": "index",
                "_index": index_name,
                "id": doc["id"],
                "title": doc["title"],
                "title_vector": title_vectors[i],
            }
            requests.append(request)
        bulk(client, requests)
    except Exception as e:
        print(f"Error while indexing batch: {e}")

if __name__ == "__main__":
    # Initialize Elasticsearch client
    client = Elasticsearch(hosts="http://localhost:9200")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModel.from_pretrained("vinai/phobert-base-v2")

    # Create or reset the Elasticsearch index
    print(f"Creating the {index_name} index.")
    client.indices.delete(index=index_name, ignore=[404])
    with open(path_index) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=index_name, body=source)

    # Read and process data
    docs = []
    count = 0
    df = pd.read_csv(path_data).fillna(' ')
    for _, row in df.iterrows():
        count += 1
        item = {
            'id': row['id'],
            'title': row['title']
        }
        docs.append(item)

        # Index in batches
        if count % batch_size == 0:
            index_batch(docs, client, tokenizer, model)
            docs = []
            print(f"Indexed {count} documents.")

    # Index remaining documents
    if docs:
        index_batch(docs, client, tokenizer, model)
        print(f"Indexed {count} documents.")

    # Refresh index
    client.indices.refresh(index=index_name)
    print("Done indexing.")
