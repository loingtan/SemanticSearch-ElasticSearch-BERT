import time

import torch
from elasticsearch import Elasticsearch
import streamlit as st
from transformers import AutoTokenizer, AutoModel


@st.cache_resource
def load_es():
    model_embedding = AutoModel.from_pretrained("vinai/phobert-base-v2")
    client = Elasticsearch(hosts="http://localhost:9200")
    return model_embedding, client


def embed_text(batch_text, tokenizer, model):
    tokenized = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        outputs = model(**tokenized)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.numpy().tolist()


def search(query, type_ranker, tokenizer=None, model=None):
    if type_ranker == 'BERT':
        time_embed = time.time()
        query_vector = embed_text([query], tokenizer, model)[0]
        print(f'TIME EMBEDDING: {time.time() - time_embed} seconds')
        script_query = {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    else:  # BM25
        script_query = {
            "match": {
                "title": {
                    "query": query,
                    "fuzziness": "AUTO"
                }
            }
        }

    response = client.search(
        index='demo_simcse',
        body={
            "size": 10,
            "query": script_query,
            "_source": {
                "includes": ["id", "title"]
            },
        },
        ignore=[400]
    )

    result = [hit["_source"]["title"] for hit in response.get("hits", {}).get("hits", [])]
    return result


def run():
    st.title('Semantic search with BERT and Elasticsearch')
    st.markdown('Compare BM25 and BERT-based semantic search results.')

    input_text = st.text_input('Write your test content:')

    if st.button('Search'):
        if input_text.strip():
            with st.spinner('Searching...'):
                bm25_results = search(input_text, 'BM25')
                tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
                bert_results = search(input_text, 'BERT', tokenizer, model_embedding)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('BM25 Results')
                if bm25_results:
                    for result in bm25_results:
                        st.success(result)
                else:
                    st.warning('No results found for BM25.')

            with col2:
                st.subheader('BERT Results')
                if bert_results:
                    for result in bert_results:
                        st.success(result)
                else:
                    st.warning('No results found for BERT.')
        else:
            st.warning('Please enter some text to search.')


if __name__ == '__main__':
    model_embedding, client = load_es()
    run()
