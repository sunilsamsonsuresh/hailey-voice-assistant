from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI

from redis import Redis
from redisvl.schema import IndexSchema
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

import pandas as pd

from dotenv import load_dotenv
load_dotenv('/home/sunilsamsonsuresh/Documents/Study/Github/Hailey-Voice-Assistant/.env')

# redis_client = Redis.from_url("redis://127.0.0.1:6379")
redis_client = Redis.from_url("redis://localhost:6379")


embed_model = OpenAIEmbedding(model_name="text-embedding-ada-002")
dataset = pd.read_csv("dataset/Mental_Health_FAQ.csv")

custom_schema = IndexSchema.from_dict(
    {
        # customize basic index specs
        "index": {
            "name": "mental_health",
            "prefix": "hailey_voice",
            "key_separator": ":",
        },
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom metadata fields
            # custom vector field definition for cohere embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 1536,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    }
)

def create_documents(csv_file):
    trial_data = csv_file[:3]
    documents = []
    for ind, row in trial_data.iterrows():
        content = f"{row['Questions']} \n {row['Answers']}"

        metadata = {
            "id": row['Question_ID']
        }

        doc = Document(
            text=content,
            metadata=metadata,
        )

        documents.append(doc)

    return documents


def create_vectors(documents):


    vector_store = RedisVectorStore(
        schema=custom_schema,  # provide customized schema
        redis_client=redis_client,
        overwrite=True,
    )

    parser = SentenceSplitter()
    if documents:
        nodes = parser.get_nodes_from_documents(documents)
        for node in nodes:
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode='all')
            )
            node.embedding = node_embedding

    vector_store.add(nodes)


def get_vector_index():
    vector_store = RedisVectorStore(
        schema=custom_schema,  # provide customized schema
        redis_client=redis_client,
        overwrite=True,
    )
    vector_db = VectorStoreIndex.from_vector_store(vector_store)

    return vector_db


if __name__ == '__main__':

    documents = create_documents(dataset)
    index = create_vectors(documents)


    # vector_db = get_vector_index()
    #
    # retriever = VectorIndexRetriever(index=vector_db)
    #
    # query_engine = RetrieverQueryEngine(
    #     retriever=retriever
    # )
    # res = query_engine.query("Who does it mean to have mental illness?")
    # print(res)


