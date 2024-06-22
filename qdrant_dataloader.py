from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.settings import Settings

import pandas as pd
import qdrant_client
from dotenv import load_dotenv
load_dotenv('/home/sunilsamsonsuresh/Documents/Study/Github/Hailey-Voice-Assistant/.env')


# embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

#
embed_model = OpenAIEmbedding(model_name="text-embedding-ada-002")
csv_file = pd.read_csv("dataset/Mental_Health_FAQ.csv")

client = qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )

collection_name = 'hailey-voice'


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


def create_embeddings(documents):


    parser = SentenceSplitter()
    if documents:
        nodes = parser.get_nodes_from_documents(documents)
        for node in nodes:
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode='all')
            )
            node.embedding = node_embedding

    return nodes


def add_docs(client, csv_file, collection_name):


    documents = create_documents(csv_file)
    nodes = create_embeddings(documents)


    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    vector_store.add(nodes)


if __name__ == "__main__":
    # add_docs(client, csv_file, collection_name)

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine()

    res = query_engine.query(input("Ask Me: "))
    print(res)