from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Document
from utils import get_vector_store
import pandas as pd
from datetime import datetime
import qdrant_client
from dotenv import dotenv_values

config = dotenv_values(".env")

api_key = config['OPENAI_API_KEY']

csv_file = pd.read_csv("dataset/Mental_Health_FAQ.csv")
csv_file = csv_file[:5]

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
            "id": row['Question_ID'],
            'datetime': datetime.now().isoformat()
        }

        doc = Document(
            text=content,
            metadata=metadata,
        )

        documents.append(doc)

    return documents


def create_embeddings(documents):

    embed_model = OpenAIEmbedding(model_name="text-embedding-ada-002", api_key=api_key)

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