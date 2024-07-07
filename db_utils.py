import qdrant_client
from qdrant_client import models
from llama_index.core import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_dataloader import create_embeddings
import pandas as pd
from datetime import datetime
import json

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)


def get_point_content(point):
    payload = point[0][0].payload
    node_content = json.loads(payload.get('_node_content'))
    return node_content


def read_csv(csv_file):
    return pd.read_csv(csv_file)


def create_text(row):
    return f"{row['Questions']} \n {row['Answers']}"


def get_point(client, collection_name, id):
    return client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="id", match=models.MatchValue(value=id)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )


def delete_point(client, collection_name, id):
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=id),
                    ),
                ],
            )
        ),
    )


def create_document(text, id):
    metadata = {
        "id": id,
        'datetime': datetime.now().isoformat()
    }
    return Document(text=text, metadata=metadata)


def update_docs(collection_name, csv_file):
    df = read_csv(csv_file)
    documents = []

    for _, row in df.iterrows():
        text = create_text(row)
        id = row['Question_ID']
        point = get_point(client, collection_name, id)

        if point[0]:
            node_content = get_point_content(point)
            if node_content.get('text') != text:
                delete_point(client, collection_name, id)
                doc = create_document(text, id)
                documents.append(doc)
        else:
            continue

    if documents:
        nodes = create_embeddings(documents)
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        vector_store.add(nodes)



if __name__ == '__main__':
    update_docs(collection_name='hailey-voice', csv_file='dataset/Mental_Health_FAQ.csv')