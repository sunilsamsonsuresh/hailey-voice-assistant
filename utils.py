import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()


def get_vector_store():
    client = qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )

    collection_name = 'hailey-voice'

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(vector_store)

    return index


def get_chat_engine(session_id):
    if "memory" not in st.session_state or st.session_state.session_id != session_id:
        st.session_state.session_id = session_id
        st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    vector_index = get_vector_store()
    retriever = vector_index.as_retriever()

    llm = OpenAI(model="gpt-4-1106-preview", temperature=0, max_tokens=250)

    system_prompt = '''
    You are helpful AI Assistant for mental health awareness, based on users questions \n
    provide an suitable response in under 120 words.
    '''

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        system_prompt=system_prompt,
        memory=st.session_state.memory,
        verbose=False
    )

    return chat_engine
