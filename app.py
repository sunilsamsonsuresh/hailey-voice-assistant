import uuid
import streamlit as st
from llama_index.core.memory import ChatMemoryBuffer
from utils import get_chat_engine

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

with st.sidebar:
    st.write(f'Session ID: {st.session_state.session_id}')

st.title("I'm Hailey - your personal voice assistant ")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

session_id = st.session_state.session_id

if "messages" not in st.session_state:
    st.session_state.messages = []


chat_engine = get_chat_engine(session_id)

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        chat_engine.reset()

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            text_response = response.response
            st.write(text_response)
            message = {"role": "assistant", "content": text_response}
            st.session_state.messages.append(message)