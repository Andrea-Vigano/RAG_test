import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.core import SimpleDirectoryReader
import os

print("Starting execution")
ANTHROPIC_API_KEY = st.secrets.anthropyic_api_key
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

VOYAGE_API_KEY = st.secrets.voyage_api_key
tokenizer = Anthropic().tokenizer
Settings.tokenizer = tokenizer

st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        print("Loading data...")
        reader = SimpleDirectoryReader(input_dir="./data/test", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=Anthropic(model="claude-3-opus-20240229"), embed_model=VoyageEmbedding(model_name="voyage-law-2", voyage_api_key=VOYAGE_API_KEY))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

