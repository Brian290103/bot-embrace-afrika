# train.py
import os
import streamlit as st

from settings import GOOGLE_API_KEY, PINECONE_API_KEY, NOTION_TOKEN,INDEX_NAME,CHUNK_SIZE,LLM_MODEL,EMBEDDING_MODEL

# ─── Set env vars ───────────────────────────────────────────────────────────────
os.environ["GOOGLE_API_KEY"]   = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["NOTION_TOKEN"]     = NOTION_TOKEN

# ─── Imports ───────────────────────────────────────────────────────────────────
from pinecone import Pinecone
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.readers.notion import NotionPageReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings

# ─── Cache & init shared resources ─────────────────────────────────────────────
@st.cache_resource
def init_resources():
    # LLM & embedding
    llm = GoogleGenAI(model=LLM_MODEL, api_key=GOOGLE_API_KEY)
    embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE

    # Pinecone vector store
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pinecone_client.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # StorageContext for indexing
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

# ─── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Embrace Afrika Trainer", layout="centered")
st.title("🌍 Embrace Afrika Bot Trainer")

storage_context = init_resources()

# — Train via URL —
st.subheader("1. Train from a Website URL")
url = st.text_input(
    label="Enter a single website URL",
    placeholder="https://example.com/..."
)
if st.button("Train via URL"):
    if not url:
        st.error("Please enter a valid URL.")
    else:
        loader = BeautifulSoupWebReader()
        with st.spinner("Fetching and indexing website…"):
            docs = loader.load_data(urls=[url])
            print("docs")
            print(docs)
            VectorStoreIndex.from_documents(
                documents=docs,
                storage_context=storage_context
            )
        st.success("✅ Website URL trained successfully!")

st.markdown("---")

# — Train via Notion Page —
st.subheader("2. Train from a Notion Page")
page_id = st.text_input(
    label="Enter Notion Page ID",
    placeholder="abcdef12-3456-7890-abcd-ef1234567890"
)
if st.button("Train via Notion"):
    if not page_id:
        st.error("Please enter a Notion page ID.")
    else:
        reader = NotionPageReader(integration_token=NOTION_TOKEN)
        with st.spinner("Fetching and indexing Notion page…"):
            docs = reader.load_data(page_ids=[page_id])
            print("docs")
            print(docs)
            VectorStoreIndex.from_documents(
                documents=docs,
                storage_context=storage_context
            )
        st.success("✅ Notion page trained successfully!")
