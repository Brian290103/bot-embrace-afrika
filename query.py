import streamlit as st
from settings import GOOGLE_API_KEY, PINECONE_API_KEY, INDEX_NAME, NOTION_TOKEN, LLM_MODEL, EMBEDDING_MODEL, CHUNK_SIZE, \
    SYSTEM_INSTRUCTIONS_EMBRACE_AFRIKA

# LlamaIndex and Pinecone Imports
from pinecone import Pinecone
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate


custom_prompt_template = PromptTemplate(
    template=f"""
        System Instructions: {SYSTEM_INSTRUCTIONS_EMBRACE_AFRIKA}
        ---------------------
        Context Information:
        ---------------------
        {{context_str}}
        ---------------------
        Based on the context above, answer the query below.

        Query: {{query_str}}
        Answer:
        """,
    prompt_type="query",
)

# Setup and cache components
@st.cache_resource
def load_query_engine():
    llm = GoogleGenAI(model=LLM_MODEL, api_key=GOOGLE_API_KEY)
    embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE

    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pinecone_client.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(text_qa_template=custom_prompt_template)

    return query_engine

# Streamlit UI
st.set_page_config(page_title="Embrace Afrika Assistant", layout="centered")
st.title("🌍 Embrace Afrika Chat Assistant")

query_text = st.text_input("Ask a question about Embrace Afrika's focus areas (Excellence, Diversity, Beauty, Fashion, Tourism, Culture) or events:")

if query_text:
    with st.spinner("Thinking..."):
        query_engine = load_query_engine()
        response = query_engine.query(query_text)
        st.success("Answer:")
        st.write(str(response))