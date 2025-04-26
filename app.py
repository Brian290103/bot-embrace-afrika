import os
from flask import Flask, request, jsonify

from settings import (
    GOOGLE_API_KEY,
    PINECONE_API_KEY,
    INDEX_NAME,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    SYSTEM_INSTRUCTIONS_EMBRACE_AFRIKA
)

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# LlamaIndex and Pinecone Imports
from pinecone import Pinecone
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate

# Flask app
app = Flask(__name__)

# Initialize query engine
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

    prompt = PromptTemplate(
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

    return index.as_query_engine(text_qa_template=prompt)

query_engine = load_query_engine()

# API Documentation Route
@app.route("/", methods=["GET"])
def docs():
    return """
    <h2>🌍 Embrace Afrika AI Chatbot API</h2>
    <p>Use this REST API to query Embrace Afrika's knowledge base.</p>
    <h3>📮 POST /query</h3>
    <p><strong>Body:</strong> JSON with a <code>query</code> field.</p>
    <pre>
    {
        "query": "What is Embrace Afrika?"
    }
    </pre>

    <h3>📦 Sample CURL</h3>
    <pre>
    curl -X POST http://localhost:5000/query \\
         -H "Content-Type: application/json" \\
         -d '{"query": "What is Embrace Afrika?"}'
    </pre>

    <h3>💻 Sample JavaScript (fetch)</h3>
    <pre>
    fetch('http://localhost:5000/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: 'What is Embrace Afrika?'})
    })
    .then(res => res.json())
    .then(data => console.log(data))
    </pre>

    <h3>📜 Sample JavaScript (Axios)</h3>
    <pre>
    axios.post('http://localhost:5000/query', {
      query: "What is Embrace Afrika?"
    }).then(res => {
      console.log(res.data);
    });
    </pre>
    """, 200

# AI Query Endpoint
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query_text = data.get('query')

    if not query_text:
        return jsonify({'error': 'Missing query'}), 400

    response = query_engine.query(query_text)
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(debug=True)
