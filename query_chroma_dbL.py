import os
import openai
import streamlit as st
import chromadb

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure ChromaDB client
CHROMA_PATH = "chroma"
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection")

# Function to perform semantic search
def semantic_search_query(query, size=5):
    results = collection.query(
        query_texts=[query],
        n_results=size
    )
    return results

# Function to generate human-like responses using OpenAI API
def generate_human_like_response(query, results):
    response_texts = []
    for i in range(len(results['documents'])):
        doc_text = results['documents'][i]
        prompt = f"Based on the following information:\n\n{doc_text}\n\nQ: {query}\nA:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300  # Increase the max_tokens to ensure more complete responses
        )
        response_texts.append(response['choices'][0]['message']['content'].strip())
    return response_texts

# Streamlit application
st.set_page_config(page_title="ChromaDB Semantic Search L", layout="wide")
st.title("AI Suggestions for Your Stress Management")

# User input for query
query = "suggest stress management for low stress level"

# Perform semantic search
model_name = "sentence-transformers/all-mpnet-base-v2"
results = semantic_search_query(query)

# Display search results
#st.write(f"Search Results for query '{query}':")
if results['documents']:
    human_like_responses = generate_human_like_response(query, results)
    for i in range(len(results['documents'])):
        doc_id = results['ids'][i]
        doc_text = results['documents'][i]
        metadata = results['metadatas'][i]
        filename = metadata[0]['filename'] if isinstance(metadata, list) and metadata else 'Unknown'
        #st.write(f"**Source:** {filename}")
        #st.write(f"**Document ID:** {doc_id}")
        st.write(f"**Human-like Response:**")
        st.write(human_like_responses[i])
        st.write("---")
        #st.write(f"**Source:** {filename}")
else:
    st.write("No documents found.")