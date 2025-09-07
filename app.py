import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@st.cache_resource
def load_vector_database():
    try:
        with open('vector_database.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Vector database not found. Please run 'uv run python generate_vectors.py' first.")
        return None

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def find_top_answers(question, vector_db, model, top_k=5):
    query_embedding = model.encode([question])
    similarities = cosine_similarity(query_embedding, vector_db['embeddings'])[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'question': vector_db['questions'][idx],
            'answer': vector_db['answers'][idx],
            'confidence': similarities[idx]
        })
    
    return results

def generate_llm_response(user_question, relevant_entries, threshold):
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Filter entries above threshold
        filtered_entries = [entry for entry in relevant_entries if entry['confidence'] >= threshold]
        
        if not filtered_entries:
            return "I don't have enough relevant information to answer your question confidently.", False
        
        # Build context from relevant entries
        context = "\n\n".join([
            f"Q: {entry['question']}\nA: {entry['answer']}\nConfidence: {entry['confidence']:.3f}"
            for entry in filtered_entries
        ])
        
        # Create prompt for GPT-4o
        prompt = f"""You are a helpful assistant answering questions based on the following relevant Q&A pairs from a knowledge base.

Relevant Knowledge Base Entries:
{context}

User Question: {user_question}

Based on the relevant entries above, provide a comprehensive and accurate answer to the user's question. If the information is not sufficient or directly relevant, say so clearly. Synthesize information from multiple entries when appropriate."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided knowledge base entries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content, True
        
    except Exception as e:
        return f"Error generating response: {str(e)}", False

def main():
    st.title("Semantic Search RAG System with GPT-4o")
    
    vector_db = load_vector_database()
    if vector_db is None:
        return
    
    model = load_model()
    
    st.write(f"Database loaded with {len(vector_db['questions'])} Q&A pairs")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    similarity_threshold = st.slider(
        "Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.05,
        help="Minimum confidence required to include entries in GPT-4o context."
    )
    
    question = st.text_input("Enter your question:", placeholder="Ask me anything...")
    
    if st.button("Submit"):
        if question:
            with st.spinner("Searching and generating answer..."):
                results = find_top_answers(question, vector_db, model)
                llm_answer, success = generate_llm_response(question, results, similarity_threshold)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("GPT-4o Generated Answer:")
                if success:
                    st.write(llm_answer)
                    
                    # Show which entries were used
                    relevant_count = len([r for r in results if r['confidence'] >= similarity_threshold])
                    st.info(f"Answer generated using {relevant_count} relevant database entries (above {similarity_threshold:.2f} threshold)")
                else:
                    st.error(llm_answer)  # Error message
                    
                    # Fallback to original behavior
                    st.subheader("Fallback - Best Match:")
                    if results[0]['confidence'] >= similarity_threshold:
                        st.write(results[0]['answer'])
                        st.write(f"**Matched Question:** *{results[0]['question']}*")
                        st.write(f"**Confidence:** {results[0]['confidence']:.3f}")
                    else:
                        st.write("No entries meet the confidence threshold.")
            
            with col2:
                st.subheader("Retrieved Entries:")
                for i, result in enumerate(results):
                    color = "🟢" if result['confidence'] >= similarity_threshold else "🔴"
                    with st.expander(f"{color} #{i+1} ({result['confidence']:.3f})"):
                        st.write(f"**Q:** {result['question']}")
                        st.write(f"**A:** {result['answer']}")
                        if result['confidence'] >= similarity_threshold:
                            st.success("Used in GPT-4o context")
                        else:
                            st.warning("Below threshold - not used")
            
        else:
            st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()