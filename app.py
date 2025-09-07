import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

def main():
    st.title("Semantic Search RAG System")
    
    vector_db = load_vector_database()
    if vector_db is None:
        return
    
    model = load_model()
    
    st.write(f"Database loaded with {len(vector_db['questions'])} Q&A pairs")
    
    similarity_threshold = st.slider(
        "Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.05,
        help="Minimum confidence required to show an answer. Below this threshold, 'I'm not sure' will be displayed."
    )
    
    question = st.text_input("Enter your question:", placeholder="Ask me anything...")
    
    if st.button("Submit"):
        if question:
            with st.spinner("Searching for the best answer..."):
                results = find_top_answers(question, vector_db, model)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Best Answer:")
                if results[0]['confidence'] >= similarity_threshold:
                    st.write(results[0]['answer'])
                    
                    st.subheader("Matched Question:")
                    st.write(f"*{results[0]['question']}*")
                    st.write(f"**Confidence:** {results[0]['confidence']:.3f}")
                else:
                    st.write("I'm not sure about that. The most similar question I found doesn't meet the confidence threshold.")
                    st.write(f"**Best match confidence:** {results[0]['confidence']:.3f} (threshold: {similarity_threshold:.3f})")
                    
                    with st.expander("See best match anyway"):
                        st.write(f"**Q:** {results[0]['question']}")
                        st.write(f"**A:** {results[0]['answer']}")
            
            with col2:
                st.subheader("Top 5 Matches:")
                for i, result in enumerate(results):
                    with st.expander(f"#{i+1} ({result['confidence']:.3f})"):
                        st.write(f"**Q:** {result['question']}")
                        st.write(f"**A:** {result['answer']}")
            
        else:
            st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()