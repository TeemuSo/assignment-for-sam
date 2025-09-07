import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

def generate_vector_database():
    print("Loading data...")
    df = pd.read_csv('data.csv')
    
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    questions = df['q'].tolist()
    answers = df['a'].tolist()
    
    print(f"Vectorizing {len(questions)} questions...")
    question_embeddings = model.encode(questions, convert_to_tensor=False)
    
    vector_db = {
        'questions': questions,
        'answers': answers,
        'embeddings': question_embeddings,
        'model_name': 'all-MiniLM-L6-v2'
    }
    
    print("Saving vector database...")
    with open('vector_database.pkl', 'wb') as f:
        pickle.dump(vector_db, f)
    
    print(f"Vector database saved! Shape: {question_embeddings.shape}")
    print("Run the Streamlit app with: uv run streamlit run app.py")

if __name__ == "__main__":
    generate_vector_database()