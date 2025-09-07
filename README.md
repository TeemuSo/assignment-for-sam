# Semantic Search RAG System with GPT-4o

A Retrieval-Augmented Generation (RAG) system built with Streamlit, sentence transformers, and GPT-4o for intelligent question answering.

## Features

- **Semantic Search**: Uses all-MiniLM-L6-v2 model for embedding questions and answers
- **GPT-4o Integration**: Generates comprehensive answers using retrieved context
- **Split-Screen Interface**: Shows GPT-4o generated answer on the left, retrieved entries on the right
- **Confidence Threshold**: Adjustable slider to filter entries included in GPT-4o context
- **Fallback Behavior**: Falls back to basic similarity matching if GPT-4o fails
- **Lightweight**: Pre-computed vectors stored locally, no external database required

## Setup

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### Installation

1. Clone or download the project
2. Navigate to the project directory:
   ```bash
   cd sam-assignment-rag
   ```

3. Initialize the environment and install dependencies:
   ```bash
   uv init
   uv add streamlit sentence-transformers pandas scikit-learn openai python-dotenv
   ```

4. Set up your OpenAI API key:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

### 1. Generate Vector Database

First, create the vector embeddings from your CSV data:

```bash
uv run python generate_vectors.py
```

This will:
- Load questions and answers from `data.csv`
- Generate embeddings using the all-MiniLM-L6-v2 model
- Save the vector database as `vector_database.pkl`

### 2. Run the Streamlit App

```bash
uv run streamlit run app.py
```

The app will be available at `http://localhost:8501`

## How It Works

1. **Data Format**: The system expects a CSV file (`data.csv`) with columns `q` (questions) and `a` (answers)
2. **Vectorization**: Questions are embedded using sentence transformers
3. **Search**: User queries are embedded and compared using cosine similarity
4. **Context Filtering**: Entries above the confidence threshold are included in GPT-4o context
5. **Answer Generation**: GPT-4o synthesizes a comprehensive answer using the relevant entries
6. **Results**: Shows GPT-4o generated answer with context information and retrieved entries

## Updating Data

To update the knowledge base:

1. Modify `data.csv` with new questions and answers
2. Regenerate the vector database:
   ```bash
   uv run python generate_vectors.py
   ```
3. Restart the Streamlit app

## Project Structure

```
sam-assignment-rag/
- app.py                 # Main Streamlit application with GPT-4o integration
- generate_vectors.py    # Script to create vector database
- data.csv              # Q&A dataset
- vector_database.pkl   # Generated embeddings (created after running generate_vectors.py)
- .env                  # Environment variables (OpenAI API key)
- pyproject.toml        # uv project configuration with OpenAI dependencies
- README.md             # This file
```

## Configuration

- **Similarity Threshold**: Adjust in the app UI (default: 0.3) - filters which entries are included in GPT-4o context
- **Top-K Results**: Modify `top_k` parameter in `find_top_answers()` function (default: 5)
- **Model**: Change the sentence transformer model in both `generate_vectors.py` and `app.py`
- **GPT-4o Parameters**: Modify temperature and max_tokens in the `generate_llm_response()` function