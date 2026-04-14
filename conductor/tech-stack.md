# Tech Stack

## Core Technologies
- **Python**: Primary programming language for the pipeline and UI.

## Machine Learning & NLP
- **Transformers (Hugging Face)**: Used for abstractive summarization (e.g., `google/flan-t5-large` or `facebook/bart-large-cnn`).
- **Torch (PyTorch)**: Backend framework for model inference.
- **Sentence-Transformers**: Used for vectorizing and ranking text chunks (e.g., `all-MiniLM-L6-v2`).

## Frontend & UI
- **Streamlit**: Core framework for the interactive web interface.
- **Streamlit Extras**: To enhance UI functionality and layout options.

## Data Processing & Utilities
- **NumPy & Pandas**: For data manipulation and structured views.
- **PyPDF**: For extracting text from PDF judgments.
- **Rouge-Score**: For evaluating the performance of generated summaries.
