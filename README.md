# PDF Q&A App (Groq + LLaMA3 + LangChain + Streamlit)
This app lets you upload a PDF and ask questions about its content using Retrieval-Augmented Generation (RAG).

## Features
PDF Upload & Q&A : Upload any PDF and ask questions about its content. The app extracts and processes the text for intelligent answers.
Groq’s LLaMA3-70B : Uses the powerful LLaMA3-70B model via Groq for fast, high-quality responses. Optimized for low-latency performance.
Semantic Search with FAISS : Documents are chunked and embedded for similarity search. FAISS ensures relevant context is retrieved efficiently.
HuggingFace Embeddings : Converts text into dense vectors using all-MiniLM-L6-v2. Enhances context relevance for accurate answers.
Streamlit Interface : Clean, interactive UI for uploading PDFs and asking questions. Runs locally or can be deployed online easily.
Secure Environment Handling : API keys and sensitive settings are stored in a .env file. Keeps your credentials safe and out of version control.

Frontend: Streamlit
LLM Backend: Groq (`llama3-70b-8192`)
Framework: LangChain
Vector Store: FAISS
Embeddings: HuggingFace `all-MiniLM-L6-v2`
PDF Parsing: PyPDF

## Setup
```bash
git clone https://github.com/<your-username>/pdf-qa-groq-app.git
cd pdf-qa-groq-app
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to the .env file
