# IntelliDocs | Premium RAG Assistant

IntelliDocs is a modern Retrieval-Augmented Generation (RAG) application that allows you to chat with your documents using Google's Gemini Pro model.

## Features
- **Document Indexing**: Support for PDF and TXT files.
- **Conversational Memory**: Remembers context of the conversation.
- **Source Citation**: Shows exactly where information was retrieved from.
- **Modern UI**: Sleek, glassmorphism-inspired design with dark mode.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. **Get API Key**:
   Get your Google API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Technology Stack
- **Frontend**: Streamlit
- **Orchestration**: LangChain
- **Embeddings**: Google Generative AI Embeddings
- **LLM**: Gemini 1.5 Pro
- **Vector Database**: FAISS
