# TransAdviceAgent

A RAG (Retrieval-Augmented Generation) powered Q&A system that provides evidence-based information and advice on transgender topics, particularly focusing on medical and surgical information.

## Overview

TransAdviceAgent combines community knowledge from Reddit discussions, personal experiences, and medical resources to answer questions about transgender care. The system uses advanced AI to retrieve relevant information and provide comprehensive, source-backed responses.

## Key Features

- **Intelligent Q&A**: Ask questions about transgender topics and get detailed, evidence-based answers
- **Source Attribution**: Every answer includes links to the original sources
- **Community Knowledge**: Leverages discussions from r/Transgender_Surgeries and other community resources
- **Medical Information**: Includes curated medical content about hormone therapy, surgical procedures, and care
- **Real-time API**: FastAPI backend for integration with web applications

## Technology Stack

- **Backend**: FastAPI with Python
- **AI/ML**: 
  - Anthropic Claude for natural language generation
  - HuggingFace transformers for text embeddings
  - LangGraph for orchestrating the RAG pipeline
- **Vector Database**: FAISS for semantic search
- **Document Storage**: SQLite for efficient document retrieval
- **Data Sources**: Reddit API, WhatsApp chat exports, curated text content

## Architecture

The system uses a sophisticated RAG pipeline that:

1. **Ingests** diverse data sources (Reddit posts, chat logs, medical texts)
2. **Processes** and chunks documents for optimal retrieval
3. **Embeds** content using sentence transformers
4. **Retrieves** relevant documents based on semantic similarity
5. **Summarizes** information using Claude AI
6. **Provides** comprehensive answers with source attribution

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   # Required for AI models
   ANTHROPIC_API_KEY=your_key_here
   LANGSMITH_API_KEY=your_key_here
   
   # Optional: For Reddit data collection
   REDDIT_CLIENT_ID=your_id
   REDDIT_CLIENT_SECRET=your_secret
   REDDIT_USER_AGENT=your_agent
   ```

3. **Build the vector store** (if you have data):
   ```bash
   python build_vectorstore.py
   ```

4. **Run the API server**:
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Ask questions**:
   ```bash
   curl -X POST "http://localhost:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"question": "What should I know about hormone therapy?"}'
   ```

## Data Collection

The system can ingest data from multiple sources:

- **Reddit**: Use `scripts/get_reddit.py` to collect posts from r/Transgender_Surgeries
- **WhatsApp**: Use `copy_whatsapp_chats.py` to process chat exports
- **Text Files**: Add any `.txt` files to the `data/` directory

### Fetching Wikipedia articles

You can fetch Wikipedia pages by category and save each page as JSON for later processing. Example:

```bash
python scripts/fetch_wikipedia_by_category.py --categories "Transgender rights,Transgender" --max-pages 200 --depth 1 --include-subcats
```

This will save files under `data/wikipedia/<category_slug>/`.

## Live Demo

🌐 **[Try the live application](https://sampease.github.io/TransAdviceAgent.html)**

For a detailed technical writeup, architecture overview, and implementation details, visit the comprehensive documentation at **[sampease.github.io/TransAdviceAgent.html](https://sampease.github.io/TransAdviceAgent.html)**.

## Important Notes

- This system is for informational purposes only and should not replace professional medical advice
- All content is sourced from community discussions and should be verified with healthcare providers
- The AI responses are generated based on available data and may not reflect the most current medical guidelines

## Large data artifacts

The repository stores large vectorstore artifacts (FAISS index, SQLite docs, and an ID map). These files are managed with Git LFS and may not be included in a shallow clone. If you need to obtain or rebuild them:

- The files are: `app/vectorstore/index.faiss`, `app/vectorstore/docs.sqlite`, and `app/vectorstore/id_map.json`.
- They are tracked via Git LFS; to fetch them after cloning run `git lfs pull` in the cloned repo or `git clone --recurse-submodules` if using the submodule setup.
- Alternatively, rebuild the vectorstore locally using the scripts in `scripts/` if you prefer not to store binary artifacts in the repo.

## Contributing

Contributions are welcome! Please ensure any additions maintain the focus on providing helpful, accurate, and respectful information for the transgender community.

## License

This project is designed to help the transgender community access information and support. Please use responsibly and ethically.