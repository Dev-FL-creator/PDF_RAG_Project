# PDF RAG Project

A full-stack application for PDF document analysis using Retrieval-Augmented Generation (RAG) with Azure AI services.

## Features

- 📄 **PDF Upload & Indexing**: Upload PDF documents with intelligent text extraction
- 🔍 **Semantic Chunking**: Smart text segmentation preserving context and meaning
- 🖼️ **Image Analysis**: GPT-4o Vision integration for understanding diagrams and charts
- 🔍 **Intelligent Search**: Hybrid search combining BM25 keyword search and vector similarity
- 🤖 **Agent Mode**: Advanced AI agent with adaptive retrieval and query optimization
- 💬 **Natural Language QA**: Ask questions about your PDFs in natural language
- 🎯 **Azure AI Integration**: Leverages Azure OpenAI, Azure AI Search, and Azure Document Intelligence
- 📊 **Table Extraction**: Preserves table structure in TSV format
- 📍 **Page Tracking**: Accurate page number tracking for all content

## Architecture Highlights

### Backend Optimizations
- **Semantic Chunking**: Uses NLTK for sentence boundary detection, preserving paragraphs and semantic units
- **Image Understanding**: GPT-4o Vision analyzes diagrams, flowcharts, and images (not just OCR)
- **Table Processing**: Extracts tables with proper structure preservation
- **Independent Image Chunks**: Each image analysis becomes a standalone chunk to prevent fragmentation
- **Page-Level Tracking**: Maintains accurate page numbers through Azure Document Intelligence bounding regions

## Project Structure

```
PDF_RAG_Project/
├── backend/                 # FastAPI backend
│   ├── app.py              # Main application with NLTK auto-initialization
│   ├── config.json         # Configuration (Azure credentials) - DO NOT COMMIT
│   ├── config.example.json # Example configuration file
│   ├── requirements.txt    # Python dependencies
│   ├── routers/
│   │   └── pdf_rag_routes.py  # PDF RAG API routes with semantic chunking
│   ├── utils/
│   │   ├── semantic_chunker.py  # Intelligent text chunking
│   │   └── image_analyzer.py    # GPT-4o Vision image analysis
│   └── AI_AGENT/
│       └── pdf_agent_engine.py  # Intelligent agent logic
├── frontend/               # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx        # Main app component
│   │   ├── RAGPage.jsx    # PDF RAG interface
│   │   └── lib/           # API utilities
│   ├── package.json
│   └── vite.config.js
├── start.ps1              # Startup script (Windows)
└── .gitignore             # Git ignore configuration
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Node.js 18+
- Azure account with:
  - Azure OpenAI Service
  - Azure AI Search
  - Azure Document Intelligence

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/Dev-FL-creator/PDF_RAG_Project.git
cd PDF_RAG_Project

# Copy example config and add your Azure credentials
cp backend/config.example.json backend/config.json
# Edit backend/config.json with your actual Azure keys
```

### 2. Configure Azure Services

Edit `backend/config.json` with your Azure credentials:

```json
{
  "openai_api_key": "YOUR_OPENAI_API_KEY",
  "openai_endpoint": "https://your-openai.openai.azure.com/",
  "openai_api_version": "2024-10-21",
  "embedding_model": "text-embedding-ada-002",
  "openai_deployment": "gpt-4o",
  
  "search_service_name": "your-search-service",
  "search_api_key": "YOUR_SEARCH_API_KEY",
  "search_api_version": "2024-07-01",
  
  "docint_endpoint": "https://your-doc-intelligence.cognitiveservices.azure.com/",
  "docint_key": "YOUR_DOCINT_KEY",
  
  "vector_metric": "cosine",
  "embedding_dimensions": 1536
}
```

### 2. Install Backend Dependencies

```powershell
cd backend
python -m venv pdf_rag_venv
.\pdf_rag_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```powershell
cd frontend
npm install
```

### 4. Run the Application

**Option A: Use startup script (recommended)**
```powershell
.\start.ps1
```

**Option B: Run manually**

Terminal 1 (Backend):
```powershell
cd backend
.\pdf_rag_venv\Scripts\Activate.ps1
python app.py
```

Terminal 2 (Frontend):
```powershell
cd frontend
npm run dev
```

### 5. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Usage

### 1. Create an Index

1. Enter a custom index name (e.g., `pdf-my-docs`) or select from dropdown
2. Click "Create Index"

### 2. Upload PDF

1. Select an index
2. Choose a PDF file
3. Click "Upload & Index"
4. Wait for processing (chunks will be displayed)

### 3. Ask Questions

1. Enter your question in natural language
2. Adjust Top-K (number of chunks to retrieve)
3. Optionally enable **Agent Mode** for smarter retrieval
4. Click "Ask"
5. View retrieved context and generated answer

### Agent Mode

When enabled, the AI agent:
- Assesses query quality
- Rewrites vague queries for better retrieval
- Adaptively broadens search if evidence is insufficient
- Evaluates evidence sufficiency
- Generates reasoning-based answers

## API Endpoints

### PDF RAG

- `POST /api/pdf_rag/upload` - Upload and index PDF
- `POST /api/pdf_rag/query` - Query PDF index
- `POST /api/pdf_rag/create_index` - Create new index
- `GET /api/pdf_rag/agent_trace_stream/{trace_id}` - Real-time agent trace (SSE)

### Index Management

- `GET /api/indexes` - List all indexes
- `DELETE /api/indexes/{index_name}` - Delete index

## Architecture

### Backend Stack

- **FastAPI**: Modern async web framework
- **Azure OpenAI**: GPT-4 for Q&A, text-embedding-ada-002 for embeddings
- **Azure AI Search**: Hybrid search with vector + keyword matching
- **Azure Document Intelligence**: Advanced PDF text extraction

### Frontend Stack

- **React 19**: UI framework
- **Vite**: Fast build tool
- **React Bootstrap**: UI components
- **EventSource**: Real-time agent trace streaming

### RAG Pipeline

1. **Extraction**: Azure Document Intelligence extracts text, tables, key-value pairs
2. **Chunking**: Text split into overlapping chunks (800 chars, 120 overlap)
3. **Embedding**: Chunks converted to 1536-dim vectors via text-embedding-ada-002
4. **Indexing**: Vectors stored in Azure AI Search with metadata
5. **Retrieval**: Hybrid search (BM25 + vector similarity + semantic reranking)
6. **Generation**: GPT-4 generates answers grounded in retrieved context

## Configuration Options

### Index Settings

- **Vector Dimensions**: 1536 (default for text-embedding-ada-002)
- **Metric**: cosine similarity
- **HNSW Parameters**: m=8, efConstruction=400, efSearch=500

### Agent Settings (in `pdf_agent_engine.py`)

- `MIN_HITS`: Minimum evidence chunks (default: 2)
- `MAX_RETRIEVAL_ATTEMPTS`: Max broadening attempts (default: 5)
- `MIN_SCORE_FLOOR`: Minimum similarity score (default: 0.70)

## Troubleshooting

### Backend won't start

- Check Python version (3.9+)
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check `config.json` for valid Azure credentials

### Frontend won't start

- Check Node.js version (18+)
- Delete `node_modules` and reinstall: `rm -r node_modules; npm install`
- Check port 5173 is available

### Upload fails

- Ensure PDF is valid and not password-protected
- Check Azure Document Intelligence quota
- Verify index name follows rules (lowercase, alphanumeric, -, _)

### Query returns no results

- Ensure documents are uploaded to the selected index
- Try increasing Top-K
- Enable Agent Mode for better query optimization

## Development

### Adding New Features

1. Backend: Add routes in `backend/routers/`
2. Frontend: Add components in `frontend/src/`
3. Update API client in `frontend/src/lib/api.js`

### Building for Production

Frontend:
```powershell
cd frontend
npm run build
# Output in frontend/dist/
```

Backend: Deploy FastAPI app to Azure App Service or similar

## License

MIT License

## Support

For issues or questions, please check:
- Azure OpenAI documentation
- Azure AI Search documentation
- FastAPI documentation
- React documentation
