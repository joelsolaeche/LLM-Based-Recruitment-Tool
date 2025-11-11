# 🤖 LLM-Based Recruitment Tool

An intelligent job-matching application powered by Large Language Models (LLMs) that connects candidates with relevant job opportunities using advanced AI technologies.

## 📋 Overview

This recruitment tool leverages LangChain, vector embeddings, and LLM technologies to create an intelligent job-matching system. The application analyzes candidate profiles (from PDF resumes) and matches them with suitable job opportunities using semantic search and Retrieval-Augmented Generation (RAG).

### Key Features

- **Multi-Assistant System**: Three specialized AI assistants for different use cases
  - **Vanilla ChatGPT**: General-purpose conversational AI assistant
  - **Jobs Finder Assistant**: Semantic job search based on candidate profiles
  - **Jobs Agent**: Advanced agentic system with additional tools for personalized cover letter generation
  
- **Resume Processing**: Automatic extraction and analysis of PDF resumes
- **Semantic Search**: Vector-based job matching using ChromaDB
- **Multi-LLM Support**: Flexible provider system supporting both OpenAI GPT and Google Gemini
- **Conversation Memory**: Context-aware conversations with chat history management
- **Interactive UI**: User-friendly interface powered by Chainlit

## 🏗️ Architecture

```
├── backend/
│   ├── app.py                          # Main Chainlit application
│   ├── config.py                       # Configuration management
│   ├── etl.py                          # ETL pipeline for vector database
│   ├── llm_factory.py                  # LLM provider factory pattern
│   ├── retriever.py                    # Job retrieval system
│   ├── utils.py                        # Utility functions (PDF processing)
│   └── models/
│       ├── chatgpt_clone.py           # General chat assistant
│       ├── jobs_finder.py             # Job matching assistant
│       ├── jobs_finder_agent.py       # Advanced agentic system
│       └── resume_summarizer_chain.py # Resume summarization chain
├── dataset/
│   └── jobs.csv                        # Job listings database
├── tests/                              # Unit tests
├── .chainlit/                          # Chainlit configuration
├── docker-compose.yml                  # Docker orchestration
├── Dockerfile                          # Container definition
└── requirements.txt                    # Python dependencies
```

## 🛠️ Technology Stack

- **LLM Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers (paraphrase-MiniLM-L6-v2)
- **LLM Providers**: OpenAI GPT-4o-mini / Google Gemini 2.5 Flash
- **Web Framework**: Chainlit
- **PDF Processing**: PyPDF2
- **Data Processing**: Pandas, NumPy
- **Containerization**: Docker & Docker Compose

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose (recommended) **OR**
- Python 3.11+
- Git

### Configuration

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd assignment
   ```

2. **Set up environment variables**
   
   Create a `.env` file from the example:
   ```bash
   cp env.example .env
   ```

   Configure your LLM provider (supports both OpenAI and Google Gemini):
   ```bash
   # Choose provider: "openai" or "gemini"
   LLM_PROVIDER="gemini"
   
   # For Google Gemini
   GEMINI_LLM_MODEL="gemini-2.5-flash"
   GOOGLE_API_KEY="your-google-api-key-here"
   
   # For OpenAI (alternative)
   OPENAI_LLM_MODEL="gpt-4o-mini"
   OPENAI_API_KEY="your-openai-api-key-here"
   
   LANGCHAIN_VERBOSE=true
   ```

## 🐳 Run with Docker (Recommended)

**Step 1:** Build and start the application
```bash
docker-compose up --build
```

**Step 2:** Initialize the vector database (first time only)
```bash
docker-compose exec llm-recruitment-tool python backend/etl.py
```

**Step 3:** Access the application at `http://localhost:8000`

**Stop the application:**
```bash
docker-compose down
```

For detailed Docker instructions, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

## 💻 Run Locally (Without Docker)

**Step 1:** Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 2:** Install dependencies
```bash
pip install -r requirements.txt
```

**Step 3:** Run the ETL pipeline to populate the vector database
```bash
python backend/etl.py
```

**Step 4:** Start the Chainlit server
```bash
chainlit run backend/app.py
```

**Step 5:** Open your browser at `http://localhost:8000`

## 📊 How It Works

1. **Data Ingestion**: The ETL pipeline (`backend/etl.py`) processes the job listings from `dataset/jobs.csv`, creates vector embeddings, and stores them in ChromaDB.

2. **Resume Upload**: Users upload their resume (PDF format) through the Chainlit interface.

3. **Resume Processing**: The system extracts text from the PDF and creates a summarized profile of the candidate.

4. **Semantic Search**: When users describe their job preferences, the system performs semantic search against the vector database to find matching opportunities.

5. **AI-Powered Responses**: The LLM generates personalized responses, job recommendations, and can even create custom cover letters based on the candidate's profile and selected job.

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/backend/test_utils.py
pytest tests/backend/models/test_chatgpt_clone.py
```

With Docker:
```bash
docker-compose exec llm-recruitment-tool python -m pytest tests/
```

## 🎨 Code Style

This project uses [Black](https://black.readthedocs.io/) for code formatting:
```bash
black --line-length=88 .
```

## 🔧 LLM Provider Support

This project implements a factory pattern (`backend/llm_factory.py`) for seamless switching between LLM providers. Simply change the `LLM_PROVIDER` value in your `.env` file - no code changes required.

**Supported Providers:**
- OpenAI (GPT-4o-mini)
- Google Gemini (Gemini 2.5 Flash)

## 📁 Project Structure

- **ETL Pipeline** (`backend/etl.py`): Processes job data and creates vector embeddings
- **LLM Factory** (`backend/llm_factory.py`): Provider-agnostic LLM instantiation
- **Chat Models**:
  - `chatgpt_clone.py`: General conversational AI with memory
  - `jobs_finder.py`: Job matching with semantic search
  - `jobs_finder_agent.py`: Advanced agent with tool usage
  - `resume_summarizer_chain.py`: Resume analysis and summarization
- **Utilities** (`backend/utils.py`): PDF processing and helper functions
- **Retriever** (`backend/retriever.py`): ChromaDB vector search implementation

## 🐛 Troubleshooting

**Issue: Port already in use**
```bash
# Change port in docker-compose.yml or kill the process using port 8000
```

**Issue: API key errors**
- Verify your `.env` file has the correct API key
- Ensure `LLM_PROVIDER` matches your configured provider

**Issue: Vector database empty**
- Run the ETL pipeline: `python backend/etl.py`
- Check that `dataset/jobs.csv` exists

## 📄 License

This project is part of the Anyone AI Machine Learning Engineering program.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Chainlit Documentation](https://docs.chainlit.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Docker Documentation](https://docs.docker.com/)