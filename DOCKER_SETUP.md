# Docker Setup Guide

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed on your system
- Git installed

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd assignment
   ```

2. **Create your `.env` file**
   
   Copy the `env.example` file and rename it to `.env`:
   ```bash
   cp env.example .env
   ```

   Edit the `.env` file and add your API key:
   ```bash
   # For Gemini
   LLM_PROVIDER="gemini"
   GOOGLE_API_KEY="your-google-api-key-here"
   
   # OR for OpenAI
   LLM_PROVIDER="openai"
   OPENAI_API_KEY="your-openai-api-key-here"
   ```

3. **Build and run the application**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   
   Open your browser and go to: `http://localhost:8000`

## Running the ETL Pipeline

Before using the application for the first time, you need to populate the vector database:

```bash
docker-compose exec llm-recruitment-tool python backend/etl.py
```

This will:
- Load the jobs dataset from `dataset/jobs.csv`
- Create embeddings for job descriptions
- Store them in the ChromaDB vector database

## Docker Commands

### Start the application
```bash
docker-compose up
```

### Start in detached mode (background)
```bash
docker-compose up -d
```

### View logs
```bash
docker-compose logs -f
```

### Stop the application
```bash
docker-compose down
```

### Rebuild the image (after code changes)
```bash
docker-compose up --build
```

### Execute commands inside the container
```bash
docker-compose exec llm-recruitment-tool <command>
```

Example: Run tests
```bash
docker-compose exec llm-recruitment-tool python -m pytest tests
```

## Troubleshooting

### Port already in use
If port 8000 is already in use, you can change it in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change 8001 to any available port
```

### Container keeps restarting
Check the logs:
```bash
docker-compose logs llm-recruitment-tool
```

Common issues:
- Missing or invalid API key in `.env` file
- Incorrect LLM_PROVIDER value

### Rebuilding from scratch
If you encounter persistent issues:
```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

## Data Persistence

The following directories are mounted as volumes for data persistence:
- `./dataset` - Job listings dataset
- `./chroma` - Vector database
- `./.chainlit` - Chainlit configuration

Data in these directories will persist even when the container is stopped or removed.
