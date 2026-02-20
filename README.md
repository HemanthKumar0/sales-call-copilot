# Sales Call Copilot

A RAG based CLI chatbot that ingests sales call transcripts, stores them in a persistent Chroma vector database, and provides intelligent querying, summarization, and insight extraction all with mandatory source citations.

## Features

- Ingest `.txt` transcript files into a persistent vector store
- Semantic search across all ingested calls
- Summarize individual calls or the most recent call
- Free form natural language queries with RAG
- Source citations on every response
- Rich formatted CLI with tables, panels, and color

## Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your API key:

   ```bash
   cp .env.example .env
   ```

   Open `.env` and set your key:

   ```
   OPENAI_API_KEY=sk-...
   ```

3. Run the app:

   ```bash
   python -m src.cli
   ```

## Usage

Once running, you'll see a `You>` prompt. Here are the available commands:

| Command | Description |
|---|---|
| `list` or `list my call ids` | Show all ingested calls |
| `ingest <filepath>` | Ingest a transcript file |
| `summarize the last call` | Summarize the most recently ingested call |
| `summarize call <id>` | Summarize a specific call by ID |
| Any other text | Free-form RAG query across all transcripts |
| `exit` / `quit` / `bye` | Exit the application |

Both `summarize` and `summarise` spellings are supported.

### Example Session

```
You> ingest data/sample_discovery_call.txt
┌─ Ingestion Complete──────────┐
│ Successfully ingested!       │
│ Call ID: a1b2c3              │
│ Title: Sample Discovery Call │
└──────────────────────────────┘

You> list
┌─ Ingested Calls ─────────────────────────────────┐
│ Call ID  │ Title                  │ Date         │
│ a1b2c3   │ Sample Discovery Call  │ 2025-01-15   │
└──────────────────────────────────────────────────┘

You> summarize the last call
┌─ Summary: Sample Discovery Call ─┐
│ The customer is evaluating ...   │
│                                  │
│ Sources:                         │
│ - Call a1b2c3, Segment 0: "..."  │
└──────────────────────────────────┘

You> What pricing was discussed?
┌─ Answer ─────────────────────────┐
│ The Growth plan was quoted at    │
│ $29 per user per month for 50    │
│ seats...                         │
│                                  │
│ Sources:                         │
│ - Call a1b2c3, Segment 5: "..."  │
└──────────────────────────────────┘

You> exit
Goodbye! 👋
```

## Project Structure

```
sales-call-copilot/
├── data/                  # Transcript .txt files
├── chroma_db/             # Persistent vector store (auto-created)
├── src/
│   ├── cli.py             # CLI entry point and command parser
│   ├── config.py          # Settings and environment config
│   ├── ingestion.py       # Transcript ingestion engine
│   ├── models.py          # Pydantic data models
│   ├── prompts.py         # LLM prompt templates
│   ├── retriever.py       # Semantic search and LLM querying
│   ├── storage.py         # Metadata store and vector store setup
│   └── utils.py           # Shared helpers
├── tests/                 # Unit and property-based tests
├── calls_metadata.json    # Ingested call metadata
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
└── .gitignore
```

## Assumptions

- Transcripts are plain `.txt` files
- The Chroma database persists to `./chroma_db/` and survives restarts
- The OpenAI `gpt-5-mini` model is used for LLM responses (temperature=0)
- The OpenAI `text-embedding-3-small` model is used for embeddings
- All LLM responses are grounded in retrieved context with source citations

