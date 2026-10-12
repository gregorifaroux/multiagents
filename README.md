# multiagents

A sandbox for multi-agent experiments with [crewAI](https://github.com/crewAIInc/crewAI), running everything locally against [Ollama](https://ollama.com).

## What's here

| File | Purpose |
| --- | --- |
| `research_write_article.py` | Three-agent pipeline (planner, writer, editor) that produces a blog post on a given topic. |
| `research_write_article.sh` | Launcher: brings Ollama up, pulls the model if missing, runs the Python script. |
| `requirements.txt` | Python deps (`crewai`, `crewai-tools`, `langchain-community`, `ddgs`). |
| `articles/` | Generated markdown output (gitignored). |

## Setup

```bash
# 1. Install Ollama (one-time)
brew install ollama

# 2. Create the venv and install deps (uv recommended)
uv venv --python 3.11 venv
source venv/bin/activate
uv pip install -r requirements.txt
```

The launcher will pull `llama3.1:8b` on first run (~5 GB).

## Run

```bash
./research_write_article.sh
# or with a custom topic
./research_write_article.sh research_write_article.py
python research_write_article.py "Autonomous vehicles"
```

Output lands in `articles/YYYYMMDD_article_<topic>.md`.

## How it works

```
      web_research tool (DDG + ScrapeWebsiteTool)
                 │
                 ▼
 [Planner] ──► [Writer] ──► [Editor] ──► markdown file
```

- **Planner** gets the one compound `web_research` tool. It searches DuckDuckGo and scrapes the top 3 pages, so the outline is grounded in real sources instead of the model's parametric memory.
- **Writer** drafts the article from the planner's outline. No tools; just the LLM.
- **Editor** polishes voice and correctness. No tools.
- All three use `ollama/llama3.1:8b` via LiteLLM.

Tasks run sequentially; each task's output feeds the next.

## Notes

- `llama3.1:8b` was chosen because it supports tool calling and runs on consumer hardware (~5 GB RAM). Swap for `qwen2.5:14b` if you have the headroom and want tighter tool-use adherence.
- DuckDuckGo rate-limits scrapers. Transient failures mid-run are expected; retries usually work.
- The one-compound-tool design (search + scrape in a single call) is a deliberate concession to the 8B model: fewer tools = fewer decisions = fewer loops stall.
