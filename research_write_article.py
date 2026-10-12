#!/usr/bin/env python
"""
Research-and-write pipeline using crewAI.

Three agents run sequentially:
    1. Planner  : outlines the article and gathers angles.
    2. Writer   : drafts the article from the planner's outline.
    3. Editor   : polishes the draft for voice and correctness.

The topic is passed in at kickoff time and substituted into the agent
backstories and task descriptions via the `{topic}` placeholder.

LLM: local Ollama server running `llama3.1:8b`.
Start Ollama first (see research_write_article.sh), then execute this file.

Output is saved as markdown under articles/.
"""

import re
import sys
import warnings
from datetime import date
from pathlib import Path

from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool
from crewai_tools import ScrapeWebsiteTool
from ddgs import DDGS

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Points crewAI at the local Ollama server. The `ollama/` prefix tells
# LiteLLM (used by crewAI under the hood) which provider adapter to pick.
llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# One compound tool: DuckDuckGo search, then fetch and read the top pages.
# Rationale: an 8B model handles a single tool better than two. Giving the
# planner "search" and "fetch" separately means it has to decide which to
# call when; combining them removes that decision.
#
# Per-page text is truncated to keep the context manageable on an 8K window.
# DDG rate-limits scrapers, so failures are swallowed per URL rather than
# bubbled up.

_scraper = ScrapeWebsiteTool()
_MAX_RESULTS = 3
_MAX_CHARS_PER_PAGE = 4000


@tool("Web Research")
def web_research(query: str) -> str:
    """
    Search the web with DuckDuckGo and read the top pages.

    Returns the concatenated page text for the top results, each labeled
    with its title and URL. Use this to gather facts, stats, quotes, and
    source links on a topic. Pass a focused search query, not a question.
    """
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=_MAX_RESULTS))
    if not hits:
        return "No results."

    sections = []
    for hit in hits:
        url = hit.get("href", "")
        title = hit.get("title", "")
        try:
            text = _scraper.run(website_url=url) or ""
        except Exception as exc:
            text = f"(fetch failed: {exc})"
        sections.append(f"## {title}\n{url}\n\n{text[:_MAX_CHARS_PER_PAGE]}")

    return "\n\n---\n\n".join(sections)


search_tool = web_research


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
# Each agent is a role-playing persona. `role`, `goal`, and `backstory` are
# all injected into the system prompt. `{topic}` is interpolated at kickoff.
# `allow_delegation=False` keeps agents on their own task instead of handing
# work to peers (simpler, more predictable for a linear pipeline).

planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory=(
        "You're working on planning a blog article about the topic: {topic}. "
        "You collect information that helps the audience learn something "
        "and make informed decisions. Your work is the basis for the Content "
        "Writer to write an article on this topic. You use web search to "
        "verify current facts, trends, and sources before writing the outline."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[search_tool],
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate opinion piece about the topic: {topic}",
    backstory=(
        "You're writing a new opinion piece about the topic: {topic}. "
        "You base your writing on the work of the Content Planner, who "
        "provides an outline and relevant context about the topic. You "
        "follow the main objectives and direction of the outline. You "
        "provide objective and impartial insights and back them up with "
        "information provided by the Content Planner. You acknowledge in "
        "your opinion piece when your statements are opinions as opposed "
        "to objective statements."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory=(
        "You are an editor who receives a blog post from the Content Writer. "
        "Your goal is to review the blog post to ensure it follows "
        "journalistic best practices, provides balanced viewpoints when "
        "providing opinions or assertions, and avoids major controversial "
        "topics or opinions when possible."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
# Tasks are the unit of work. Each is bound to one agent. `description` is
# the user prompt; `expected_output` tells the agent what shape of answer
# counts as "done" and is also what gets passed to the next task in the
# crew.

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including an introduction, "
        "key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output=(
        "A comprehensive content plan document with an outline, "
        "audience analysis, SEO keywords, and resources."
    ),
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named in an engaging manner.\n"
        "4. Ensure the post is structured with an engaging introduction, "
        "insightful body, and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and alignment with the brand's voice."
    ),
    expected_output=(
        "A well-written blog post in markdown format, ready for publication. "
        "Each section should have 2 or 3 paragraphs."
    ),
    agent=writer,
)

edit = Task(
    description=(
        "Proofread the given blog post for grammatical errors and "
        "alignment with the brand's voice."
    ),
    expected_output=(
        "A well-written blog post in markdown format, ready for publication. "
        "Each section should have 2 or 3 paragraphs."
    ),
    agent=editor,
)


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------
# Default process is sequential: tasks run in list order and each task's
# output is fed to the next. Order matters here (plan -> write -> edit).
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _slugify_topic(topic: str) -> str:
    """Make a topic safe for a filename: strip punctuation, spaces to underscores."""
    cleaned = re.sub(r"[^\w\s-]", "", topic).strip()
    return re.sub(r"\s+", "_", cleaned) or "topic"


ARTICLES_DIR = Path(__file__).resolve().parent / "articles"


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else "Artificial Intelligence"
    result = crew.kickoff(inputs={"topic": topic})

    ARTICLES_DIR.mkdir(exist_ok=True)
    out_path = ARTICLES_DIR / f"{date.today():%Y%m%d}_article_{_slugify_topic(topic)}.md"
    out_path.write_text(result.raw, encoding="utf-8")

    print(result.raw)
    print(f"\n[saved to {out_path}]")


if __name__ == "__main__":
    main()
