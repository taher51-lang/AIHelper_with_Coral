<div align="center">

# 🪸 Coral
### The Dev-to-Creator Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC.svg)](https://tailwindcss.com/)

*Turn your raw GitHub commits and code metrics into a high-converting, multi-platform audience distribution strategy.*

</div>

---

## 🚀 The Problem

Developers build amazing things, but rarely talk about them effectively. Their GitHub is full of high-signal data (commits, PRs, complex architectures) that never sees the light of day on social platforms where audiences are built.

## ✨ The Solution: Coral

**Coral** is an agentic marketing pipeline that bridges the gap between engineering execution and audience building. Instead of generic "AI text wrappers," Coral uses a **Multi-Agent Strategy Engine** to query your actual GitHub database, formulate a marketing strategy, and write platform-specific copy (Twitter, LinkedIn, Instagram).

### 🛠 Core Features

- **🗣️ Natural Language to SQL**: Tell the AI what you want (e.g., *"Make a launch post about my new RAG repo"*). Coral translates this into a SQL query and fetches your exact live GitHub metrics.
- **🧠 Two-Stage Marketing Agent**: 
  - **Stage 1 (The Strategist)**: Analyzes the raw code data to determine the best marketing angle, hook, and value proposition.
  - **Stage 2 (The Copywriter)**: Takes the approved strategy and drafts a Twitter thread, a LinkedIn post, and an Instagram Midjourney prompt.
- **🔍 Transparent Reasoning Panel**: Click the "Why this campaign?" badge in the UI to see *exactly* why the AI chose its specific marketing angle and narrative hook.
- **💬 Agentic Refinement Loop**: Don't like the generated campaign? Type *"Make it punchier"* or *"Focus more on the backend features"*. Coral passes this feedback back into the reasoning loop to intelligently rewrite the strategy and copy.
- **🎨 Midnight Ocean UI**: A sleek, custom-designed dark mode interface that feels like a premium SaaS product.

---

## 🏗 Architecture

1. **User Intent** -> Fast API Endpoint
2. **SQL Agent** -> Translates intent into a Postgres/Coral SQL query.
3. **Data Synthesizer** -> Formats the raw repository/commit data into context.
4. **Marketing Engine** -> LangChain/Groq powered 2-stage reasoning chain (Strategy -> Execution).
5. **Frontend UI** -> Interacts via AJAX to display tabs, the reasoning panel, and handle refinement loops.

---

## 🚦 Getting Started

### Prerequisites
- Python 3.10+
- `GROQ_API_KEY` (for the fast, agentic LLM reasoning)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/coral.git
cd coral
```

2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Set up your `.env` file
```bash
echo "GROQ_API_KEY=your_api_key_here" > .env
```

4. Run the backend server
```bash
uvicorn main:app --port 8000 --reload
```

5. Open the UI
Simply open `index.html` in your favorite modern browser, or serve it using a simple HTTP server:
```bash
python -m http.server 3000
```

---

## 🏆 Hackathon Context

This project was built over the course of a hackathon to prove that AI shouldn't just be a "text generation box." By utilizing **Agentic Workflows** (planning before writing) and **Transparent Reasoning** (showing the user the strategy), Coral acts as a genuine collaborative co-pilot for developer relations.
