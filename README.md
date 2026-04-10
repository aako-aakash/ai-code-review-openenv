# 🔍 AI Code Review – OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)

---

## 🎯 Problem Motivation

Code review is one of the most time-consuming activities in modern software development. Developers spend 10–30% of their time reviewing pull requests, yet bugs still slip through. An AI agent that can autonomously review code—detecting bugs, suggesting fixes, and proposing improvements—would dramatically accelerate development cycles.

This OpenEnv environment simulates a realistic code review workflow across three difficulty levels, enabling the evaluation and training of AI agents in this high-value domain.

---

## 🌍 Environment Design

### Core Interface (OpenEnv Spec)

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset()` | `→ Observation` | Start a new episode with fresh state |
| `step(action)` | `→ (Observation, reward, done, info)` | Execute one agent action |
| `state()` | `→ dict` | Return full internal state snapshot |

### Architecture

```
code_review_env/
├── env/
│   ├── __init__.py        # Package exports
│   ├── models.py          # Pydantic: Action, Observation, TaskDefinition
│   ├── tasks.py           # 3 predefined tasks with ground truth
│   ├── grader.py          # Deterministic scoring engine
│   └── environment.py     # Core OpenEnv environment logic
├── app.py                 # FastAPI HTTP server
├── inference.py           # Baseline agent script
├── openenv.yaml           # OpenEnv spec manifest
├── Dockerfile             # Docker deployment
├── requirements.txt
└── README.md
```

---

## ⚡ Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `analyze_code` | none | Trigger initial code analysis (+0.05) |
| `flag_bug` | target, description, line_number? | Flag a specific bug (+0.20 if correct) |
| `suggest_fix` | target, description | Propose a fix for a bug (+0.30 if correct) |
| `suggest_improvement` | description | Suggest code improvements (+0.10 if relevant) |
| `approve_pr` | none | Approve the PR and close episode |
| `request_changes` | description? | Request changes and close episode |

---

## 👁️ Observation Space

```json
{
  "task_id": "task_1",
  "task_goal": "Review the function below...",
  "code_snippet": "def calculate_average(numbers):...",
  "detected_issues": [...],
  "suggested_fixes": [...],
  "suggested_improvements": [...],
  "review_history": [...],
  "remaining_steps": 12,
  "pr_approved": false,
  "pr_changes_requested": false,
  "cumulative_reward": 0.5,
  "message": "Correct! Bug 'bug_1_a' detected (+0.20)."
}
```

---

## 🏆 Reward Function

```
Step rewards:
  analyze_code (first time)     → +0.05
  correct bug detection          → +0.20
  correct fix suggestion         → +0.30
  relevant improvement           → +0.10
  wrong / irrelevant action      → -0.05 to -0.10
  repeated action                → -0.10

Completion rewards:
  Full correct review + approve  → +1.00
  Partial (changes requested)    → proportional bonus
```

### Episode Score (Grader)

```
score = bugs_found/total_bugs   * 0.50
      + fixes_correct/total     * 0.30
      + improvements/expected   * 0.20
```

Score is deterministic and in [0.0, 1.0].

---

## 📋 Tasks

### Task 1 – Easy (Off-by-One Bug)
- **Bug**: `range(len(numbers) + 1)` → `IndexError`
- **Fix**: `range(len(numbers))`
- **Improvement**: Use `sum()` built-in
- **Max steps**: 15

### Task 2 – Medium (Logic + Performance)
- **Bug 1**: Inner loop starts at 0 instead of `i+1`
- **Bug 2**: O(n²) nested loops (performance)
- **Fix**: `range(i+1, len(items))` + `collections.Counter`
- **Improvement**: Type hints, Counter pattern
- **Max steps**: 20

### Task 3 – Hard (Security + Logic + Performance)
- **Bug 1**: SQL Injection (f-string interpolation)
- **Bug 2**: Wrong pagination offset `page * page_size` → `(page-1) * page_size`
- **Bug 3**: N+1 query anti-pattern
- **Fixes**: Parameterised queries, corrected offset, JOIN query
- **Improvements**: Context manager, SQL LIMIT/OFFSET
- **Max steps**: 25

---

## 🚀 Setup & Running

### Option 1: Local Python

```bash
# Clone the project
git clone https://huggingface.co/spaces/YOUR_USER/ai-code-review-openenv
cd ai-code-review-openenv

# Install dependencies
pip install -r requirements.txt

# Start the server (in one terminal)
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Run inference (in another terminal)
export ENV_BASE_URL="http://localhost:7860"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
python inference.py
```

### Option 2: Docker

```bash
# Build the image
docker build -t ai-code-review-openenv .

# Run the server
docker run -p 7860:7860 ai-code-review-openenv

# In another terminal, run inference
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

### Option 3: Quick API test (curl)

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "flag_bug",
    "target": "bug_1_a",
    "description": "Off-by-one error in range causes IndexError",
    "line_number": 4
  }'

# Get state
curl http://localhost:7860/state

# List tasks
curl http://localhost:7860/tasks
```

---

## 🤗 Hugging Face Deployment

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
   - SDK: **Docker**
   - Tag it: `openenv`

2. Push the code:
```bash
git init
git remote add origin https://huggingface.co/spaces/YOUR_USER/ai-code-review-openenv
git add .
git commit -m "Initial OpenEnv deployment"
git push origin main
```

3. The Space auto-builds and exposes the API at:
   `https://YOUR_USER-ai-code-review-openenv.hf.space`

4. Run inference against the HF Space:
```bash
export ENV_BASE_URL="https://YOUR_USER-ai-code-review-openenv.hf.space"
python inference.py
```

---

## 📊 Baseline Results

Using the rule-based fallback agent (no LLM required):

| Task | Score | Reward | Steps |
|------|-------|--------|-------|
| task_1 (easy)   | 1.0000 | +1.65 | 5 |
| task_2 (medium) | 1.0000 | +2.20 | 8 |
| task_3 (hard)   | 1.0000 | +2.60 | 10 |
| **Mean**        | **1.0000** | **+6.45** | **23** |

The fallback agent uses hardcoded ground-truth actions to validate correctness. A real LLM agent will typically score 0.6–0.9 depending on model capability.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Health status |
| `/reset` | POST | Reset environment `{"task_id": "task_1"}` |
| `/step` | POST | Take action `{"action_type": "...", ...}` |
| `/state` | GET | Full internal state |
| `/tasks` | GET | List all tasks |
| `/tasks/{id}` | GET | Task details |
| `/docs` | GET | Interactive Swagger UI |

---

## 📝 License

MIT License – see [LICENSE](LICENSE)

## Author :  
# AAKASH
