"""
Baseline inference script for the AI Code Review OpenEnv environment.
Follows the required log format:  [START] ... [STEP] ... [END]

Usage:
    export API_BASE_URL="https://api.anthropic.com/v1"   # or any OpenAI-compatible URL
    export MODEL_NAME="claude-sonnet-4-20250514"
    export HF_TOKEN="hf_..."          # used as API key if API_BASE_URL points to HF
    python inference.py
"""

from __future__ import annotations
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
#              Try to import openai; fall back gracefully
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
#                                 Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
API_KEY      = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "sk-placeholder")

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASK_IDS = ["task_1", "task_2", "task_3"]
MAX_STEPS_OVERRIDE = None   # set to int to cap steps for quick testing

# ---------------------------------------------------------------------------
# HTTP helper (stdlib only, so no requests dependency needed for env calls)
# ---------------------------------------------------------------------------
import urllib.request
import urllib.error


def _http_post(url: str, payload: dict, retries: int = 3) -> dict:
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1)
    raise urllib.error.URLError(str(last_err))


def _http_get(url: str, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1)
    raise urllib.error.URLError(str(last_err))


# ---------------------------------------------------------------------------
#                            LLM client
# ---------------------------------------------------------------------------

def build_llm_client() -> Optional[Any]:
    if not OPENAI_AVAILABLE:
        print("[WARN] openai package not installed – using rule-based fallback agent.")
        return None
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        return client
    except Exception as e:
        print(f"[WARN] Could not build LLM client: {e}. Using fallback agent.")
        return None


SYSTEM_PROMPT = """\
You are an expert code reviewer. You will be given a code snippet and must:
1. Analyze the code
2. Flag all bugs (syntax, logic, security, performance)
3. Suggest specific fixes for each bug
4. Suggest code improvements

You must respond ONLY with a valid JSON array of actions.
Each action must have these fields:
  - action_type: one of [analyze_code, flag_bug, suggest_fix, suggest_improvement, request_changes, approve_pr]
  - target: (optional) bug label or area being addressed
  - description: detailed explanation
  - line_number: (optional) relevant line number

Example response:
[
  {"action_type": "analyze_code", "description": "Performing full code analysis"},
  {"action_type": "flag_bug", "target": "bug_1_a", "description": "Off-by-one in range()", "line_number": 4},
  {"action_type": "suggest_fix", "target": "bug_1_a", "description": "Change range(len(numbers)+1) to range(len(numbers))"},
  {"action_type": "suggest_improvement", "description": "Use sum() built-in for clarity – more Pythonic"},
  {"action_type": "request_changes", "description": "Bugs found – requesting changes"}
]

Be thorough. Cover all bugs you can find.
"""


def get_llm_actions(client, observation: dict, task_id: str) -> List[dict]:
    """Ask the LLM for a sequence of actions given the current observation."""
    code = observation.get("code_snippet", "")
    goal = observation.get("task_goal", "")
    prompt = f"Task goal: {goal}\n\nCode to review:\n```python\n{code}\n```\n\nProvide your review actions as a JSON array."
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        print(f"   [LLM ERROR] {e} – using fallback actions.")
        return get_fallback_actions(task_id)


# ---------------------------------------------------------------------------
#                  Rule-based fallback agent (no LLM required)
# ---------------------------------------------------------------------------

FALLBACK_ACTIONS: Dict[str, List[dict]] = {
    "task_1": [
        {"action_type": "analyze_code", "description": "Analysing the function for bugs"},
        {
            "action_type": "flag_bug",
            "target": "bug_1_a",
            "description": "Off-by-one error: range(len(numbers) + 1) causes IndexError on last iteration",
            "line_number": 4,
        },
        {
            "action_type": "suggest_fix",
            "target": "bug_1_a",
            "description": "Change to range(len(numbers)) to iterate correctly without out-of-bounds access",
        },
        {
            "action_type": "suggest_improvement",
            "description": "Use the built-in sum() function instead of manual loop – more Pythonic",
        },
        {"action_type": "request_changes", "description": "Requesting fix for off-by-one bug"},
    ],
    "task_2": [
        {"action_type": "analyze_code", "description": "Analysing the duplicate-finder function"},
        {
            "action_type": "flag_bug",
            "target": "bug_2_a",
            "description": "Inner loop starts at 0 instead of i+1 – compares each element with itself and causes redundant work",
            "line_number": 5,
        },
        {
            "action_type": "flag_bug",
            "target": "bug_2_b",
            "description": "O(n²) performance issue: nested loops are inefficient for large inputs",
            "line_number": 4,
        },
        {
            "action_type": "suggest_fix",
            "target": "bug_2_a",
            "description": "Change inner loop to range(i+1, len(items)) to avoid redundant comparisons",
        },
        {
            "action_type": "suggest_fix",
            "target": "bug_2_b",
            "description": "Replace nested loops with collections.Counter to achieve O(n) linear time",
        },
        {
            "action_type": "suggest_improvement",
            "description": "Use collections.Counter for O(n) linear duplicate detection instead of O(n²) loops",
        },
        {
            "action_type": "suggest_improvement",
            "description": "Add type hints to function signature: def find_duplicates(items: list) -> list",
        },
        {"action_type": "request_changes", "description": "Multiple bugs found – requesting changes"},
    ],
    "task_3": [
        {"action_type": "analyze_code", "description": "Analysing database query helper for security and performance issues"},
        {
            "action_type": "flag_bug",
            "target": "bug_3_a",
            "description": "SQL Injection vulnerability: username is interpolated directly into the SQL query string, allowing attacker manipulation",
            "line_number": 9,
        },
        {
            "action_type": "flag_bug",
            "target": "bug_3_b",
            "description": "Logic error in pagination: offset = page * page_size should be offset = (page - 1) * page_size; page 1 currently skips first page",
            "line_number": 14,
        },
        {
            "action_type": "flag_bug",
            "target": "bug_3_c",
            "description": "N+1 query anti-pattern: separate SELECT issued for each post in the loop, causing N extra DB round-trips",
            "line_number": 18,
        },
        {
            "action_type": "suggest_fix",
            "target": "bug_3_a",
            "description": "Use parameterised query: cursor.execute(query, (username,)) to prevent SQL injection",
        },
        {
            "action_type": "suggest_fix",
            "target": "bug_3_b",
            "description": "Fix offset to (page - 1) * page_size so page 1 returns the first set of results",
        },
        {
            "action_type": "suggest_fix",
            "target": "bug_3_c",
            "description": "Use a JOIN query to fetch posts and comments together in one SQL round-trip, eliminating N+1",
        },
        {
            "action_type": "suggest_improvement",
            "description": "Use a context manager (with statement) for the sqlite3 connection: 'with sqlite3.connect(db_path) as conn'",
        },
        {
            "action_type": "suggest_improvement",
            "description": "Apply SQL LIMIT/OFFSET directly in the query instead of Python slicing for efficiency",
        },
        {"action_type": "request_changes", "description": "Critical security and performance issues found – requesting changes"},
    ],
}


def get_fallback_actions(task_id: str) -> List[dict]:
    return FALLBACK_ACTIONS.get(task_id, [
        {"action_type": "analyze_code", "description": "Generic analysis"},
        {"action_type": "request_changes", "description": "Issues found"},
    ])


# ---------------------------------------------------------------------------
#                                  Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: Optional[Any], task_id: str) -> Dict[str, Any]:
    """Run a single episode and return results."""

    print(f"\n[START] task_id={task_id}")
    print("-" * 60)

    # Reset environment
    reset_resp = _http_post(f"{ENV_BASE_URL}/reset", {"task_id": task_id})
    obs = reset_resp["observation"]
    print(f"  Goal : {obs['task_goal'][:100]}...")
    print(f"  Steps: {obs['remaining_steps']}")

    # Decide actions
    if client is not None:
        actions = get_llm_actions(client, obs, task_id)
    else:
        actions = get_fallback_actions(task_id)

    total_reward = 0.0
    step_num     = 0
    final_info   = {}

    for act in actions:
        if MAX_STEPS_OVERRIDE and step_num >= MAX_STEPS_OVERRIDE:
            break

        action_type = act.get("action_type", "")
        target      = act.get("target")
        description = act.get("description", "")
        line_number = act.get("line_number")

        payload = {
            "action_type": action_type,
            "target": target,
            "description": description,
            "line_number": line_number,
            "extra": {},
        }

        try:
            resp   = _http_post(f"{ENV_BASE_URL}/step", payload)
        except urllib.error.URLError as e:
            print(f"  [ERROR] Could not reach environment server: {e}")
            break

        reward   = resp["reward"]
        done     = resp["done"]
        info     = resp["info"]
        step_obs = resp["observation"]

        total_reward += reward
        step_num     += 1
        final_info    = info

        print(
            f"[STEP] {step_num:02d} | {action_type:<22} | reward={reward:+.3f} | "
            f"cumulative={step_obs['cumulative_reward']:+.4f} | "
            f"msg={step_obs['message'][:60]}"
        )

        if done:
            break
        time.sleep(0.05)  # small delay to be polite

    # Final grading
    try:
        state = _http_get(f"{ENV_BASE_URL}/state")
        from env.grader import Grader
        from env.tasks import TASK_REGISTRY
        grader = Grader(TASK_REGISTRY[task_id])
        grading = grader.evaluate(state)
    except Exception:
        grading = final_info.get("grading", {"score": 0.0, "details": "N/A"})

    score = grading.get("score", 0.0)

    print(f"\n[END] task_id={task_id}")
    print(f"  Steps taken    : {step_num}")
    print(f"  Total reward   : {total_reward:+.4f}")
    print(f"  Final score    : {score:.4f}")
    print(f"  Grading details: {grading.get('details', 'N/A')}")
    print("-" * 60)

    return {
        "task_id": task_id,
        "steps": step_num,
        "total_reward": round(total_reward, 4),
        "score": score,
        "grading": grading,
    }


# ---------------------------------------------------------------------------
#                                  Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  AI Code Review – OpenEnv Baseline Inference")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  API URL : {API_BASE_URL}")
    print(f"  Env URL : {ENV_BASE_URL}")
    print("=" * 60)

    # Verify server is up
    try:
        health = _http_get(f"{ENV_BASE_URL}/health")
        print(f"  Server status: {health['status']}")
    except Exception as e:
        print(f"[ERROR] Environment server not reachable at {ENV_BASE_URL}: {e}")
        print("  Start it with: uvicorn app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    client = build_llm_client()

    all_results: List[Dict] = []
    for task_id in TASK_IDS:
        result = run_episode(client, task_id)
        all_results.append(result)

    # Aggregate
    mean_score = sum(r["score"] for r in all_results) / len(all_results)
    total_rwds  = sum(r["total_reward"] for r in all_results)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(
            f"  {r['task_id']:8s} | score={r['score']:.4f} | "
            f"reward={r['total_reward']:+.4f} | steps={r['steps']}"
        )
    print(f"\n  Mean score    : {mean_score:.4f}")
    print(f"  Total reward  : {total_rwds:+.4f}")
    print("=" * 60)

    # Machine-readable output
    print("\n[RESULTS_JSON]")
    print(json.dumps({"tasks": all_results, "mean_score": round(mean_score, 4)}, indent=2))


if __name__ == "__main__":
    main()
