"""
FastAPI server that exposes the CodeReviewEnvironment via HTTP.
Compatible with OpenEnv validator and Hugging Face Spaces.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import CodeReviewEnvironment
from env.models import Action

app = FastAPI(
    title="AI Code Review – OpenEnv",
    description="OpenEnv environment for AI-powered pull-request code review.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (one per server; fine for single-user HF Space)
_env: Optional[CodeReviewEnvironment] = None


def _get_env() -> CodeReviewEnvironment:
    global _env
    if _env is None:
        _env = CodeReviewEnvironment(task_id="task_1")
        _env.reset()
    return _env


# ---------------------------------------------------------------------------
#                      Request / Response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"


class StepRequest(BaseModel):
    action_type: str
    target: Optional[str] = None
    description: Optional[str] = None
    line_number: Optional[int] = None
    extra: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
#                                      Routes
# ---------------------------------------------------------------------------


@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "environment": "AI Code Review OpenEnv", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset", summary="Reset the environment")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    task_id = req.task_id or "task_1"
    try:
        _env = CodeReviewEnvironment(task_id=task_id)
        obs = _env.reset(task_id=task_id)
        return {"observation": obs.dict(), "done": False, "info": {}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", summary="Take one step in the environment")
def step(req: StepRequest):
    env = _get_env()
    action = Action(
        action_type=req.action_type,
        target=req.target,
        description=req.description,
        line_number=req.line_number,
        extra=req.extra,
    )
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", summary="Get full internal state")
def state():
    env = _get_env()
    return env.state()


@app.get("/tasks", summary="List available tasks")
def list_tasks():
    from env.tasks import TASK_REGISTRY
    return {
        tid: {
            "title": t.title,
            "difficulty": t.difficulty,
            "goal": t.goal,
            "max_steps": t.max_steps,
            "num_bugs": len(t.bugs),
            "num_fixes": len(t.fixes),
            "num_improvements": len(t.improvements),
        }
        for tid, t in TASK_REGISTRY.items()
    }


@app.get("/tasks/{task_id}", summary="Get task details")
def get_task(task_id: str):
    from env.tasks import TASK_REGISTRY
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    t = TASK_REGISTRY[task_id]
    return {
        "task_id": t.task_id,
        "title": t.title,
        "difficulty": t.difficulty,
        "goal": t.goal,
        "code_snippet": t.code_snippet,
        "max_steps": t.max_steps,
        "num_bugs": len(t.bugs),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
