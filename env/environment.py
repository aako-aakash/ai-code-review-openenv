"""
AI Code Review – OpenEnv Environment
=====================================
Implements the three core OpenEnv interface methods:
  reset()  → Observation
  step()   → (Observation, float, bool, dict)
  state()  → dict
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from env.models import Action, BugRecord, Observation, TaskDefinition
from env.grader import Grader
from env.tasks import TASK_REGISTRY


class CodeReviewEnvironment:
    """
    OpenEnv-compliant environment that simulates an AI code-review agent
    working through a pull-request review task.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, task_id: str = "task_1"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            )
        self._task_id = task_id
        self._task: Optional[TaskDefinition] = None
        self._grader: Optional[Grader] = None
        self._reset_internal_state()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset the environment (optionally switching to a different task).
        Returns the initial Observation.
        """
        if task_id is not None:
            if task_id not in TASK_REGISTRY:
                raise ValueError(f"Unknown task_id '{task_id}'.")
            self._task_id = task_id

        self._task = deepcopy(TASK_REGISTRY[self._task_id])
        self._grader = Grader(self._task)
        self._reset_internal_state()
        return self._build_observation(message="Episode started. Review the code carefully.")

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one action.

        Returns
        -------
        observation : Observation
        reward      : float
        done        : bool
        info        : dict
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")

        # ---- validate action type -------------------------------------
        if not action.is_valid():
            reward = -0.05
            msg = (
                f"Unknown action_type '{action.action_type}'. "
                f"Valid types: {list(Action.__fields__['action_type'].field_info.description)}. "
                "(-0.05)"
            )
            self._steps_taken += 1
            self._cumulative_reward += reward
            self._record_history(action, reward, msg)
            obs = self._build_observation(message=msg)
            done = self._is_done()
            return obs, reward, done, self._info(reward, msg)

        # ---- dispatch -------------------------------------------------
        reward, msg = self._dispatch(action)
        self._steps_taken += 1
        self._cumulative_reward += reward
        self._record_history(action, reward, msg)

        done = self._is_done()
        obs = self._build_observation(message=msg)
        info = self._info(reward, msg)

        if done:
            grading = self._grader.evaluate(self.state())
            info["final_score"] = grading["score"]
            info["grading"] = grading

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return the full internal state as a plain dict."""
        return {
            "task_id": self._task_id,
            "current_code": self._task.code_snippet if self._task else "",
            "detected_issues": deepcopy(self._detected_issues),
            "suggested_fixes": deepcopy(self._suggested_fixes),
            "suggested_improvements": deepcopy(self._suggested_improvements),
            "review_history": deepcopy(self._review_history),
            "steps_taken": self._steps_taken,
            "max_steps": self._task.max_steps if self._task else 0,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "pr_approved": self._pr_approved,
            "pr_changes_requested": self._pr_changes_requested,
            "episode_done": self._is_done(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_internal_state(self):
        self._detected_issues: List[Dict[str, Any]] = []
        self._suggested_fixes: List[Dict[str, Any]] = []
        self._suggested_improvements: List[Dict[str, Any]] = []
        self._review_history: List[Dict[str, Any]] = []
        self._steps_taken: int = 0
        self._cumulative_reward: float = 0.0
        self._pr_approved: bool = False
        self._pr_changes_requested: bool = False
        self._analyzed: bool = False

    def _is_done(self) -> bool:
        if self._task is None:
            return True
        if self._steps_taken >= self._task.max_steps:
            return True
        if self._pr_approved or self._pr_changes_requested:
            return True
        return False

    def _dispatch(self, action: Action) -> Tuple[float, str]:
        t = action.action_type

        if t == "analyze_code":
            return self._act_analyze_code(action)
        elif t == "flag_bug":
            return self._act_flag_bug(action)
        elif t == "suggest_fix":
            return self._act_suggest_fix(action)
        elif t == "suggest_improvement":
            return self._act_suggest_improvement(action)
        elif t == "approve_pr":
            return self._act_approve_pr(action)
        elif t == "request_changes":
            return self._act_request_changes(action)
        else:
            return -0.05, "Unrecognised action (-0.05)."

    # -- Individual action handlers ------------------------------------

    def _act_analyze_code(self, action: Action) -> Tuple[float, str]:
        if self._analyzed:
            return -0.10, "Code already analysed – repeated action (-0.10)."
        self._analyzed = True
        return 0.05, (
            "Code analysis complete (+0.05). "
            "Use flag_bug, suggest_fix, suggest_improvement to document findings."
        )

    def _act_flag_bug(self, action: Action) -> Tuple[float, str]:
        already_flagged = [d["bug_id_matched"] for d in self._detected_issues if "bug_id_matched" in d]
        reward, msg = self._grader.reward_for_flag_bug(
            action.target, action.description, already_flagged
        )
        if reward > 0:
            # find which bug was matched
            from env.grader import _bug_matches
            bug_id_matched = None
            for bug in self._task.bugs:
                if _bug_matches(action.target, action.description, bug):
                    bug_id_matched = bug.bug_id
                    break
            entry = {
                "target": action.target,
                "description": action.description,
                "line_number": action.line_number,
                "bug_id_matched": bug_id_matched,
            }
            self._detected_issues.append(entry)
        return reward, msg

    def _act_suggest_fix(self, action: Action) -> Tuple[float, str]:
        already_fixed = [f["bug_id_matched"] for f in self._suggested_fixes if "bug_id_matched" in f]
        reward, msg = self._grader.reward_for_suggest_fix(
            action.target, action.description, already_fixed
        )
        if reward > 0:
            from env.grader import _fix_matches
            bug_id_matched = None
            for fix in self._task.fixes:
                if _fix_matches(action.description, fix):
                    bug_id_matched = fix.bug_id
                    break
            entry = {
                "target": action.target,
                "description": action.description,
                "bug_id_matched": bug_id_matched,
            }
            self._suggested_fixes.append(entry)
        return reward, msg

    def _act_suggest_improvement(self, action: Action) -> Tuple[float, str]:
        already_hit = [i["improvement_id_matched"] for i in self._suggested_improvements if "improvement_id_matched" in i]
        reward, msg = self._grader.reward_for_suggest_improvement(
            action.description, already_hit
        )
        if reward > 0:
            from env.grader import _improvement_matches
            imp_id_matched = None
            for imp in self._task.improvements:
                if _improvement_matches(action.description, imp):
                    imp_id_matched = imp.improvement_id
                    break
            entry = {
                "description": action.description,
                "improvement_id_matched": imp_id_matched,
            }
            self._suggested_improvements.append(entry)
        return reward, msg

    def _act_approve_pr(self, action: Action) -> Tuple[float, str]:
        if self._pr_approved or self._pr_changes_requested:
            return -0.05, "Review already closed."
        # Award completion bonus only if all bugs found
        found_bugs = len(self._detected_issues)
        total_bugs = len(self._task.bugs)
        if found_bugs == total_bugs:
            self._pr_approved = True
            return 1.0, "PR approved with all bugs detected! Completion bonus (+1.0)."
        self._pr_approved = True
        return 0.10, (
            f"PR approved, but only {found_bugs}/{total_bugs} bugs were detected. "
            "Partial approval (+0.10)."
        )

    def _act_request_changes(self, action: Action) -> Tuple[float, str]:
        if self._pr_approved or self._pr_changes_requested:
            return -0.05, "Review already closed."
        self._pr_changes_requested = True
        found_bugs = len(self._detected_issues)
        total_bugs = len(self._task.bugs)
        if found_bugs > 0:
            bonus = 0.50 * (found_bugs / total_bugs)
            return round(bonus, 3), (
                f"Changes requested with {found_bugs}/{total_bugs} bugs flagged. "
                f"Partial completion reward (+{bonus:.3f})."
            )
        return 0.0, "Changes requested but no bugs were flagged."

    # -- Utility --------------------------------------------------------

    def _build_observation(self, message: str = "") -> Observation:
        task = self._task
        return Observation(
            task_id=self._task_id,
            task_goal=task.goal if task else "",
            code_snippet=task.code_snippet if task else "",
            detected_issues=deepcopy(self._detected_issues),
            suggested_fixes=deepcopy(self._suggested_fixes),
            suggested_improvements=deepcopy(self._suggested_improvements),
            review_history=deepcopy(self._review_history),
            remaining_steps=(task.max_steps - self._steps_taken) if task else 0,
            pr_approved=self._pr_approved,
            pr_changes_requested=self._pr_changes_requested,
            cumulative_reward=round(self._cumulative_reward, 4),
            message=message,
        )

    def _record_history(self, action: Action, reward: float, msg: str):
        self._review_history.append(
            {
                "step": self._steps_taken,
                "action_type": action.action_type,
                "target": action.target,
                "description": action.description,
                "reward": round(reward, 4),
                "message": msg,
            }
        )

    def _info(self, reward: float, msg: str) -> Dict[str, Any]:
        return {
            "step_reward": round(reward, 4),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "steps_taken": self._steps_taken,
            "message": msg,
        }
