"""
Deterministic grader for the AI Code Review environment.

Scoring formula (per episode):
  score = (bugs_detected_correctly / total_bugs)        * 0.50
        + (fixes_correct / total_fixes)                 * 0.30
        + (improvements_found / expected_improvements)  * 0.20

All comparisons are case-insensitive substring / keyword checks so the
grader is robust to minor wording differences while remaining deterministic.
"""

from __future__ import annotations
from typing import Any, Dict, List
from env.models import TaskDefinition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return text.lower().strip()


def _bug_matches(agent_target: str | None, agent_description: str | None, bug) -> bool:
    """
    Return True if the agent's flag_bug action matches *bug*.
    Match rules (any one is sufficient):
      1. agent_target == bug.bug_id  (exact)
      2. bug.bug_id substring appears in agent_target or agent_description
      3. any word from bug.description appears in agent_description (>= 3 chars)
    """
    target = _normalise(agent_target or "")
    desc = _normalise(agent_description or "")

    if bug.bug_id in target:
        return True
    if bug.bug_id in desc:
        return True

    # keyword overlap (ignore short stop-words)
    bug_words = {w for w in _normalise(bug.description).split() if len(w) >= 4}
    agent_words = set((target + " " + desc).split())
    overlap = bug_words & agent_words
    return len(overlap) >= 2


def _fix_matches(agent_description: str | None, fix) -> bool:
    """Return True if the agent's suggested fix matches the canonical fix."""
    desc = _normalise(agent_description or "")
    return _normalise(fix.correct_fix) in desc


def _improvement_matches(agent_description: str | None, improvement) -> bool:
    """Return True if the agent's improvement suggestion contains ≥1 keyword."""
    desc = _normalise(agent_description or "")
    return any(_normalise(kw) in desc for kw in improvement.keywords)


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------


class Grader:
    """Scores a completed (or in-progress) episode against the task ground truth."""

    def __init__(self, task: TaskDefinition):
        self.task = task

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the final (or current) state of an episode.

        Parameters
        ----------
        state : dict
            The dict returned by environment.state() – must contain
            ``detected_issues``, ``suggested_fixes``, ``suggested_improvements``.

        Returns
        -------
        dict with keys:
            score            – float in [0, 1]
            bug_score        – float in [0, 1]
            fix_score        – float in [0, 1]
            improvement_score– float in [0, 1]
            bugs_hit         – list of bug_ids correctly detected
            fixes_hit        – list of bug_ids correctly fixed
            improvements_hit – list of improvement_ids hit
            details          – human-readable breakdown
        """
        task = self.task

        detected: List[Dict] = state.get("detected_issues", [])
        fixes: List[Dict] = state.get("suggested_fixes", [])
        improvements: List[Dict] = state.get("suggested_improvements", [])

        # ---- bugs --------------------------------------------------------
        bugs_hit: List[str] = []
        for bug in task.bugs:
            for entry in detected:
                if _bug_matches(entry.get("target"), entry.get("description"), bug):
                    if bug.bug_id not in bugs_hit:
                        bugs_hit.append(bug.bug_id)
                    break

        # ---- fixes -------------------------------------------------------
        fixes_hit: List[str] = []
        for fix in task.fixes:
            for entry in fixes:
                if _fix_matches(entry.get("description"), fix):
                    if fix.bug_id not in fixes_hit:
                        fixes_hit.append(fix.bug_id)
                    break

        # ---- improvements ------------------------------------------------
        improvements_hit: List[str] = []
        for imp in task.improvements:
            for entry in improvements:
                if _improvement_matches(entry.get("description"), imp):
                    if imp.improvement_id not in improvements_hit:
                        improvements_hit.append(imp.improvement_id)
                    break

        # ---- sub-scores --------------------------------------------------
        total_bugs = max(len(task.bugs), 1)
        total_fixes = max(len(task.fixes), 1)
        total_imps = max(len(task.improvements), 1)

        bug_score = len(bugs_hit) / total_bugs
        fix_score = len(fixes_hit) / total_fixes
        imp_score = len(improvements_hit) / total_imps

        score = round(bug_score * 0.50 + fix_score * 0.30 + imp_score * 0.20, 4)

        details = (
            f"Bugs detected: {len(bugs_hit)}/{total_bugs} | "
            f"Fixes correct: {len(fixes_hit)}/{total_fixes} | "
            f"Improvements: {len(improvements_hit)}/{total_imps} | "
            f"Score: {score:.4f}"
        )

        return {
            "score": score,
            "bug_score": round(bug_score, 4),
            "fix_score": round(fix_score, 4),
            "improvement_score": round(imp_score, 4),
            "bugs_hit": bugs_hit,
            "fixes_hit": fixes_hit,
            "improvements_hit": improvements_hit,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Per-action reward helpers (called by environment.step)
    # ------------------------------------------------------------------

    def reward_for_flag_bug(
        self,
        target: str | None,
        description: str | None,
        already_flagged_ids: List[str],
    ) -> tuple[float, str]:
        """Return (reward, message) for a flag_bug action."""
        for bug in self.task.bugs:
            if _bug_matches(target, description, bug):
                if bug.bug_id in already_flagged_ids:
                    return -0.10, f"Bug '{bug.bug_id}' already flagged – duplicate action."
                return 0.20, f"Correct! Bug '{bug.bug_id}' detected (+0.20)."
        return -0.10, "Incorrect bug flag – no matching ground-truth bug found (-0.10)."

    def reward_for_suggest_fix(
        self,
        target: str | None,
        description: str | None,
        already_fixed_ids: List[str],
    ) -> tuple[float, str]:
        """Return (reward, message) for a suggest_fix action."""
        for fix in self.task.fixes:
            if _fix_matches(description, fix):
                if fix.bug_id in already_fixed_ids:
                    return -0.10, f"Fix for '{fix.bug_id}' already suggested – duplicate."
                return 0.30, f"Correct fix for bug '{fix.bug_id}' (+0.30)."
        return -0.10, "Incorrect fix suggestion – does not match any expected fix (-0.10)."

    def reward_for_suggest_improvement(
        self,
        description: str | None,
        already_hit_ids: List[str],
    ) -> tuple[float, str]:
        """Return (reward, message) for a suggest_improvement action."""
        for imp in self.task.improvements:
            if _improvement_matches(description, imp):
                if imp.improvement_id in already_hit_ids:
                    return -0.05, "Improvement already suggested."
                return 0.10, f"Good improvement suggestion '{imp.improvement_id}' (+0.10)."
        return -0.05, "Improvement suggestion not matching expected themes (-0.05)."
