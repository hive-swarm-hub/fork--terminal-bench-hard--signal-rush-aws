#!/usr/bin/env python3
"""
Eval Run Analyzer — parallel multi-agent analysis of a terminal-bench-hard eval run.

Architecture:
  1. Inventory: scan job dir, classify tasks as pass/fail/timeout/pending
  2. Fan-out: dispatch parallel analyst agents (one per task cluster)
  3. Cross-check: fact-checker verifies claims against raw data
  4. Synthesize: produce actionable report

Usage:
    python analyze_eval.py [job_dir]
    python analyze_eval.py  # auto-picks latest
"""

import argparse
import json
import os
import sys
import time
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from claude_code_orchestrate import Agent, Bash, Read, AgentResult


# ─── Data types ──────────────────────────────────────────────────────────────

@dataclass
class TaskStatus:
    name: str
    dir_name: str
    path: str
    reward: float | None = None  # 1.0 = pass, 0.0 = fail, None = pending/error
    exception: str | None = None
    has_trial_log: bool = False
    has_trajectory: bool = False
    trial_log_lines: int = 0
    cost_usd: float | None = None
    n_episodes: int | None = None


def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(tag: str, msg: str):
    print(f"[{ts()}] [{tag}] {msg}", flush=True)


# ─── Phase 1: Inventory ─────────────────────────────────────────────────────

def inventory(job_dir: str) -> tuple[dict, list[TaskStatus]]:
    """Scan the job directory and classify all tasks."""
    result_path = os.path.join(job_dir, "result.json")
    with open(result_path) as f:
        run_result = json.load(f)

    stats = run_result["stats"]
    eval_key = list(stats["evals"].keys())[0]
    eval_data = stats["evals"][eval_key]

    # Build reward map
    reward_map = {}
    for score_str, task_ids in eval_data.get("reward_stats", {}).get("reward", {}).items():
        for tid in task_ids:
            task_name = tid.split("__")[0]
            reward_map[task_name] = float(score_str)

    # Build exception map
    exception_map = {}
    for exc_type, task_ids in eval_data.get("exception_stats", {}).items():
        for tid in task_ids:
            task_name = tid.split("__")[0]
            exception_map[task_name] = exc_type

    # Scan task dirs
    tasks = []
    for entry in sorted(os.listdir(job_dir)):
        entry_path = os.path.join(job_dir, entry)
        if not os.path.isdir(entry_path) or entry in (".", ".."):
            continue
        if "__" not in entry:
            continue

        task_name = entry.split("__")[0]
        trial_log = os.path.join(entry_path, "trial.log")
        trajectory = os.path.join(entry_path, "agent", "trajectory.json")

        log_lines = 0
        if os.path.exists(trial_log):
            with open(trial_log) as f:
                log_lines = sum(1 for _ in f)

        # Read per-task result for cost/episodes
        cost = None
        n_eps = None
        task_result = os.path.join(entry_path, "result.json")
        if os.path.exists(task_result):
            try:
                with open(task_result) as f:
                    tr = json.load(f)
                ar = tr.get("agent_result", {})
                cost = ar.get("cost_usd")
                md = ar.get("metadata", {})
                n_eps = md.get("n_episodes")
            except Exception:
                pass

        tasks.append(TaskStatus(
            name=task_name,
            dir_name=entry,
            path=entry_path,
            reward=reward_map.get(task_name),
            exception=exception_map.get(task_name),
            has_trial_log=os.path.exists(trial_log),
            has_trajectory=os.path.exists(trajectory),
            trial_log_lines=log_lines,
            cost_usd=cost,
            n_episodes=n_eps,
        ))

    summary = {
        "score": eval_data["metrics"][0]["mean"],
        "n_trials": eval_data["n_trials"],
        "n_errors": eval_data["n_errors"],
        "passed": [t.name for t in tasks if t.reward == 1.0],
        "failed": [t.name for t in tasks if t.reward == 0.0],
        "timed_out": [t.name for t in tasks if t.exception == "AgentTimeoutError"],
        "pending": [t.name for t in tasks if t.reward is None and t.exception is None],
        "total_cost": sum(t.cost_usd or 0 for t in tasks),
    }

    return summary, tasks


# ─── Phase 2: Parallel analysis ─────────────────────────────────────────────

def analyze_passed_tasks(tasks: list[TaskStatus], job_dir: str) -> AgentResult:
    """Analyze tasks that passed — what went right?"""
    task_block = "\n".join(
        f"- {t.name}: reward={t.reward}, episodes={t.n_episodes}, cost=${t.cost_usd:.2f}, log={t.trial_log_lines} lines"
        for t in tasks
    )
    return Agent(
        description="Analyze passed tasks",
        prompt=f"""You are analyzing terminal-bench-hard tasks that PASSED in the current eval run.

## Passed tasks
{task_block}

## Instructions
For each passed task:
1. Read the trial.log at {job_dir}/<dir_name>/trial.log
2. Read any trajectory at {job_dir}/<dir_name>/agent/trajectory.json (if it exists, just first 200 lines)
3. Read the task result at {job_dir}/<dir_name>/result.json

Determine:
- What strategy did the agent use?
- How many episodes did it take?
- Was there anything unusual (close calls, retries, resets)?
- What made this task succeed while others failed?

## Task directories
{chr(10).join(f'- {t.name}: {t.dir_name}' for t in tasks)}

Output a structured analysis per task, then a summary of what patterns lead to success.""",
        model="sonnet",
        on_tool_call=lambda tc: log("passed", f"  {tc.name}({list(tc.input.keys())})"),
    )


def analyze_timeout_tasks(tasks: list[TaskStatus], job_dir: str) -> AgentResult:
    """Analyze tasks that timed out — why did they stall?"""
    task_block = "\n".join(
        f"- {t.name}: exception={t.exception}, episodes={t.n_episodes}, cost=${t.cost_usd or 0:.2f}, log={t.trial_log_lines} lines"
        for t in tasks
    )
    return Agent(
        description="Analyze timeout tasks",
        prompt=f"""You are analyzing terminal-bench-hard tasks that TIMED OUT (AgentTimeoutError) in the current eval run.

## Timed-out tasks
{task_block}

## Instructions
For each timed-out task:
1. Read the trial.log at {job_dir}/<dir_name>/trial.log — focus on the LAST 100 lines to see what was happening when it timed out
2. Read the exception.txt at {job_dir}/<dir_name>/exception.txt if it exists
3. Read the task result at {job_dir}/<dir_name>/result.json for token/episode counts

Determine:
- Was the agent stuck in a loop? On what?
- Was it making progress but ran out of time?
- Did it call reset_terminal? How many times?
- Was there a stall (tail -f, heredoc, interactive prompt)?
- How many episodes did it complete before timing out?
- What was the last thing the agent was trying to do?

## Task directories
{chr(10).join(f'- {t.name}: {t.dir_name}' for t in tasks)}

Output per-task analysis with SPECIFIC log excerpts, then a summary of timeout patterns.""",
        model="sonnet",
        on_tool_call=lambda tc: log("timeout", f"  {tc.name}({list(tc.input.keys())})"),
    )


def analyze_failed_tasks(tasks: list[TaskStatus], job_dir: str) -> AgentResult:
    """Analyze tasks that failed (not timeout) — what went wrong?"""
    task_block = "\n".join(
        f"- {t.name}: reward=0.0, episodes={t.n_episodes}, cost=${t.cost_usd or 0:.2f}, log={t.trial_log_lines} lines"
        for t in tasks
    )
    return Agent(
        description="Analyze failed tasks",
        prompt=f"""You are analyzing terminal-bench-hard tasks that FAILED (reward=0.0, no timeout) in the current eval run.

## Failed tasks (completed but wrong answer)
{task_block}

## Instructions
For each failed task:
1. Read the trial.log at {job_dir}/<dir_name>/trial.log
2. Read the verifier output at {job_dir}/<dir_name>/verifier/ (ls the dir, read files)
3. Read the task result at {job_dir}/<dir_name>/result.json

Determine:
- What did the agent produce vs what was expected?
- How close was it to passing? (partial credit? almost-right output?)
- What was the agent's strategy?
- Was there a specific error or wrong approach?
- Could a small fix (prompt hint, infrastructure change) flip this task?

## Task directories
{chr(10).join(f'- {t.name}: {t.dir_name}' for t in tasks)}

Rank tasks by "closeness to passing" — which ones are most flippable?
Output per-task analysis, then a ranked list of flippability.""",
        model="sonnet",
        on_tool_call=lambda tc: log("failed", f"  {tc.name}({list(tc.input.keys())})"),
    )


def analyze_agent_code(job_dir: str) -> AgentResult:
    """Analyze the agent code that produced this run."""
    return Agent(
        description="Analyze agent code",
        prompt=f"""You are analyzing the agent code used in a terminal-bench-hard eval run.

## Instructions
1. Read /home/ubuntu/terminal-bench-hard/agent/agent.py (the full file, it's ~1900 lines)
2. Read /home/ubuntu/terminal-bench-hard/agent/prompt-templates/ (any files there)
3. Read the run config at {job_dir}/config.json
4. Read the job log at {job_dir}/job.log (first 50 and last 50 lines)

Determine:
- What agent version is this? (check git log for the latest commit message)
- What are the key features enabled? (reset_terminal, sanitize_command, stall detection, etc.)
- What's the execution model? (sequential markers, parallel pool, fast-path threshold)
- What's the prompt strategy? (minimal? domain hints? task-specific rules?)
- Are there any obvious bugs or misconfigurations?
- What's different from previous versions? (run: git log --oneline -5)

Output a concise code review focused on what could affect eval performance.""",
        model="sonnet",
        on_tool_call=lambda tc: log("code", f"  {tc.name}({list(tc.input.keys())})"),
    )


# ─── Phase 3: Cross-check ───────────────────────────────────────────────────

def cross_check(
    summary: dict,
    passed_analysis: AgentResult,
    timeout_analysis: AgentResult,
    failed_analysis: AgentResult,
    code_analysis: AgentResult,
) -> AgentResult:
    """Fact-check and cross-reference all analyses."""
    return Agent(
        description="Cross-check analyses",
        prompt=f"""You are a fact-checker for a terminal-bench-hard eval analysis. Verify claims against raw data.

## Run summary
Score: {summary['score']:.3f}
Passed: {summary['passed']}
Failed (non-timeout): {[t for t in summary['failed'] if t not in summary['timed_out']]}
Timed out: {summary['timed_out']}
Pending: {summary['pending']}
Total cost: ${summary['total_cost']:.2f}

## Passed tasks analysis
{passed_analysis.result}

## Timeout tasks analysis
{timeout_analysis.result}

## Failed tasks analysis
{failed_analysis.result}

## Code review
{code_analysis.result}

## Verification instructions
1. Check any specific score/episode/cost claims against the actual result.json files in /home/ubuntu/terminal-bench-hard/jobs/2026-04-03__03-36-05/
2. Verify code claims against /home/ubuntu/terminal-bench-hard/agent/agent.py
3. Check for contradictions between the four analyses
4. Verify any "close to passing" claims by reading the actual verifier output
5. Cross-reference timeout patterns — are the same root causes cited consistently?

Output:
```
VERIFIED: <claim> — confirmed by <source>
REFUTED: <claim> — actual: <truth>
CONTRADICTIONS: <what two analyses disagree on>
QUALITY SCORE: X/10
KEY CORRECTIONS: <what the final report should fix>
```""",
        model="sonnet",
        on_tool_call=lambda tc: log("check", f"  {tc.name}({list(tc.input.keys())})"),
    )


# ─── Phase 4: Synthesize ────────────────────────────────────────────────────

def synthesize(
    summary: dict,
    passed_analysis: AgentResult,
    timeout_analysis: AgentResult,
    failed_analysis: AgentResult,
    code_analysis: AgentResult,
    cross_check_result: AgentResult,
) -> AgentResult:
    """Produce the final actionable report."""
    return Agent(
        description="Final synthesis",
        prompt=f"""You are producing the final analysis report for a terminal-bench-hard eval run.

## Run summary
Score: {summary['score']:.3f} ({len(summary['passed'])}/{summary['n_trials']})
Passed: {summary['passed']}
Timed out: {summary['timed_out']}
Total cost: ${summary['total_cost']:.2f}

## Analysis results (4 parallel analysts + cross-checker)

### Passed tasks
{passed_analysis.result}

### Timeout tasks
{timeout_analysis.result}

### Failed tasks
{failed_analysis.result}

### Code review
{code_analysis.result}

### Cross-check corrections
{cross_check_result.result}

## Analysis metadata
- Passed analyst: {passed_analysis.num_turns} turns, {passed_analysis.total_tool_calls} tools, ${passed_analysis.total_cost_usd or 0:.3f}
- Timeout analyst: {timeout_analysis.num_turns} turns, {timeout_analysis.total_tool_calls} tools, ${timeout_analysis.total_cost_usd or 0:.3f}
- Failed analyst: {failed_analysis.num_turns} turns, {failed_analysis.total_tool_calls} tools, ${failed_analysis.total_cost_usd or 0:.3f}
- Code analyst: {code_analysis.num_turns} turns, {code_analysis.total_tool_calls} tools, ${code_analysis.total_cost_usd or 0:.3f}
- Cross-checker: {cross_check_result.num_turns} turns, {cross_check_result.total_tool_calls} tools, ${cross_check_result.total_cost_usd or 0:.3f}

## Instructions
Produce a final report with ONLY fact-checked findings. Include:

1. **RUN VERDICT** — is this run better/worse/same as previous? Compare to V10 mean of 0.177
2. **WHAT WORKED** — why did the passing tasks pass?
3. **TIMEOUT ROOT CAUSES** — categorize the 6 timeouts by pattern
4. **CLOSEST TO FLIPPING** — rank failed tasks by flippability with specific fixes
5. **AGENT ISSUES** — any bugs or misconfigurations found in the code
6. **RECOMMENDED CHANGES** — concrete, ranked, with expected impact

Write for a developer who will implement fixes in the next iteration.""",
        model="opus",
        on_tool_call=lambda tc: log("synth", f"  {tc.name}({list(tc.input.keys())})"),
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", nargs="?", help="Job directory to analyze")
    args = parser.parse_args()

    if args.job_dir:
        job_dir = args.job_dir
    else:
        jobs_root = "/home/ubuntu/terminal-bench-hard/jobs"
        latest = sorted(os.listdir(jobs_root))[-1]
        job_dir = os.path.join(jobs_root, latest)

    pipeline_start = time.time()
    print(f"{'=' * 70}")
    print(f"  Eval Run Analyzer")
    print(f"  Job: {job_dir}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    # Phase 1: Inventory
    log("inventory", "scanning job directory...")
    summary, tasks = inventory(job_dir)

    passed = [t for t in tasks if t.reward == 1.0]
    timed_out = [t for t in tasks if t.exception == "AgentTimeoutError"]
    failed_no_timeout = [t for t in tasks if t.reward == 0.0 and t.exception != "AgentTimeoutError"]
    pending = [t for t in tasks if t.reward is None and t.exception is None]

    print(f"\n  Score:    {summary['score']:.3f}")
    print(f"  Passed:   {len(passed)} — {[t.name for t in passed]}")
    print(f"  Timeout:  {len(timed_out)} — {[t.name for t in timed_out]}")
    print(f"  Failed:   {len(failed_no_timeout)} — {[t.name for t in failed_no_timeout]}")
    print(f"  Pending:  {len(pending)} — {[t.name for t in pending]}")
    print(f"  Cost:     ${summary['total_cost']:.2f}")

    # Phase 2: Parallel analysis (4 agents)
    log("analyze", "dispatching 4 parallel analysts...")
    phase2_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        f_passed = pool.submit(analyze_passed_tasks, passed, job_dir) if passed else None
        f_timeout = pool.submit(analyze_timeout_tasks, timed_out, job_dir) if timed_out else None
        f_failed = pool.submit(analyze_failed_tasks, failed_no_timeout, job_dir) if failed_no_timeout else None
        f_code = pool.submit(analyze_agent_code, job_dir)

        # Collect results with progress
        results = {}
        for name, future in [("passed", f_passed), ("timeout", f_timeout), ("failed", f_failed), ("code", f_code)]:
            if future is None:
                results[name] = Agent(description="noop", prompt="Say: No tasks in this category.", model="haiku")
                log("analyze", f"  {name}: skipped (no tasks)")
            else:
                r = future.result()
                elapsed = time.time() - phase2_start
                log("analyze", f"  {name}: done in {elapsed:.0f}s — {r.num_turns} turns, {r.total_tool_calls} tools, ${r.total_cost_usd or 0:.3f}")
                results[name] = r

    log("analyze", f"all analysts done in {time.time() - phase2_start:.0f}s")

    # Phase 3: Cross-check
    log("check", "dispatching cross-checker...")
    check_start = time.time()
    check_result = cross_check(
        summary, results["passed"], results["timeout"], results["failed"], results["code"]
    )
    log("check", f"done in {time.time() - check_start:.0f}s — {check_result.num_turns} turns, {check_result.total_tool_calls} tools")

    # Phase 4: Synthesize
    log("synth", "dispatching final synthesis (opus)...")
    synth_start = time.time()
    report = synthesize(
        summary, results["passed"], results["timeout"], results["failed"], results["code"], check_result
    )
    log("synth", f"done in {time.time() - synth_start:.0f}s")

    # Output
    elapsed = time.time() - pipeline_start
    total_analysis_cost = sum(
        r.total_cost_usd or 0
        for r in [results["passed"], results["timeout"], results["failed"], results["code"], check_result, report]
    )

    print(f"\n{'█' * 70}")
    print(f"  FINAL REPORT")
    print(f"{'█' * 70}")
    print(report.result)

    # Save
    output_path = os.path.join(os.path.dirname(job_dir), "eval_analysis.md")
    with open(output_path, "w") as f:
        f.write(f"# Eval Run Analysis: {os.path.basename(job_dir)}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report.result)
        f.write(f"\n\n---\n\n## Analysis Metadata\n\n")
        f.write(f"- Pipeline time: {elapsed:.0f}s\n")
        f.write(f"- Analysis cost: ${total_analysis_cost:.3f}\n")
        f.write(f"- Agents: 4 analysts + 1 cross-checker + 1 synthesizer\n\n")
        for name, r in results.items():
            f.write(f"### {name.title()} Analyst\n")
            f.write(f"Turns: {r.num_turns}, Tools: {r.total_tool_calls}, Cost: ${r.total_cost_usd or 0:.3f}\n")
            f.write(f"Tools used: {r.tool_names}\n\n")
            f.write(f"{r.result}\n\n")
        f.write(f"### Cross-Check\n{check_result.result}\n\n")

    print(f"\n{'=' * 70}")
    print(f"  Pipeline: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Analysis cost: ${total_analysis_cost:.3f}")
    print(f"  Report: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
