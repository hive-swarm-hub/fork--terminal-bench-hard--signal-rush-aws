#!/usr/bin/env python3
"""
Failure Analyzer — dispatch one agent per failed task, then review + synthesize.

Pipeline:
  1. Inventory failed tasks from latest eval
  2. Fan-out: one analyst agent per failed task (parallel)
  3. Reviewer: cross-checks all analyses, finds contradictions and patterns
  4. Synthesizer: produces root-cause report, looking for systemic bugs
"""

import json
import os
import sys
import time
import concurrent.futures
from datetime import datetime

from claude_code_orchestrate import Agent, Bash


def ts():
    return datetime.now().strftime("%H:%M:%S")

def log(tag, msg):
    print(f"[{ts()}] [{tag}] {msg}", flush=True)


def get_job_dir():
    """Find the latest job directory."""
    jobs = "/home/ubuntu/terminal-bench-hard/jobs"
    latest = sorted(os.listdir(jobs))[-1]
    return os.path.join(jobs, latest)


def get_failed_tasks(job_dir):
    """Extract failed tasks from result.json."""
    with open(os.path.join(job_dir, "result.json")) as f:
        d = json.load(f)
    e = list(d["stats"]["evals"].values())[0]

    failed_ids = e.get("reward_stats", {}).get("reward", {}).get("0.0", [])
    exceptions = {}
    for exc, tids in e.get("exception_stats", {}).items():
        for tid in tids:
            exceptions[tid.split("__")[0]] = exc

    tasks = []
    for tid in failed_ids:
        name = tid.split("__")[0]
        task_dir = os.path.join(job_dir, tid)
        tasks.append({
            "name": name,
            "tid": tid,
            "dir": task_dir,
            "exception": exceptions.get(name),
        })
    return tasks, e["metrics"][0]["mean"], e["n_trials"], e["n_errors"]


def analyze_one_task(task, job_dir):
    """Dispatch one analyst agent for a single failed task."""
    name = task["name"]
    d = task["dir"]
    exc = task["exception"]

    exc_info = f"Exception: {exc}" if exc else "No exception (completed but wrong answer)"

    r = Agent(
        description=f"Analyze {name}",
        prompt=f"""You are analyzing why the task "{name}" FAILED in a terminal-bench eval run.
{exc_info}

## Your task
Find the ROOT CAUSE of failure. Not symptoms — the actual bug or wrong decision.

## Files to read (in this order)
1. {d}/trial.log — the full execution log. Read ALL of it.
2. {d}/result.json — agent metrics (tokens, episodes, cost)
3. {d}/exception.txt — if it exists, the exception traceback
4. {d}/verifier/ — ls this dir, then read any output files (this shows what the verifier expected vs got)
5. {d}/agent/ — ls this dir, check for trajectory.json (read first 100 + last 100 lines if large)
6. {d}/artifacts/ — ls this dir to see what the agent produced

## What to determine
- What was the agent trying to do when it failed?
- If timeout: was it stuck in a loop? On what command? How many episodes did it complete?
- If wrong answer: what did it produce vs what was expected? How close was it?
- Did the agent use reset_terminal? How many times?
- Were there stalls (tail -f, heredoc, interactive prompt)?
- Was the agent's strategy reasonable or fundamentally wrong?
- Is there a BUG in the agent code that caused this? (check for patterns like broken command execution, missing output, garbled markers)

## Output format
```
TASK: {name}
FAILURE TYPE: timeout | wrong_answer | crash | infrastructure
ROOT CAUSE: <one sentence>
EVIDENCE: <specific log lines or verifier output>
AGENT STRATEGY: <what approach did it take>
EPISODES COMPLETED: N
CLOSENESS TO PASSING: far | close | very_close
FLIPPABLE BY: <what specific change would fix this>
POSSIBLE AGENT BUG: <yes/no — if yes, describe>
```

Be SPECIFIC. Quote log lines. Name exact commands that failed.""",
        model="sonnet",
        on_tool_call=lambda tc, n=name: log(n[:15], f"{tc.name}({list(tc.input.keys())})"),
    )
    return name, r


def review_analyses(analyses, job_dir, score, n_trials, n_errors):
    """Reviewer agent: cross-check all analyses, find patterns and bugs."""
    analysis_block = ""
    for name, r in analyses:
        analysis_block += f"\n{'='*60}\n## {name}\nTurns: {r.num_turns}, Tools: {r.total_tool_calls}, Cost: ${r.total_cost_usd or 0:.3f}\n{'='*60}\n{r.result}\n"

    return Agent(
        description="Review all analyses",
        prompt=f"""You are a senior reviewer examining {len(analyses)} failure analyses from a terminal-bench eval run.

## Run stats
Score: {score:.3f} ({n_trials} trials, {n_errors} errors)

## All task analyses
{analysis_block}

## Your job
1. Read the ACTUAL agent code at /home/ubuntu/terminal-bench-hard/agent/agent.py
2. For each analysis, VERIFY the root cause claim by reading the actual trial.log
3. Look for PATTERNS across failures:
   - Do multiple tasks fail for the SAME reason?
   - Is there a systemic bug in the agent code causing multiple failures?
   - Are the analysts' "flippable by" suggestions consistent?
4. Check if any analyst MISSED something obvious
5. Look for agent code bugs that could explain MULTIPLE failures at once

## Specific things to verify
- Read /home/ubuntu/terminal-bench-hard/agent/agent.py and check:
  - _execute_commands: any bugs in the marker polling or fast-path logic?
  - _sanitize_command: does it break any commands?
  - _reset_terminal: does it leave the session in a bad state?
  - The prompt template: any instructions that could mislead the agent?
- Run `git log --oneline -10` to see what version this is
- Check the run config at {job_dir}/config.json

## Output format
```
PATTERN ANALYSIS:
- Pattern 1: <N tasks share this failure pattern> — <description>
- Pattern 2: ...

AGENT BUGS FOUND:
- Bug 1: <file:line> — <description> — affects tasks: [...]
- Bug 2: ...

ANALYST CORRECTIONS:
- <task>: analyst said X but actual cause is Y

VERIFIED ROOT CAUSES:
- <task>: <confirmed root cause>

SYSTEMIC ISSUES:
- <issue that affects many tasks>

PRIORITY FIXES:
1. <fix> — would unblock N tasks
2. <fix> — would unblock N tasks
```""",
        model="opus",
        on_tool_call=lambda tc: log("reviewer", f"{tc.name}({list(tc.input.keys())})"),
    )


def synthesize(analyses, review, score, n_trials, n_errors):
    """Final synthesis — actionable bug report."""
    analysis_summaries = "\n".join(
        f"- {name}: {r.result[:200]}..." for name, r in analyses
    )

    return Agent(
        description="Synthesize findings",
        prompt=f"""You are producing the final failure analysis report.

## Run stats
Score: {score:.3f} ({n_trials} trials, {n_errors} errors)

## Reviewer's verified findings
{review.result}

## Individual task analyses (summaries)
{analysis_summaries}

## Instructions
Produce a concise, actionable report:

1. **SYSTEMIC BUGS** — bugs in agent code that cause MULTIPLE failures. Include file, line, and fix.
2. **FAILURE TAXONOMY** — group all failures by root cause type
3. **QUICK WINS** — tasks closest to passing, ranked by effort to fix
4. **RECOMMENDED FIXES** — ordered by impact (most tasks unblocked first)

Keep it under 2000 words. Be concrete — file paths, line numbers, specific changes.
Only include findings verified by the reviewer.""",
        model="sonnet",
    )


def main():
    start = time.time()
    job_dir = get_job_dir()

    print(f"{'=' * 70}")
    print(f"  Failure Analyzer — {job_dir}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    # Phase 1: Inventory
    log("init", "scanning for failed tasks...")
    tasks, score, n_trials, n_errors = get_failed_tasks(job_dir)

    if not tasks:
        print("No failed tasks found!")
        return

    print(f"\n  Score: {score:.3f} ({n_trials} trials, {n_errors} errors)")
    print(f"  Failed tasks ({len(tasks)}):")
    for t in tasks:
        exc = f" [{t['exception']}]" if t['exception'] else ""
        print(f"    {t['name']}{exc}")

    # Phase 2: One agent per failed task (parallel)
    log("analyze", f"dispatching {len(tasks)} parallel analyst agents...")
    phase2_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), 7)) as pool:
        futures = {
            pool.submit(analyze_one_task, t, job_dir): t["name"]
            for t in tasks
        }
        analyses = []
        for future in concurrent.futures.as_completed(futures):
            tname = futures[future]
            try:
                name, r = future.result()
                log("analyze", f"  {name} DONE — {r.num_turns} turns, {r.total_tool_calls} tools, ${r.total_cost_usd or 0:.3f}")
                analyses.append((name, r))
            except Exception as e:
                log("analyze", f"  {tname} FAILED: {e}")
                # Create a stub result
                stub = Agent(description="stub", prompt=f"Say: Analysis of {tname} failed with error: {e}", model="haiku")
                analyses.append((tname, stub))

    analyses.sort(key=lambda x: x[0])
    log("analyze", f"all analysts done in {time.time() - phase2_start:.0f}s")

    # Print previews
    for name, r in analyses:
        preview = r.result[:150].replace("\n", " ")
        print(f"\n  [{name}] {preview}...")

    # Phase 3: Reviewer (opus)
    log("review", "dispatching reviewer agent (opus)...")
    review_start = time.time()
    review = review_analyses(analyses, job_dir, score, n_trials, n_errors)
    log("review", f"done in {time.time() - review_start:.0f}s — {review.num_turns} turns, {review.total_tool_calls} tools, ${review.total_cost_usd or 0:.3f}")

    # Phase 4: Synthesize
    log("synth", "synthesizing final report...")
    synth_start = time.time()
    report = synthesize(analyses, review, score, n_trials, n_errors)
    log("synth", f"done in {time.time() - synth_start:.0f}s")

    # Output
    elapsed = time.time() - start
    total_cost = sum(r.total_cost_usd or 0 for _, r in analyses) + (review.total_cost_usd or 0) + (report.total_cost_usd or 0)

    print(f"\n{'█' * 70}")
    print(f"  FAILURE ANALYSIS REPORT")
    print(f"{'█' * 70}\n")
    print(report.result)

    # Also print full reviewer findings
    print(f"\n{'─' * 70}")
    print(f"  REVIEWER FINDINGS (detailed)")
    print(f"{'─' * 70}\n")
    print(review.result)

    # Save
    output_path = os.path.join(os.path.dirname(job_dir), "failure_analysis.md")
    with open(output_path, "w") as f:
        f.write(f"# Failure Analysis: {os.path.basename(job_dir)}\n\n")
        f.write(f"Score: {score:.3f}, {len(tasks)} failures analyzed\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Synthesized Report\n\n")
        f.write(report.result)
        f.write("\n\n## Reviewer Findings\n\n")
        f.write(review.result)
        f.write("\n\n## Per-Task Analyses\n\n")
        for name, r in analyses:
            f.write(f"### {name}\n")
            f.write(f"Turns: {r.num_turns}, Tools: {r.total_tool_calls}, Cost: ${r.total_cost_usd or 0:.3f}\n\n")
            f.write(r.result)
            f.write("\n\n")

    print(f"\n{'=' * 70}")
    print(f"  Pipeline: {elapsed:.0f}s ({elapsed/60:.1f}min), Cost: ${total_cost:.3f}")
    print(f"  Report: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
