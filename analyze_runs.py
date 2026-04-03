#!/usr/bin/env python3
"""
Terminal-Bench-Hard Run Analyzer
================================
Multi-agent orchestration that:
1. Pulls top runs from the hive leaderboard
2. Dispatches analyst agents to study what's working / failing
3. Runs a structured debate between analysts
4. Fact-checks every claim against real data
5. Synthesizes actionable findings

Usage:
    pip install -e /home/ubuntu/claude-code-orchestrate
    python analyze_runs.py [--rounds 2] [--top 5]
"""

import argparse
import json
import sys
import time
import concurrent.futures
import textwrap
from datetime import datetime
from claude_code_orchestrate import Agent, Bash, Read


def ts():
    """Timestamp prefix for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def banner(text: str, char: str = "=", width: int = 70):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def section(text: str):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def log(tag: str, msg: str):
    print(f"[{ts()}] [{tag}] {msg}", flush=True)


def preview(text: str, label: str, max_lines: int = 15):
    """Print a preview of a long text block."""
    lines = text.strip().split("\n")
    print(f"\n  ┌── {label} ({len(lines)} lines, {len(text)} chars) ──")
    for line in lines[:max_lines]:
        print(f"  │ {line}")
    if len(lines) > max_lines:
        print(f"  │ ... ({len(lines) - max_lines} more lines)")
    print(f"  └{'─' * 50}")


# ─── Phase 1: Collect top runs from hive ────────────────────────────────────

def fetch_top_runs(n: int = 5) -> list[dict]:
    """Pull the top N runs from the hive leaderboard."""
    log("collect", f"fetching top {n} runs from hive leaderboard...")
    raw = Bash(command=f"cd /home/ubuntu/terminal-bench-hard && hive run list --json --per-page {n}")
    data = json.loads(raw)
    runs = data.get("runs", [])
    log("collect", f"got {len(runs)} runs")
    print()
    print(f"  {'ID':<14} {'Score':<8} {'Agent':<20} {'TLDR'}")
    print(f"  {'─'*14} {'─'*8} {'─'*20} {'─'*40}")
    for r in runs:
        print(f"  {r['id'][:12]:<14} {str(r.get('score','?')):<8} {r.get('agent_id','?'):<20} {r.get('tldr','')[:50]}")
    print()
    return runs


def fetch_feed_context() -> str:
    """Pull recent feed posts for shared context."""
    log("collect", "fetching recent feed posts...")
    result = Bash(command="cd /home/ubuntu/terminal-bench-hard && hive feed list --since 7d 2>/dev/null || echo '(no feed)'")
    lines = result.strip().split("\n")
    log("collect", f"got {len(lines)} lines of feed context")
    return result


# ─── Phase 2: Analyst agents ────────────────────────────────────────────────

def build_analyst_prompt(analyst_id: int, assigned_runs: list[dict], all_runs_summary: str, feed: str) -> str:
    run_block = ""
    for r in assigned_runs:
        run_block += textwrap.dedent(f"""\
        ### Run {r['id'][:12]}
        - Agent: {r.get('agent_id')}
        - Score: {r.get('score')}
        - Branch: {r.get('branch')}
        - TLDR: {r.get('tldr', 'N/A')}
        - Fork URL: {r.get('fork_url', 'N/A')}
        - Verified: {r.get('verified')}

        """)

    return textwrap.dedent(f"""\
    You are **Analyst #{analyst_id}**, studying terminal-bench-hard runs.

    ## Your mission
    Figure out what techniques/changes are **working** (leading to higher scores)
    and what is **not working** (leading to regressions or no improvement).

    ## Leaderboard overview
    {all_runs_summary}

    ## Your assigned runs to deep-dive
    {run_block}

    ## Recent community feed
    {feed}

    ## Instructions
    1. For each assigned run, use `hive run view <id>` to get fork URL and branch details.
    2. If a fork URL is available, try to read the agent code at that fork
       (clone or fetch the branch and diff against the base).
       Focus on `agent/agent.py` and `agent/prompt-templates/`.
    3. Read any local job results in /home/ubuntu/terminal-bench-hard/jobs/ that match.
    4. Read /home/ubuntu/terminal-bench-hard/tasks.md to understand which tasks are hard.
    5. Read /home/ubuntu/terminal-bench-hard/program.md for constraints.

    ## Output format
    Produce a structured analysis:

    ```
    FINDINGS:
    - [WORKS] <technique> — evidence: <what you saw>
    - [FAILS] <technique> — evidence: <what you saw>
    - [UNCLEAR] <technique> — evidence: <what you saw>

    HYPOTHESES:
    - H1: <hypothesis about what drives score improvements>
    - H2: ...

    KEY DIFFS:
    - <run_id>: <summary of what changed vs parent/base>

    RECOMMENDATIONS:
    - <actionable suggestion>
    ```

    Be specific. Cite file paths, line numbers, and concrete code patterns.
    Do NOT speculate without evidence — mark uncertain claims as [UNCLEAR].
    """)


def run_analysts(runs: list[dict], feed: str, n_analysts: int = 2) -> list[str]:
    all_summary = "\n".join(
        f"  {r['id'][:12]}  score={r.get('score')}  agent={r.get('agent_id')}  tldr={r.get('tldr','')[:50]}"
        for r in runs
    )

    assignments: list[list[dict]] = [[] for _ in range(n_analysts)]
    for i, run in enumerate(runs):
        assignments[i % n_analysts].append(run)

    section(f"PHASE 2: Dispatching {n_analysts} analyst agents")
    for idx in range(n_analysts):
        run_ids = [r['id'][:12] for r in assignments[idx]]
        log("analysts", f"analyst #{idx+1} assigned runs: {run_ids}")

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_analysts) as pool:
        futures = {}
        for idx in range(n_analysts):
            prompt = build_analyst_prompt(idx + 1, assignments[idx], all_summary, feed)
            log("analysts", f"launching analyst #{idx+1} ({len(assignments[idx])} runs, prompt={len(prompt)} chars)...")
            f = pool.submit(
                Agent,
                description=f"Analyst #{idx+1}: study {len(assignments[idx])} runs",
                prompt=prompt,
                model="sonnet",
            )
            futures[f] = idx + 1

        results = []
        for future in concurrent.futures.as_completed(futures):
            aid = futures[future]
            elapsed = time.time() - start
            result = future.result()
            log("analysts", f"analyst #{aid} DONE in {elapsed:.0f}s ({len(result)} chars)")
            preview(result, f"Analyst #{aid} findings", max_lines=20)
            results.append((aid, result))

    results.sort(key=lambda x: x[0])
    log("analysts", f"all analysts complete in {time.time() - start:.0f}s total")
    return [r[1] for r in results]


# ─── Phase 3: Debate rounds ─────────────────────────────────────────────────

def run_debate_round(
    round_num: int,
    analyst_findings: list[str],
    previous_debate: str,
    fact_check_report: str,
) -> tuple[str, str]:
    n = len(analyst_findings)
    findings_block = ""
    for i, f in enumerate(analyst_findings):
        findings_block += f"\n{'='*60}\n## Analyst #{i+1} Findings\n{'='*60}\n{f}\n"

    context = f"""
## Previous debate context
{previous_debate if previous_debate else "(This is the first round — no prior debate.)"}

## Fact-checker corrections from prior round
{fact_check_report if fact_check_report else "(First round — no prior fact-check.)"}
"""

    debater_prompts = []
    for i in range(n):
        stance = "advocate" if i % 2 == 0 else "skeptic"
        debater_prompts.append(textwrap.dedent(f"""\
        You are **Debater #{i+1}** (stance: {stance}) in round {round_num} of a
        structured debate about what's working in terminal-bench-hard.

        Your role as {stance}:
        {"- Champion the strongest findings and push for bold conclusions." if stance == "advocate" else "- Challenge weak evidence, point out confounders, demand rigor."}

        ## All analyst findings
        {findings_block}

        {context}

        ## Instructions
        1. Review ALL analyst findings (not just your own).
        2. Identify the **top 3 agreements** across analysts.
        3. Identify the **top 3 disagreements** or contradictions.
        4. As a {stance}, argue your position on the disagreements.
        5. Propose **specific experiments** to resolve open questions.

        ## Output format
        ```
        AGREEMENTS:
        - A1: <finding all analysts agree on> — strength: HIGH/MEDIUM/LOW

        DISAGREEMENTS:
        - D1: <point of contention> — my position: <your argument>

        EXPERIMENTS TO RUN:
        - E1: <specific testable hypothesis and how to test it>

        SYNTHESIS:
        <2-3 paragraph summary of your overall position>
        ```
        """))

    # ── Dispatch debaters in parallel ──
    section(f"DEBATE ROUND {round_num}: Dispatching {n} debaters")
    for i in range(n):
        stance = "advocate" if i % 2 == 0 else "skeptic"
        log("debate", f"debater #{i+1} ({stance}) prompt = {len(debater_prompts[i])} chars")

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(
                Agent,
                description=f"R{round_num} Debater #{i+1}",
                prompt=debater_prompts[i],
                model="sonnet",
            ): i + 1
            for i in range(n)
        }
        debate_parts = []
        for future in concurrent.futures.as_completed(futures):
            did = futures[future]
            elapsed = time.time() - start
            result = future.result()
            log("debate", f"debater #{did} DONE in {elapsed:.0f}s ({len(result)} chars)")
            preview(result, f"Debater #{did}", max_lines=15)
            debate_parts.append((did, result))

    debate_parts.sort(key=lambda x: x[0])
    debate_transcript = "\n\n".join(
        f"--- Debater #{d} ---\n{text}" for d, text in debate_parts
    )
    log("debate", f"all debaters complete in {time.time() - start:.0f}s")

    # ── Fact-checker ──
    section(f"FACT-CHECK ROUND {round_num}")
    log("fact-check", f"dispatching fact-checker (debate transcript = {len(debate_transcript)} chars)...")
    fc_start = time.time()
    fc_report = run_fact_checker(round_num, debate_transcript, analyst_findings)
    log("fact-check", f"fact-checker DONE in {time.time() - fc_start:.0f}s ({len(fc_report)} chars)")
    preview(fc_report, "Fact-check report", max_lines=20)

    return debate_transcript, fc_report


def run_fact_checker(round_num: int, debate_transcript: str, analyst_findings: list[str]) -> str:
    findings_block = "\n".join(
        f"## Analyst #{i+1}\n{f}\n" for i, f in enumerate(analyst_findings)
    )

    prompt = textwrap.dedent(f"""\
    You are the **Fact-Checker** for debate round {round_num} of a terminal-bench-hard analysis.

    Your job: verify every factual claim in the debate against ground truth.
    You have ZERO tolerance for unverified claims.

    ## Debate transcript to fact-check
    {debate_transcript}

    ## Original analyst findings for reference
    {findings_block}

    ## Verification procedure
    For each factual claim (score numbers, code patterns, technique descriptions):
    1. Run `hive run list --json` and `hive run view <id> --json` to verify scores.
    2. Read actual source files to verify code claims:
       - /home/ubuntu/terminal-bench-hard/agent/agent.py
       - /home/ubuntu/terminal-bench-hard/agent/prompt-templates/
    3. Check job results in /home/ubuntu/terminal-bench-hard/jobs/ for metric claims.
    4. Read /home/ubuntu/terminal-bench-hard/tasks.md for task difficulty claims.
    5. Use `hive search` or `hive feed list` to verify community claims.

    ## Output format
    ```
    VERIFIED CLAIMS:
    - [OK] "<claim>" — confirmed by <source>

    REFUTED CLAIMS:
    - [WRONG] "<claim>" — actual: <truth>, source: <where you found it>

    UNVERIFIABLE CLAIMS:
    - [UNVERIFIABLE] "<claim>" — reason: <why you can't check>

    CORRECTIONS:
    - C1: <what the debaters should fix in the next round>

    DATA QUALITY SCORE: X/10
    (How well-supported are the overall debate conclusions?)
    ```

    Be ruthless. Check everything you can. The goal is to keep the debate honest.
    """)

    return Agent(
        description=f"R{round_num} Fact-checker",
        prompt=prompt,
        model="sonnet",
    )


# ─── Phase 4: Synthesis ─────────────────────────────────────────────────────

def synthesize(
    analyst_findings: list[str],
    debate_history: list[str],
    fact_check_reports: list[str],
) -> str:
    findings_block = "\n---\n".join(f"Analyst #{i+1}:\n{f}" for i, f in enumerate(analyst_findings))
    debate_block = "\n===\n".join(debate_history)
    fc_block = "\n---\n".join(f"Round {i+1}:\n{r}" for i, r in enumerate(fact_check_reports))

    prompt = textwrap.dedent(f"""\
    You are the **Synthesizer** — the final agent in a multi-agent analysis pipeline
    for terminal-bench-hard. Your job: produce the definitive summary.

    ## Analyst findings
    {findings_block}

    ## Debate transcripts (all rounds)
    {debate_block}

    ## Fact-checker reports (all rounds)
    {fc_block}

    ## Instructions
    Produce a final report with:

    1. **WHAT WORKS** — techniques with strong evidence (verified by fact-checker)
    2. **WHAT FAILS** — techniques that hurt performance (verified)
    3. **OPEN QUESTIONS** — promising but unverified hypotheses
    4. **RECOMMENDED NEXT EXPERIMENTS** — ranked by expected impact
    5. **SCORE TRAJECTORY** — how scores evolved and what drove changes
    6. **TASK-LEVEL INSIGHTS** — which of the 20 tasks are closest to flipping
       (0→1) and what might push them over

    Write for a developer who wants to improve the agent's score from 0.30 to 0.40+.
    Be concrete: name files, line numbers, specific code changes.
    Only include findings that survived fact-checking.
    """)

    section("PHASE 4: Final Synthesis (opus)")
    log("synthesis", f"dispatching synthesis agent (prompt = {len(prompt)} chars)...")
    start = time.time()
    result = Agent(
        description="Final synthesis",
        prompt=prompt,
        model="opus",
    )
    log("synthesis", f"synthesis DONE in {time.time() - start:.0f}s ({len(result)} chars)")
    return result


# ─── Main pipeline ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze terminal-bench-hard top runs")
    parser.add_argument("--top", type=int, default=5, help="Number of top runs to analyze")
    parser.add_argument("--rounds", type=int, default=2, help="Number of debate rounds")
    parser.add_argument("--analysts", type=int, default=2, help="Number of analyst agents")
    args = parser.parse_args()

    pipeline_start = time.time()

    banner("Terminal-Bench-Hard Run Analyzer")
    print(f"  Config: top={args.top} runs, {args.analysts} analysts, {args.rounds} debate rounds")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Pipeline: collect → analyze → debate×{args.rounds} → synthesize")

    # Phase 1: Collect data
    section("PHASE 1: Collecting data from hive")
    runs = fetch_top_runs(args.top)
    feed = fetch_feed_context()

    if not runs:
        log("FATAL", "no runs found on leaderboard — aborting")
        return

    log("collect", f"data collection complete — {len(runs)} runs + feed context")

    # Phase 2: Analyst deep-dives (parallel)
    analyst_findings = run_analysts(runs, feed, n_analysts=args.analysts)

    # Phase 3: Debate rounds (sequential, each builds on the last)
    debate_history = []
    fact_check_reports = []
    prev_debate = ""
    prev_fc = ""

    for round_num in range(1, args.rounds + 1):
        debate, fc = run_debate_round(round_num, analyst_findings, prev_debate, prev_fc)
        debate_history.append(debate)
        fact_check_reports.append(fc)
        prev_debate = debate
        prev_fc = fc

    # Phase 4: Final synthesis
    report = synthesize(analyst_findings, debate_history, fact_check_reports)

    # Output
    banner("FINAL REPORT", "█")
    print(report)

    # Save to file
    output_path = "/home/ubuntu/terminal-bench-hard/analysis_report.md"
    with open(output_path, "w") as f:
        f.write("# Terminal-Bench-Hard Run Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report)
        f.write("\n\n---\n\n## Raw Analyst Findings\n\n")
        for i, findings in enumerate(analyst_findings):
            f.write(f"### Analyst #{i+1}\n{findings}\n\n")
        f.write("## Debate Transcripts\n\n")
        for i, debate in enumerate(debate_history):
            f.write(f"### Round {i+1}\n{debate}\n\n")
        f.write("## Fact-Check Reports\n\n")
        for i, fc in enumerate(fact_check_reports):
            f.write(f"### Round {i+1}\n{fc}\n\n")

    elapsed = time.time() - pipeline_start
    banner("PIPELINE COMPLETE")
    print(f"  Total time:  {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Agents used: {args.analysts} analysts + {args.rounds * (args.analysts + 1)} debate/fc + 1 synthesis")
    print(f"  Report:      {output_path}")
    print()


if __name__ == "__main__":
    main()
