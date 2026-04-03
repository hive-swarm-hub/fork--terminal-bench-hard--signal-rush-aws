#!/usr/bin/env python3
"""
Eval Monitor — watches the latest eval run, triggers failure analysis when done.

Usage:
    python monitor_eval.py              # monitor latest run, poll every 60s
    python monitor_eval.py --poll 30    # poll every 30s
    python monitor_eval.py --job <dir>  # monitor specific job
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime


def ts():
    return datetime.now().strftime("%H:%M:%S")


def get_latest_job():
    jobs = "/home/ubuntu/terminal-bench-hard/jobs"
    dirs = [d for d in sorted(os.listdir(jobs)) if os.path.isdir(os.path.join(jobs, d)) and d[0].isdigit()]
    return os.path.join(jobs, dirs[-1])


def read_status(job_dir):
    """Read current eval status from result.json."""
    result_path = os.path.join(job_dir, "result.json")
    if not os.path.exists(result_path):
        return None

    with open(result_path) as f:
        d = json.load(f)

    finished = d.get("finished_at")
    n_total = d.get("n_total_trials", 0)
    stats = d.get("stats", {})
    n_trials = stats.get("n_trials", 0)
    n_errors = stats.get("n_errors", 0)

    eval_data = None
    evals = stats.get("evals", {})
    if evals:
        eval_data = list(evals.values())[0]

    score = eval_data["metrics"][0]["mean"] if eval_data else 0
    passed = eval_data.get("reward_stats", {}).get("reward", {}).get("1.0", []) if eval_data else []
    failed = eval_data.get("reward_stats", {}).get("reward", {}).get("0.0", []) if eval_data else []
    exceptions = {}
    if eval_data:
        for exc, tids in eval_data.get("exception_stats", {}).items():
            for tid in tids:
                exceptions[tid.split("__")[0]] = exc

    # Count task dirs to see total progress
    task_dirs = [d for d in os.listdir(job_dir) if "__" in d and os.path.isdir(os.path.join(job_dir, d))]

    return {
        "finished": finished,
        "n_total": n_total,
        "n_trials": n_trials,
        "n_errors": n_errors,
        "n_task_dirs": len(task_dirs),
        "score": score,
        "n_passed": len(passed),
        "n_failed": len(failed),
        "passed": [t.split("__")[0] for t in passed],
        "failed": [t.split("__")[0] for t in failed],
        "exceptions": exceptions,
    }


def print_status(s, job_dir):
    """Pretty-print the current status."""
    bar_total = 40
    done = s["n_trials"]
    total = max(s["n_total"], s["n_task_dirs"], done)
    bar_filled = int(bar_total * done / total) if total > 0 else 0

    bar = "█" * bar_filled + "░" * (bar_total - bar_filled)

    print(f"\r[{ts()}] [{bar}] {done}/{total}  "
          f"score={s['score']:.3f}  "
          f"pass={s['n_passed']}  fail={s['n_failed']}  err={s['n_errors']}  "
          f"{'DONE' if s['finished'] else 'running...'}",
          end="", flush=True)


def run_failure_analysis(job_dir):
    """Import and run the failure analyzer."""
    print(f"\n\n[{ts()}] Eval complete! Launching failure analysis...")
    print(f"{'=' * 70}")

    # Import the analyzer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_failures import get_failed_tasks, analyze_one_task, review_analyses, synthesize

    import concurrent.futures

    start = time.time()
    tasks, score, n_trials, n_errors = get_failed_tasks(job_dir)

    if not tasks:
        print("No failed tasks — nothing to analyze!")
        return

    print(f"\n  Score: {score:.3f} ({n_trials} trials, {n_errors} errors)")
    print(f"  Failed: {len(tasks)} tasks")
    for t in tasks:
        exc = f" [{t['exception']}]" if t['exception'] else ""
        print(f"    {t['name']}{exc}")

    # Phase 1: parallel analysis
    print(f"\n[{ts()}] Dispatching {len(tasks)} parallel analyst agents...")
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
                print(f"  [{ts()}] {name} done — {r.num_turns} turns, ${r.total_cost_usd or 0:.3f}")
                analyses.append((name, r))
            except Exception as e:
                print(f"  [{ts()}] {tname} FAILED: {e}")

    analyses.sort(key=lambda x: x[0])

    # Phase 2: reviewer
    print(f"\n[{ts()}] Dispatching reviewer (opus)...")
    review = review_analyses(analyses, job_dir, score, n_trials, n_errors)
    print(f"  [{ts()}] Reviewer done — {review.num_turns} turns, ${review.total_cost_usd or 0:.3f}")

    # Phase 3: synthesize
    print(f"\n[{ts()}] Synthesizing...")
    report = synthesize(analyses, review, score, n_trials, n_errors)
    print(f"  [{ts()}] Done.")

    # Print report
    elapsed = time.time() - start
    total_cost = sum(r.total_cost_usd or 0 for _, r in analyses) + (review.total_cost_usd or 0) + (report.total_cost_usd or 0)

    print(f"\n{'█' * 70}")
    print(f"  FAILURE ANALYSIS REPORT")
    print(f"{'█' * 70}\n")
    print(report.result)
    print(f"\n{'─' * 70}")
    print(f"  REVIEWER FINDINGS")
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
            f.write(f"### {name}\n{r.result}\n\n")

    print(f"\n{'=' * 70}")
    print(f"  Analysis: {elapsed:.0f}s ({elapsed/60:.1f}min), ${total_cost:.3f}")
    print(f"  Report: {output_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", help="Job directory to monitor")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()

    job_dir = args.job or get_latest_job()
    print(f"Monitoring: {job_dir}")
    print(f"Poll interval: {args.poll}s")
    print()

    prev_trials = -1

    while True:
        s = read_status(job_dir)
        if s is None:
            print(f"[{ts()}] Waiting for result.json...", flush=True)
            time.sleep(args.poll)
            continue

        print_status(s, job_dir)

        # Print new passes/fails when they appear
        if s["n_trials"] != prev_trials and prev_trials >= 0:
            print()  # newline after progress bar
            if s["n_passed"] > 0:
                print(f"  Passed: {s['passed'][-3:]}")  # last 3
            new_fails = [f for f in s["failed"] if s["exceptions"].get(f)]
            if new_fails:
                print(f"  Timeouts: {new_fails[-3:]}")

        prev_trials = s["n_trials"]

        if s["finished"]:
            print()  # newline after progress bar
            print(f"\n[{ts()}] Eval FINISHED at {s['finished']}")
            print(f"  Final score: {s['score']:.3f} ({s['n_passed']} passed, {s['n_failed']} failed, {s['n_errors']} errors)")
            print(f"  Passed: {s['passed']}")
            print(f"  Failed: {s['failed']}")

            if s["n_failed"] > 0:
                run_failure_analysis(job_dir)
            else:
                print("\nAll tasks passed! Nothing to analyze.")

            break

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
