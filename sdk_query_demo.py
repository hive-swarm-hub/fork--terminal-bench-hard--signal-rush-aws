#!/usr/bin/env python3
"""
Demo: Agent() vs AgentRun() — final string vs full structured result.
"""

from claude_code_orchestrate import Agent, AgentRun

PROMPT = (
    "Read /home/ubuntu/terminal-bench-hard/tasks.md and tell me which 3 tasks "
    "have the highest baseline pass rate. Then read the first 20 lines of "
    "/home/ubuntu/terminal-bench-hard/agent/agent.py and tell me what class it defines. "
    "Be brief."
)


def demo_agent():
    """Old way: Agent() returns a plain string."""
    print("=" * 70)
    print("  Agent() — returns str")
    print("=" * 70)

    result = Agent(description="Read tasks", prompt=PROMPT, model="haiku")

    print(f"type:   {type(result).__name__}")
    print(f"length: {len(result)} chars")
    print(f"value:\n{result}")


def demo_agent_run():
    """New way: AgentRun() returns AgentResult with full process."""
    print("\n" + "=" * 70)
    print("  AgentRun() — returns AgentResult")
    print("=" * 70)

    r = AgentRun(
        description="Read tasks",
        prompt=PROMPT,
        model="haiku",
        on_tool_call=lambda tc: print(f"    [live] tool: {tc.name}({list(tc.input.keys())})"),
        on_text=lambda t: print(f"    [live] text: {t[:80]}...") if len(t) > 80 else print(f"    [live] text: {t}"),
    )

    # ── Final result (same as Agent()) ──
    print(f"\n--- Final result ---")
    print(f"str(r):  {str(r)[:200]}...")
    print(f"r.result: {r.result[:200]}...")

    # ── Metadata ──
    print(f"\n--- Metadata ---")
    print(f"turns:       {r.num_turns}")
    print(f"duration:    {r.duration_ms}ms (api: {r.duration_api_ms}ms)")
    print(f"cost:        ${r.total_cost_usd}")
    print(f"session:     {r.session_id}")
    print(f"is_error:    {r.is_error}")

    # ── Convenience properties ──
    print(f"\n--- Convenience ---")
    print(f"tool_names:       {r.tool_names}")
    print(f"total_tool_calls: {r.total_tool_calls}")
    print(f"text_turns:       {len(r.text_turns)} responses")

    # ── Turn-by-turn breakdown ──
    print(f"\n--- Turn-by-turn ({len(r.turns)} turns) ---")
    for turn in r.turns:
        print(f"\n  Turn #{turn.index} (model={turn.model})")
        if turn.thinking:
            print(f"    thinking: {turn.thinking[:100]}...")
        if turn.text:
            print(f"    text:     {turn.text[:120]}...")
        for tc in turn.tool_calls:
            args_preview = str(tc.input)[:100]
            print(f"    tool:     {tc.name}({args_preview})")
        for tr in turn.tool_results:
            content_preview = (tr.content or "")[:100]
            print(f"    result:   [{tr.tool_use_id[:20]}...] {content_preview}...")

    # ── Raw messages (for power users) ──
    print(f"\n--- Raw messages ---")
    print(f"  {len(r.raw_messages)} messages captured")
    for i, msg in enumerate(r.raw_messages):
        print(f"  [{i}] {type(msg).__name__}")

    # ── repr ──
    print(f"\n--- repr ---")
    print(f"  {repr(r)}")


if __name__ == "__main__":
    print("Running Agent() (old way)...")
    demo_agent()
    print("\n\nRunning AgentRun() (new way)...")
    demo_agent_run()
