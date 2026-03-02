# Mini-Castor

**The xv6 of Agent Operating Systems.**

Learn how to cage an LLM inside a deterministic OS kernel — in 425 lines of Python.

---

## What Is This?

In CS education, MIT's [xv6](https://pdos.csail.mit.edu/6.828/2023/xv6.html) teaches OS fundamentals in ~9000 lines of C. Students learn process scheduling, virtual memory, and syscalls without wading through millions of lines of Linux kernel code.

**Mini-Castor does the same thing for Agent OS design.**

LLM agents today run in "prompt + while loop" frameworks with no isolation, no budget limits, and no way to interrupt a hallucinating model mid-action. Real agent systems need the same controls that operating systems provide to user processes: syscall boundaries, preemptive scheduling, and capability-based security.

Mini-Castor strips away all production complexity and teaches four core concepts in a single Python file:

| Concept | OS Analogy | What You'll Learn |
|---|---|---|
| **Syscall Proxy** | System call interface | All agent actions go through a single validated gateway |
| **Checkpoint/Replay** | Process suspend/resume | Agent state is a replay log, not a serialized coroutine |
| **Capability Budgets** | Resource quotas | Finite, depletable budgets prevent runaway agents |
| **HITL Feedback** | Hardware interrupts | Destructive ops auto-suspend; humans approve/reject/modify |

No Pydantic. No SQLAlchemy. No external dependencies. Just `asyncio` and `dataclasses`.

---

## Quick Start

```bash
# Clone
git clone https://github.com/substrate-lab/mini-castor.git
cd mini-castor

# Run the interactive demo (no API keys needed)
python demo.py

# Run the tests
pip install pytest pytest-asyncio
pytest tests/ -v
```

### What the Demo Shows

```
═══════════════════════════════════════════════════════
  Phase 1: Running agent (will suspend at delete_emails)
═══════════════════════════════════════════════════════

    Agent received: Found 847 emails matching 'older than 30 days'
    Agent received: Analysis: 712 of these emails are older than 30 days

  Agent suspended! Here's the checkpoint:

  PID:    research_assistant-0
  Status: SUSPENDED
  Budget: {'io': 16}
  Log:    2 syscalls recorded
  PENDING HITL: delete_emails({'criteria': 'older than 30 days'})

═══════════════════════════════════════════════════════
  Phase 2: Human-in-the-Loop Decision
═══════════════════════════════════════════════════════

  The agent wants to: delete_emails(criteria='older than 30 days')

  Choose your response:
    [a] Approve  — execute the delete as requested
    [r] Reject   — block the action, agent re-plans
    [m] Modify   — approve intent, but change scope

  Your choice (a/r/m): _
```

Try all three options. Watch how the kernel replays cached syscalls and continues live execution after each one.

---

## How It Works

### The Kernel in One Diagram

```
Agent Function (untrusted)
    │
    │  await proxy.syscall("tool_name", {args})
    │
    v
┌──────────────────────────────────────────────┐
│              SyscallProxy                     │
│                                              │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐  │
│  │ REPLAY  │──>│  FAST    │──>│  SLOW    │  │
│  │  PATH   │   │  PATH    │   │  PATH    │  │
│  │         │   │          │   │          │  │
│  │ return  │   │ budget   │   │ suspend  │  │
│  │ cached  │   │ deduct   │   │ for HITL │  │
│  │ response│   │ execute  │   │ review   │  │
│  │         │   │ log      │   │          │  │
│  └─────────┘   └──────────┘   └──────────┘  │
│                                              │
│  Checkpoint: [syscall_log, budgets, status]  │
└──────────────────────────────────────────────┘
```

### The 6-Step Syscall Pipeline

Every `proxy.syscall()` follows this path:

```python
# Step 1: REPLAY — if we're re-running after suspend, serve cached response
if replaying:
    assert request matches log[replay_index]
    return cached response

# Step 2: TOOL LOOKUP — find the tool metadata
meta = registry[tool_name]

# Step 3: HITL GATE — destructive tools always suspend
if meta.destructive:
    set pending_hitl, raise SuspendInterrupt

# Step 4: BUDGET CHECK — deduct before execute (refund on failure)
if budget < cost:
    raise BudgetExhaustedError
budget -= cost

# Step 5: EXECUTE — run the actual tool
try:
    result = tool(**args)
except:
    budget += cost  # refund!
    raise

# Step 6: LOG — append to syscall_log for future replays
syscall_log.append(request, result)
return result
```

### Why Checkpoint/Replay?

Python coroutines can't be serialized. You can't `pickle` an `async def` that's halfway through execution — it holds C-level stack frames, event loop references, and closure state.

Castor's solution: **don't serialize the coroutine at all.** Instead:

1. Record every syscall and its response in a log
2. To "suspend": raise an exception that unwinds the entire call stack
3. To "resume": re-run the function from the top, serve cached responses from the log
4. The agent function fast-forwards through cached syscalls, then continues live

The agent doesn't know it's being replayed. From its perspective, `proxy.syscall()` just returned the same value.

### Why HITL Modify Doesn't Mutate the Request

This is the subtlest point in the design. When a human says "approve, but only delete files older than 7 days," why don't we just edit the pending request?

Because it would **break replay determinism:**

```
On replay, the agent function emits:  delete(scope="all")      ← original
But the log would contain:            delete(scope="7+ days")  ← mutated
                                      ^^^^^ MISMATCH → ReplayDivergenceError
```

Instead, we log the ORIGINAL request with `MODIFIED` status and the human's feedback. On replay, the agent sees the feedback and the LLM re-plans:

```
Syscall #1: delete(scope="all")     → MODIFIED("only 7+ days")  [cached]
Syscall #2: delete(scope="7+ days") → "deleted 42 files"        [new, live]
```

Full audit trail. Replay integrity preserved. The human writes natural language, not JSON.

---

## File Structure

```
mini-castor/
├── mini_castor.py           # The entire kernel (425 lines)
│     Part 1: Data models    #   Checkpoint, SyscallRecord, SuspendInterrupt
│     Part 2: Tool registry  #   @tool decorator, ToolMeta
│     Part 3: SyscallProxy   #   The replay gateway (the heart of the kernel)
│     Part 4: HITL handler   #   approve / reject / modify
│     Part 5: Kernel         #   Agent lifecycle management
│
├── demo.py                  # Interactive demo with mock tools (217 lines)
├── tests/
│   └── test_mini_castor.py  # 19 tests across 5 concept groups (507 lines)
└── pyproject.toml           # Zero runtime dependencies
```

**Total: 425 lines of kernel code.** Every line is commented. Read it top to bottom.

---

## Learning Path

Read the code in this order:

1. **`mini_castor.py` Part 1** — Data models. Understand `Checkpoint` and `SyscallRecord`. These are the "structs" of the OS.

2. **`mini_castor.py` Part 3** — SyscallProxy. This is the most important class. Read the 6-step pipeline carefully. Understand why replay comes first, why budget is deducted before execution, and why failure triggers a refund.

3. **`tests/test_mini_castor.py` TestReplay** — Run these tests and trace the execution. Watch how a pre-populated `syscall_log` causes the proxy to serve cached responses.

4. **`mini_castor.py` Part 4** — HITL handler. Understand the difference between approve, reject, and modify. Think about why modify DOESN'T mutate the request.

5. **`tests/test_mini_castor.py` TestFullRoundTrip** — The three round-trip tests show the complete suspend → HITL → replay cycle for each feedback mode.

6. **`demo.py`** — Run it. Try all three HITL options. Watch the audit trail.

---

## From Mini-Castor to Production

Mini-Castor teaches the concepts. [Project Castor](https://github.com/substrate-lab/castor) is the production implementation with:

| Feature | Mini-Castor | Project Castor |
|---|---|---|
| Validation | Basic tool lookup | Pydantic V2 auto-schema + error feedback |
| Budgets | Simple deduct/refund | Atomic delegation, parent-child reclaim |
| HITL | approve/reject/modify | + child HITL propagation across sub-agents |
| Context management | None | Lodge (FIFO eviction, pinning, search_memory) |
| Sub-agent spawning | None | sync + async fan-out/fan-in with budget delegation |
| Persistence | None (in-memory) | SQLAlchemy + SQLite |
| CLI | None | `castor list\|show\|reject\|modify` |
| LLM integration | Mock only | LLMSyscall wrapper for any async LLM client |
| Tests | 19 | 170 |
| Lines | 425 | ~1,500 |

---

## License

Apache 2.0 — See [LICENSE](LICENSE).
