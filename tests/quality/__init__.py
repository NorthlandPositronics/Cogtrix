"""Cogtrix quality harness — Phase 1 (deterministic, mock-LLM, no API cost).

Measures behavioural quality metrics across scenarios of varying complexity:
  - Tool call accuracy      (did the agent call the right tools?)
  - Phantom call rate       (did hallucinated tool markup appear?)
  - Task completion         (did the agent produce a coherent final response?)
  - Context pair integrity  (no orphaned ToolMessage/AIMessage pairs)
  - Turn efficiency         (completed within budget?)
"""
