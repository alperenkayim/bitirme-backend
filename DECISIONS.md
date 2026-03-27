# Architecture Decisions

## ADR-001: Use `uv` instead of `pip`

**Decision:** Use `uv` as the package manager.
**Reason:** Project requirement. `uv` is significantly faster than `pip`, handles virtual environments automatically, and produces a lockfile (`uv.lock`) for reproducibility.
**How to apply:** Always use `uv add <package>` instead of `pip install`. Never run `pip` directly.

---

## ADR-002: Google Gemini (AI Studio) as LLM provider

**Decision:** Use `google_genai:gemini-2.0-flash` via Google AI Studio.
**Reason:** Free daily rate limits with no credit card required. Easy to set up with just `GOOGLE_API_KEY`. Switching to OpenRouter or other providers requires only changing one line in `agent.py`.
**How to apply:** Get API key from https://aistudio.google.com/app/apikey and put in `.env`.
**Alternative:** OpenRouter — set `OPENAI_API_KEY` + `OPENAI_BASE_URL=https://openrouter.ai/api/v1` and change model string.

---

## ADR-003: Deep Agent before ReAct Agent

**Decision:** Implement Deep Agent first (Week 2), ReAct Agent later.
**Reason:** Project assignment order. Deep Agent is more complex (hierarchical, sub-agents) so scaffolding it first establishes the infrastructure for both.

---

## ADR-004: LangGraph Studio as frontend for now

**Decision:** Use the hosted LangGraph Studio UI (`smith.langchain.com/studio`) instead of building a custom frontend.
**Reason:** Week 2 task says to use the LangChain-provided UI to move fast and prototype. Custom UI is a future-week task.

---

## ADR-005: Keep `.env` out of git

**Decision:** Never commit `.env` to the repository.
**Reason:** API keys would be exposed publicly. `.gitignore` excludes `.env`. Share keys only via secure channels.

---

## ADR-006: Single backend repo, separate frontend repo

**Decision:** `bitirme-backend` for Python agent code, `bitirme-frontend` for UI code.
**Reason:** Project requirement — instructor expects two separate repositories. Also good separation of concerns.
