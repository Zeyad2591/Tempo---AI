"""Setup check: which parts of Tempo are reachable on this machine, then one test request.

python -m scripts.check
"""

from __future__ import annotations

import asyncio

from tempo.config import load_settings
from tempo.core.factory import build_engine
from tempo.core.schemas import Request


async def main() -> None:
    s = load_settings()
    specs = {m.name: m for m in s.models}
    keys = "OpenRouter key" if s.openrouter_key else f"{len(s.gemini_keys)} Gemini key(s)"
    print(
        f"Cloud: {s.cloud_provider} ({keys}) | light: {specs['light'].model_id} "
        f"| large: {specs['large'].model_id}"
    )
    ollama = f"{s.ollama_host} {s.ollama_model}" if s.ollama_model else "off (API-only)"
    print("Ollama:", ollama, "| embeddings:", s.embed_provider)
    print(
        "Tavily search:",
        "yes" if s.tavily_key else "no",
        "| Tiger:",
        "yes" if s.database_url else "no",
    )
    engine = await build_engine(s)
    print("Status:", engine.status())
    for text in [
        "What is the capital of Australia?",
        "How many days do customers have to request a refund for a Halden robot arm?",
        "Who was elected Pope in May 2025?",
    ]:
        r = await engine.ask(Request(text=text))
        print(f"\nQ: {text}\nA: {r.answer[:200]}\n   {r.latency_ms} ms, ${r.cost:.6f}")
        for line in r.receipt:
            print("   -", line)


if __name__ == "__main__":
    asyncio.run(main())
