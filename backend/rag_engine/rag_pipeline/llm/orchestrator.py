from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from rag_engine.rag_pipeline.retrieval.retrieval_engine import RetrievedContext

logger = logging.getLogger(__name__)


# ─── Deep-trigger keywords ────────────────────────────────────────────────────

DEEP_TRIGGERS: set[str] = {
    "explain", "describe", "compare", "analyze", "analyse",
    "summarise", "summarize", "why", "how does", "how do",
    "what is the difference", "elaborate", "detail",
    "break down", "walk me through", "illustrate",
    "what are the steps", "in depth", "overview",
    "pros and cons", "advantages", "disadvantages",
}


# ─── Prompt templates ─────────────────────────────────────────────────────────

FAST_PROMPT = """\
You are an Enterprise Knowledge Assistant.
Answer using ONLY the information provided in the context below.

RULES:
- Respond with 3 to 5 bullet points ONLY.
- Each bullet must be one clear, direct sentence under 20 words.
- Do NOT write introductions, summaries, or follow-up sentences.
- Do NOT explain your reasoning or add commentary.
- Do NOT add any information not present in the context.
- Do NOT mention the documents or context explicitly.

If the answer is not found in the context, respond EXACTLY with:
"I could not find an answer to this question in the provided documents."\
"""

DEEP_PROMPT = """\
You are an Enterprise Knowledge Assistant.
Answer using ONLY the information provided in the context below.

RULES:
- Always complete your answer fully. Never stop mid-sentence or mid-point.
- Every section, point, or explanation you begin MUST be finished completely.
- Begin with a 1-2 sentence summary, then expand with full structured detail.
- Use numbered lists or headers where helpful for clarity.
- Combine related ideas into a cohesive, flowing explanation.
- Explain relationships between concepts and steps clearly.
- Do NOT add any information not present in the context.
- Do NOT mention the documents or context explicitly.

If the answer is not found in the context, respond EXACTLY with:
"I could not find an answer to this question in the provided documents."\
"""

FALLBACK_RESPONSE = (
    "I could not find an answer to this question in the provided documents."
)

USER_PROMPT_TEMPLATE = """\
CONTEXT FROM DOCUMENTS:
{context}

---

QUESTION: {question}

INSTRUCTIONS:
- Answer strictly using the context above.
- If the context is insufficient, say exactly: "{fallback}"
- Do not invent, infer, or add outside knowledge.

Answer:\
"""


# ─── Response dataclass ───────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    question:     str
    answer:       str
    sources:      List[dict]
    is_grounded:  bool
    model_used:   str
    context_used: str = ""
    raw_output:   str = ""


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class LLMOrchestrator:
    """
    Wraps the Ollama / HuggingFace LLM call with:
      · Smart fast / deep model selection (single call, no double-override)
      · Metadata-clean context (strips score annotations before prompt)
      · Hard token split: 300 (fast) vs 1200 (deep)
      · Strict grounding prompt with injected fallback string
      · Fallback detection
      · Source citation passthrough
    """

    def __init__(
        self,
        fast_model:  str   = "phi3:mini",
        deep_model:  str   = "mistral:latest",
        backend:     str   = "ollama",
        base_url:    str   = "http://localhost:11434",
        max_tokens:  int   = 800,
        temperature: float = 0.1,
    ):
        self.fast_model  = fast_model
        self.deep_model  = deep_model
        self.model       = fast_model       # active model; set by select_model()
        self.backend     = backend
        self.base_url    = base_url
        self.max_tokens  = max_tokens       # kept for reference; not used in _get_model_config
        self.temperature = temperature      # kept for reference; not used in _get_model_config
        self._client     = None
        self._pipe       = None

        if backend == "ollama":
            self._setup_ollama()
        elif backend == "huggingface":
            self._setup_hf()
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. Use 'ollama' or 'huggingface'."
            )

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup_ollama(self) -> None:
        try:
            from ollama import Client
            self._client = Client(host=self.base_url)
            logger.info(
                f"Ollama client ready → {self.base_url} | "
                f"fast={self.fast_model} | deep={self.deep_model}"
            )
        except ImportError:
            raise ImportError("Run: pip install ollama")

    def _setup_hf(self) -> None:
        try:
            from transformers import pipeline
            import torch
            logger.info(f"Loading HuggingFace model: {self.model} …")
            self._pipe = pipeline(
                "text-generation",
                model          = self.model,
                device_map     = "auto",
                torch_dtype    = torch.float16,
                max_new_tokens = self.max_tokens,
            )
            logger.info("HuggingFace pipeline ready.")
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

    # ── Model selection ───────────────────────────────────────────────────

    def select_model(self, query: str, mode: str = "auto") -> None:
        """
        Set self.model based on explicit mode or keyword auto-detection.
        Called ONCE from pipeline.query() — never internally from answer().
        """
        if mode == "fast":
            self.model = self.fast_model

        elif mode == "deep":
            self.model = self.deep_model

        else:  # auto
            q = query.lower()
            is_complex = (
                len(query) > 120
                or q.count("?") > 1
                or any(kw in q for kw in DEEP_TRIGGERS)
            )
            self.model = self.deep_model if is_complex else self.fast_model

        logger.info(f"Model selected: {self.model!r}  (mode={mode!r})")

    # ── Per-model config ──────────────────────────────────────────────────

    def _get_model_config(self) -> dict:
        """
        Hard-coded token budgets to guarantee visible output difference.
        Fast: 300 tokens  → forces bullet brevity
        Deep: 1200 tokens → allows full structured explanation
        """
        if self.model == self.fast_model:
            return {
                "system_prompt": FAST_PROMPT,
                "temperature":   0.1,
                "max_tokens":    300,
            }
        else:
            return {
                "system_prompt": DEEP_PROMPT,
                "temperature":   0.25,
                "max_tokens":    1200,
            }

    # ── Metadata cleaner ──────────────────────────────────────────────────

    @staticmethod
    def _clean_context(raw_context: str) -> str:
        """
        Strip retrieval metadata (score, page score annotations) that leak
        from the vector store into the prompt and appear in model output.

        Handles:
          - Standalone lines:  "(score: 0.63)"  /  "page score: 0.61"
          - Inline suffixes:   "...some text (score: 0.63)"
        """
        cleaned_lines = []
        for line in raw_context.splitlines():
            # Drop lines that are purely a score annotation
            if re.match(r"^\s*\(?\s*(score|page score)\s*:", line, re.IGNORECASE):
                continue
            # Strip inline score/page-score annotations
            line = re.sub(r"\s*\(\s*score\s*:\s*[\d.]+\)", "", line, flags=re.IGNORECASE)
            line = re.sub(r"\s*\(\s*page score\s*:\s*[\d.]+\)", "", line, flags=re.IGNORECASE)
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    # ── Main entry point ──────────────────────────────────────────────────

    def answer(self, context: RetrievedContext) -> LLMResponse:
        """
        Given a RetrievedContext, build prompt → call LLM → return LLMResponse.

        NOTE: select_model() must be called by the pipeline BEFORE this method.
              answer() intentionally does NOT call select_model() to avoid
              overriding the caller's explicit mode choice.
        """
        # Short-circuit: no context → return fallback without an LLM call
        if context.is_empty():
            logger.warning("Empty context — returning fallback without LLM call.")
            return LLMResponse(
                question    = context.query,
                answer      = FALLBACK_RESPONSE,
                sources     = [],
                is_grounded = False,
                model_used  = self.model,
            )

        # Clean metadata before building prompt
        clean_ctx = self._clean_context(context.context_text)

        prompt = USER_PROMPT_TEMPLATE.format(
            context  = clean_ctx,
            question = context.query,
            fallback = FALLBACK_RESPONSE,
        )

        raw_output  = self._call_llm(prompt)
        is_grounded = FALLBACK_RESPONSE.lower() not in raw_output.lower()

        return LLMResponse(
            question    = context.query,
            answer      = raw_output.strip(),
            sources     = context.sources,
            is_grounded = is_grounded,
            model_used  = self.model,
            context_used= clean_ctx,
            raw_output  = raw_output,
        )

    # ── LLM backends ──────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str) -> str:
        if self.backend == "ollama":
            return self._call_ollama(user_prompt)
        if self.backend == "huggingface":
            return self._call_hf(user_prompt)
        raise ValueError(f"Unknown backend: {self.backend!r}")

    def _call_ollama(self, user_prompt: str) -> str:
        if self._client is None:
            raise RuntimeError("Ollama client not initialized.")

        cfg = self._get_model_config()
        response = self._client.chat(
            model    = self.model,
            messages = [
                {"role": "system", "content": cfg["system_prompt"]},
                {"role": "user",   "content": user_prompt},
            ],
            options  = {
                "temperature": cfg["temperature"],
                "num_predict": cfg["max_tokens"],
                "stop":        [],
                "num_ctx":     4096,
            },
        )
        return response["message"]["content"]

    def _call_hf(self, user_prompt: str) -> str:
        if self._pipe is None:
            raise RuntimeError("HuggingFace pipeline not initialized.")

        cfg         = self._get_model_config()
        full_prompt = (
            f"[INST] <<SYS>>\n{cfg['system_prompt']}\n<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
        output = self._pipe(full_prompt, return_full_text=False)
        return output[0]["generated_text"]