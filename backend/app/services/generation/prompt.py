"""
System/user prompt construction. Kept as plain string templates (not an
LLM-framework prompt object) so the exact text sent to the model is easy to
read, diff, and reason about in a review.
"""

from app.services.generation.context_selector import ContextBundle

SYSTEM_PROMPT = """You are a document question-answering assistant. Answer ONLY using \
the numbered context passages provided below. Follow these rules strictly:

1. Do not use any knowledge that is not present in the provided context.
2. Every factual sentence in your answer must be immediately followed by the \
marker(s) of the context passage(s) that support it, like [1] or [2][4].
3. If the context does not contain enough information to answer the question, \
respond with exactly: "INSUFFICIENT_EVIDENCE: " followed by a one-sentence \
explanation of what is missing. Do not guess or fill gaps with outside knowledge.
4. Do not cite a passage that does not actually support the sentence it is \
attached to.
5. Be concise. Do not repeat the question or add commentary outside the answer.
"""


def build_user_prompt(question: str, context: ContextBundle) -> str:
    if not context.chunks:
        return (
            f"Question: {question}\n\n"
            "No context passages were retrieved. Respond with INSUFFICIENT_EVIDENCE "
            "as instructed."
        )
    return f"Context passages:\n\n{context.prompt_text}\n\nQuestion: {question}\n\nAnswer:"


def build_regeneration_prompt(question: str, context: ContextBundle, failed_claim_texts: list[str]) -> str:
    """
    Used by the self-correction loop (rag/pipeline.py) when the previous
    answer contained claims that failed NLI verification. Takes plain claim
    text (not the `Claim` dataclass) so generation/ has no dependency on
    verification/ — verification already depends on generation/ (for
    ContextChunk), and this keeps that a one-way edge.
    """
    failed_text = "\n".join(f'- "{t}"' for t in failed_claim_texts)
    return (
        f"Context passages:\n\n{context.prompt_text}\n\n"
        f"Question: {question}\n\n"
        "Your previous answer included claims that were NOT verifiably supported by the "
        "context passages:\n"
        f"{failed_text}\n\n"
        "Write a new answer that avoids making these unsupported claims. Only include "
        "statements that are directly supported by the context passages above, each "
        "immediately followed by its citation marker. If you cannot answer without making "
        "unsupported claims, respond with INSUFFICIENT_EVIDENCE as instructed.\n\nAnswer:"
    )
