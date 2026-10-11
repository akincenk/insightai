"""Thin wrapper around the OpenAI Python SDK (v1 client)."""

import os
from functools import lru_cache

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini")
EMBED_BATCH_SIZE = 100

INSTRUCTIONS = (
    "Answer the user's question using only the provided document excerpts. "
    "If the excerpts do not contain the answer, say you could not find it in the documents."
)


class MissingAPIKeyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise MissingAPIKeyError("OPENAI_API_KEY is not set")
    return OpenAI()  # reads OPENAI_API_KEY from the environment


def embed(texts: list[str]) -> np.ndarray:
    """Return one float32 embedding row per input text."""
    client = get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i : i + EMBED_BATCH_SIZE])
        vectors.extend(d.embedding for d in resp.data)
    return np.asarray(vectors, dtype=np.float32)


def answer(question: str, contexts: list[str]) -> str:
    excerpts = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    resp = get_client().responses.create(
        model=CHAT_MODEL,
        instructions=INSTRUCTIONS,
        input=f"Document excerpts:\n{excerpts}\n\nQuestion: {question}",
    )
    return resp.output_text
