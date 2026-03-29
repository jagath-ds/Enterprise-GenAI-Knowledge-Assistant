
from __future__ import annotations
import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import tiktoken 


# ─── Config ──────────────────────────────────────────────────────────────────

CHUNK_SIZE_TOKENS   = 400   # max tokens per chunk  (fits well in 4096-token prompts)
CHUNK_OVERLAP_TOKENS = 80   # overlap between consecutive chunks
MIN_CHUNK_TOKENS    = 80    # discard chunks shorter than this (noise)
ENCODING_MODEL      = "cl100k_base"   # same encoding used by most modern LLMs


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single retrievable unit with full provenance."""
    chunk_id:    str            # sha256 of content (dedup key)
    text:        str            # the actual text to embed
    doc_id:      str            # source document identifier
    source_path: str            # file path / URL
    page_number: Optional[int]  # page if available
    chunk_index: int            # position within the document
    total_chunks: int           # filled in after all chunks are generated
    token_count: int            # pre-computed for budget tracking
    metadata:    dict = field(default_factory=dict)  # any extra fields

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Tokenizer singleton ─────────────────────────────────────────────────────

_enc = None

def get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding(ENCODING_MODEL)
    return _enc


def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text))


# ─── Sentence splitter ───────────────────────────────────────────────────────

_SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences. Handles common abbreviations to reduce
    false-positive splits (e.g. 'Mr. Smith' or 'e.g. this').
    """
    # Protect common abbreviations
    protected = re.sub(
        r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|Fig|Vol|No)\.\s',
        lambda m: m.group(0).replace('. ', '.<PROTECT>'),
        text
    )
    raw_sentences = _SENTENCE_ENDINGS.split(protected)
    # Restore protected periods
    return [s.replace('<PROTECT>', ' ').strip() for s in raw_sentences if s.strip()]


# ─── Core chunker ────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Token-aware, sentence-boundary-respecting chunker with sliding overlap.

    Usage:
        chunker = SemanticChunker()
        chunks  = chunker.chunk_document(text, doc_id="report_q3", source_path="docs/q3.pdf")
    """

    def __init__(
        self,
        chunk_size:    int = CHUNK_SIZE_TOKENS,
        overlap:       int = CHUNK_OVERLAP_TOKENS,
        min_chunk:     int = MIN_CHUNK_TOKENS,
    ):
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.min_chunk  = min_chunk

    # ── Public API ────────────────────────────────────────────────────────

    def chunk_document(
        self,
        text:        str,
        doc_id:      str,
        source_path: str,
        page_number: Optional[int] = None,
        extra_meta:  dict          = None,
    ) -> List[TextChunk]:
        """
        Chunk a single document/page text into overlapping token-bounded chunks.
        Returns a list of TextChunk objects ready for embedding.
        """
        text = self._clean(text)
        if not text:
            return []

        sentences     = split_into_sentences(text)
        raw_chunks    = self._build_chunks(sentences)
        chunks        = []

        for idx, chunk_text in enumerate(raw_chunks):
            token_count = count_tokens(chunk_text)
            if token_count < self.min_chunk:
                continue  # skip noise

            chunk_id = self._make_id(doc_id, chunk_text)
            chunks.append(TextChunk(
                chunk_id    = chunk_id,
                text        = chunk_text,
                doc_id      = doc_id,
                source_path = source_path,
                page_number = page_number,
                chunk_index = idx,
                total_chunks= 0,      # filled below
                token_count = token_count,
                metadata    = extra_meta or {},
            ))

        # Back-fill total_chunks
        n = len(chunks)
        for c in chunks:
            c.total_chunks = n

        return chunks

    def chunk_pages(
        self,
        pages:       List[dict],   # [{"text": ..., "page": 1}, ...]
        doc_id:      str,
        source_path: str,
        extra_meta:  dict = None,
    ) -> List[TextChunk]:
        """Chunk a multi-page document, preserving page numbers."""
        all_chunks: List[TextChunk] = []
        running_idx = 0
        for page in pages:
            page_chunks = self.chunk_document(
                text        = page["text"],
                doc_id      = doc_id,
                source_path = source_path,
                page_number = page.get("page"),
                extra_meta  = extra_meta,
            )
            for c in page_chunks:
                c.chunk_index = running_idx
                running_idx  += 1
            all_chunks.extend(page_chunks)

        # Rewrite total_chunks with final count
        n = len(all_chunks)
        for c in all_chunks:
            c.total_chunks = n

        return all_chunks

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_chunks(self, sentences: List[str]) -> List[str]:
        """
        Greedy sentence packing with token overlap.
        Fills a buffer until adding the next sentence would exceed chunk_size,
        then emits the chunk and starts a new one seeded with the overlap tail.
        """
        chunks: List[str] = []
        buffer: List[str] = []
        buf_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens(sent)

            # If a single sentence exceeds chunk_size, hard-split it
            if sent_tokens > self.chunk_size:
                # Flush current buffer first
                if buffer:
                    chunks.append(" ".join(buffer))
                    buffer, buf_tokens = [], 0
                # Hard split the long sentence
                for sub in self._hard_split(sent):
                    chunks.append(sub)
                continue

            if buf_tokens + sent_tokens > self.chunk_size and buffer:
                # Emit current chunk
                chunks.append(" ".join(buffer))
                # Seed overlap: walk back from end until overlap quota filled
                buffer, buf_tokens = self._build_overlap(buffer)

            buffer.append(sent)
            buf_tokens += sent_tokens

        if buffer:
            chunks.append(" ".join(buffer))

        return chunks

    def _build_overlap(self, buffer: List[str]) -> tuple[List[str], int]:
        """Return trailing sentences that together fill ~overlap tokens."""
        overlap_buf: List[str] = []
        overlap_tokens = 0
        for sent in reversed(buffer):
            t = count_tokens(sent)
            if overlap_tokens + t > self.overlap:
                break
            overlap_buf.insert(0, sent)
            overlap_tokens += t
        return overlap_buf, overlap_tokens

    def _hard_split(self, text: str) -> List[str]:
        """Token-level split for sentences that exceed chunk_size."""
        enc    = get_encoder()
        tokens = enc.encode(text)
        parts  = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            part_tokens = tokens[i: i + self.chunk_size]
            parts.append(enc.decode(part_tokens))
        return parts

    @staticmethod
    def _clean(text: str) -> str:
        """Normalize whitespace and remove non-printable characters."""
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def _make_id(doc_id: str, text: str) -> str:
        h = hashlib.sha256(f"{doc_id}::{text}".encode()).hexdigest()
        return f"{doc_id}_{h[:12]}"
