"""Document loading and chunking for RAG pipeline."""

import os
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document

from .utils import get_logger

logger = get_logger(__name__)


class CorpusLoader:
    """Load and chunk documentation corpus."""

    def __init__(
        self,
        corpus_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        use_parent_child_chunking: bool = True,
        parent_max_tokens: int = 900,
    ):
        """
        Initialize corpus loader.

        Args:
            corpus_path: Path to corpus directory (RST files)
            chunk_size: Child chunk size in tokens (converted to chars: token * 4)
            chunk_overlap: Overlap between child chunks in tokens
            use_parent_child_chunking: If True, index child chunks but keep parent sections for context
            parent_max_tokens: Maximum parent chunk size in tokens before splitting parent sections
        """
        self.corpus_path = corpus_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_parent_child_chunking = use_parent_child_chunking
        self.parent_max_tokens = parent_max_tokens

        # Convert token sizes to character sizes
        self.char_chunk_size = chunk_size * 4
        self.char_overlap = chunk_overlap * 4
        self.parent_char_chunk_size = parent_max_tokens * 4

        self.chunks = []
        self.chunk_metadata = {}

    @staticmethod
    def _is_rst_heading(heading_line: str, underline_line: str) -> bool:
        """Detect a reStructuredText heading pattern."""
        title = heading_line.strip()
        underline = underline_line.strip()

        if not title or not underline:
            return False
        if len(underline) < len(title):
            return False
        if len(set(underline)) != 1:
            return False

        return underline[0] in {"=", "-", "~", "^", '"', "'", "*", "+", "#", "<", ">"}

    @staticmethod
    def _rst_heading_level(underline_line: str) -> int:
        """Return a stable approximate level for an RST underline heading."""
        order = ["=", "-", "~", "^", '"', "'", "*", "+", "#", "<", ">"]
        char = underline_line.strip()[0]
        try:
            return order.index(char) + 1
        except ValueError:
            return 99

    def _find_sections(self, lines: List[str]) -> List[Dict]:
        """
        Return section ranges with heading hierarchy metadata.

        Each section dict contains:
            start_idx: zero-based inclusive start line index
            end_idx: zero-based inclusive end line index
            title: section title
            level: heading level
            parent_title: nearest parent section title, if any
            section_path: breadcrumb-like section path
        """
        headings = []
        i = 0

        while i < len(lines) - 1:
            if self._is_rst_heading(lines[i], lines[i + 1]):
                headings.append({
                    "idx": i,
                    "title": lines[i].strip(),
                    "level": self._rst_heading_level(lines[i + 1]),
                })
                i += 2
                continue
            i += 1

        if not headings:
            return [{
                "start_idx": 0,
                "end_idx": len(lines) - 1,
                "title": "document",
                "level": 1,
                "parent_title": None,
                "section_path": "document",
            }]

        sections: List[Dict] = []
        stack: List[Dict] = []

        # Optional preamble before first heading
        if headings[0]["idx"] > 0:
            sections.append({
                "start_idx": 0,
                "end_idx": headings[0]["idx"] - 1,
                "title": "document",
                "level": 0,
                "parent_title": None,
                "section_path": "document",
            })

        for h_idx, heading in enumerate(headings):
            start_idx = heading["idx"]
            end_idx = headings[h_idx + 1]["idx"] - 1 if h_idx + 1 < len(headings) else len(lines) - 1
            title = heading["title"]
            level = heading["level"]

            while stack and stack[-1]["level"] >= level:
                stack.pop()

            parent_title = stack[-1]["title"] if stack else None
            section_path = " > ".join([s["title"] for s in stack] + [title])

            sections.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "title": title,
                "level": level,
                "parent_title": parent_title,
                "section_path": section_path,
            })

            stack.append({"title": title, "level": level})

        return sections

    def _line_offsets(self, content: str) -> List[int]:
        """Return the starting character offset for each line."""
        offsets = [0]
        for index, char in enumerate(content):
            if char == "\n":
                offsets.append(index + 1)
        return offsets

    def _make_chunk_document(
        self,
        filename: str,
        source: str,
        content: str,
        lines: List[str],
        start_idx: int,
        end_idx: int,
        section_title: str,
        section_path: Optional[str] = None,
        parent_title: Optional[str] = None,
        chunk_level: str = "child",
        parent_id: Optional[str] = None,
        parent_index: Optional[int] = None,
        child_index: Optional[int] = None,
    ) -> Document:
        """Create a chunk document with accurate line and character metadata."""
        chunk_lines = lines[start_idx:end_idx + 1]
        chunk_text = "\n".join(chunk_lines).strip("\n")

        line_offsets = self._line_offsets(content)
        start_line = start_idx + 1
        end_line = end_idx + 1
        start_char = line_offsets[start_idx] if start_idx < len(line_offsets) else 0
        end_char = line_offsets[end_idx + 1] - 1 if end_idx + 1 < len(line_offsets) else len(content)

        metadata = {
            "source": source,
            "filename": filename,
            "section_title": section_title,
            "section_path": section_path or section_title,
            "parent_title": parent_title,
            "chunk_level": chunk_level,
            "parent_id": parent_id,
            "parent_index": parent_index,
            "child_index": child_index,
            "start_char": start_char,
            "end_char": end_char,
            "start_line": start_line,
            "end_line": end_line,
        }
        return Document(page_content=chunk_text, metadata=metadata)

    def _split_line_range_by_chars(
        self,
        lines: List[str],
        start_idx: int,
        end_idx: int,
        max_chars: int,
        overlap_chars: int = 0,
    ) -> List[Tuple[int, int]]:
        """
        Split a zero-based inclusive line range into zero-based inclusive chunks.
        """
        ranges: List[Tuple[int, int]] = []
        current_start = start_idx

        while current_start <= end_idx:
            current_chars = 0
            current_end = current_start

            while current_end <= end_idx:
                line = lines[current_end]
                added = len(line) + (1 if current_end > current_start else 0)
                if current_chars + added > max_chars and current_end > current_start:
                    break
                current_chars += added
                current_end += 1

            actual_end = max(current_start, current_end - 1)
            ranges.append((current_start, actual_end))

            if actual_end >= end_idx:
                break

            if overlap_chars > 0:
                overlap_start = actual_end + 1
                overlap_total = 0
                while overlap_start > current_start and overlap_total < overlap_chars:
                    overlap_start -= 1
                    overlap_total += len(lines[overlap_start]) + 1
                current_start = max(overlap_start, current_start + 1)
            else:
                current_start = actual_end + 1

        return ranges

    def _chunk_section_flat(
        self,
        filename: str,
        source: str,
        content: str,
        lines: List[str],
        section: Dict,
        next_doc_index: int,
    ) -> Tuple[List[Document], int]:
        """Original-style section-aware chunking, with parent metadata set to self."""
        docs: List[Document] = []

        child_ranges = self._split_line_range_by_chars(
            lines=lines,
            start_idx=section["start_idx"],
            end_idx=section["end_idx"],
            max_chars=self.char_chunk_size,
            overlap_chars=self.char_overlap,
        )

        for child_start, child_end in child_ranges:
            doc_index = next_doc_index
            parent_id = f"{filename}:flat:{doc_index}"

            doc = self._make_chunk_document(
                filename=filename,
                source=source,
                content=content,
                lines=lines,
                start_idx=child_start,
                end_idx=child_end,
                section_title=section["title"],
                section_path=section.get("section_path"),
                parent_title=section.get("parent_title"),
                chunk_level="child",
                parent_id=parent_id,
                parent_index=doc_index,
                child_index=doc_index,
            )
            docs.append(doc)
            next_doc_index += 1

        return docs, next_doc_index

    def _chunk_section_parent_child(
        self,
        filename: str,
        source: str,
        content: str,
        lines: List[str],
        section: Dict,
        next_doc_index: int,
    ) -> Tuple[List[Document], int]:
        """
        Create parent chunks and child chunks for a section.

        Parent chunks are larger context units.
        Child chunks are smaller retrieval units.
        """
        docs: List[Document] = []

        parent_ranges = self._split_line_range_by_chars(
            lines=lines,
            start_idx=section["start_idx"],
            end_idx=section["end_idx"],
            max_chars=self.parent_char_chunk_size,
            overlap_chars=0,
        )

        for parent_num, (parent_start, parent_end) in enumerate(parent_ranges):
            parent_id = (
                f"{filename}:"
                f"{section['start_idx'] + 1}-"
                f"{section['end_idx'] + 1}:"
                f"parent-{parent_num}"
            )

            parent_index = next_doc_index

            parent_doc = self._make_chunk_document(
                filename=filename,
                source=source,
                content=content,
                lines=lines,
                start_idx=parent_start,
                end_idx=parent_end,
                section_title=section["title"],
                section_path=section.get("section_path"),
                parent_title=section.get("parent_title"),
                chunk_level="parent",
                parent_id=parent_id,
                parent_index=parent_index,
                child_index=None,
            )

            docs.append(parent_doc)
            next_doc_index += 1

            child_ranges = self._split_line_range_by_chars(
                lines=lines,
                start_idx=parent_start,
                end_idx=parent_end,
                max_chars=self.char_chunk_size,
                overlap_chars=self.char_overlap,
            )

            for child_start, child_end in child_ranges:
                child_index = next_doc_index

                child_doc = self._make_chunk_document(
                    filename=filename,
                    source=source,
                    content=content,
                    lines=lines,
                    start_idx=child_start,
                    end_idx=child_end,
                    section_title=section["title"],
                    section_path=section.get("section_path"),
                    parent_title=section.get("parent_title"),
                    chunk_level="child",
                    parent_id=parent_id,
                    parent_index=parent_index,
                    child_index=child_index,
                )

                docs.append(child_doc)
                next_doc_index += 1

        return docs, next_doc_index

    def load_and_chunk(self) -> List[Document]:
        """
        Load all RST files from corpus and chunk them.

        Returns:
            List of Document objects with chunks
        """
        logger.info(f"Loading corpus from {self.corpus_path}")

        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus path not found: {self.corpus_path}")

        chunks: List[Document] = []
        next_doc_index = 0

        rst_files = sorted([f for f in os.listdir(self.corpus_path) if f.endswith(".rst")])

        logger.info(f"Found {len(rst_files)} RST files")

        for filename in rst_files:
            filepath = os.path.join(self.corpus_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                lines = content.splitlines()
                if not lines:
                    continue

                sections = self._find_sections(lines)
                logger.info(f"{filename}: found {len(sections)} sections")

                for section in sections:
                    if self.use_parent_child_chunking:
                        section_docs, next_doc_index = self._chunk_section_parent_child(
                            filename=filename,
                            source=filepath,
                            content=content,
                            lines=lines,
                            section=section,
                            next_doc_index=next_doc_index,
                        )
                    else:
                        section_docs, next_doc_index = self._chunk_section_flat(
                            filename=filename,
                            source=filepath,
                            content=content,
                            lines=lines,
                            section=section,
                            next_doc_index=next_doc_index,
                        )

                    chunks.extend(section_docs)

            except Exception as e:
                logger.warning(f"Could not load {filename}: {e}")

        logger.info(f"Created {len(chunks)} chunks")

        parent_count = sum(1 for c in chunks if c.metadata.get("chunk_level") == "parent")
        child_count = sum(1 for c in chunks if c.metadata.get("chunk_level") == "child")
        logger.info(f"Chunk levels: parent={parent_count}, child={child_count}")

        self.chunks = chunks
        return chunks

    def get_chunk_stats(self) -> Dict[str, any]:
        """Get statistics about chunks."""
        if not self.chunks:
            return {}

        sizes = [len(c.page_content) for c in self.chunks]
        parent_count = sum(1 for c in self.chunks if c.metadata.get("chunk_level") == "parent")
        child_count = sum(1 for c in self.chunks if c.metadata.get("chunk_level") == "child")

        return {
            "num_chunks": len(self.chunks),
            "num_parent_chunks": parent_count,
            "num_child_chunks": child_count,
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_size": sum(sizes) // len(sizes),
            "total_chars": sum(sizes),
        }


def load_corpus(
    corpus_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    use_parent_child_chunking: bool = True,
    parent_max_tokens: int = 900,
) -> List[Document]:
    """
    Convenience function to load and chunk corpus.

    Args:
        corpus_path: Path to corpus directory
        chunk_size: Child chunk size in tokens
        chunk_overlap: Child chunk overlap in tokens
        use_parent_child_chunking: Whether to create parent and child chunks
        parent_max_tokens: Maximum parent chunk size in tokens

    Returns:
        List of Document chunks
    """
    loader = CorpusLoader(
        corpus_path,
        chunk_size,
        chunk_overlap,
        use_parent_child_chunking=use_parent_child_chunking,
        parent_max_tokens=parent_max_tokens,
    )
    return loader.load_and_chunk()