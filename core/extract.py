"""PDF text extraction and question splitting.

Two responsibilities:
1. Pull raw text out of an uploaded PDF.
2. Split that text into individual questions so the LLM can tag each one.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pdfplumber

# Question markers seen in real university papers:
#   "1."   "Q1."   "Q.1"   "Q 1"   "1)"   "(a)"  "Question 1:"
# A marker only starts a new question when it is at the start of a line and
# is followed by text, so numbered lists inside a question body are not split.
_QUESTION_MARKER = re.compile(
    r"""^\s*
        (?:
            (?:question|q)\s*\.?\s*(?P<qnum>\d{1,3})   # Question 4 / Q.4 / Q4
          | (?P<plain>\d{1,3})                          # 4.  4)  (4)
        )
        \s*[\.\):\-]?\s+
        (?=\S)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "(a)" / "(i)" / "a)" style sub-parts inside one question.
_SUBPART_MARKER = re.compile(r"^\s*[\(\[]?[a-zivx]{1,4}[\)\]]?[\.\)]?\s+(?=\S)", re.IGNORECASE)

# Marks declared by the paper, e.g. "[5 Marks]", "(10 marks)", or a
# trailing "5 M". Requiring brackets or a trailing position avoids mistaking
# measurements such as "a 5 m long wire" for a marks value.
_MARKS = re.compile(
    r"(?:\[\s*(\d{1,2})\s*marks?\s*\]"
    r"|\(\s*(\d{1,2})\s*marks?\s*\)"
    r"|(\d{1,2})\s*marks?\s*$"
    r"|(\d{1,2})\s*m\s*$)",
    re.IGNORECASE,
)

# Boilerplate that is never a question.
_NOISE = re.compile(
    r"^\s*(?:page\s+\d+|turn\s+over|continue\s+on\s+the\s+back|"
    r"subject\s*code|model\s+paper|answer\s+all\s+the\s+questions)\s*$",
    re.IGNORECASE,
)


@dataclass
class Question:
    """One question pulled out of a question paper."""

    number: int
    text: str
    marks: int | None = None
    source_file: str = ""
    page: int | None = None
    labels: dict = field(default_factory=dict)

    @property
    def preview(self) -> str:
        """First line of the question, trimmed for display in a table."""
        first = self.text.strip().splitlines()[0] if self.text.strip() else ""
        return first[:120] + ("..." if len(first) > 120 else "")


@dataclass
class ExtractionResult:
    """Everything extracted from one uploaded PDF."""

    filename: str
    pages: int
    full_text: str
    questions: list[Question]
    error: str | None = None


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> ExtractionResult:
    """Extract full text from PDF bytes. Never raises; errors are returned."""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = len(pdf.pages)
            page_texts = [(page.extract_text() or "") for page in pdf.pages]
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        return ExtractionResult(filename, 0, "", [], f"Could not read PDF: {exc}")

    full_text = "\n".join(page_texts).strip()
    questions = split_into_questions(full_text, filename)
    return ExtractionResult(filename, pages, full_text, questions)


def _clean_body(body: str) -> tuple[str, int | None]:
    """Strip noise from a question body and pull out its marks value."""
    marks = None
    match = _MARKS.search(body)
    if match:
        marks = int(next(value for value in match.groups() if value is not None))
        body = _MARKS.sub(" ", body)

    # Drop trailing section headers such as "OR" left behind by a split.
    lines = [ln.rstrip() for ln in body.splitlines()]
    while lines and (not lines[-1].strip() or _NOISE.match(lines[-1])):
        lines.pop()
    return "\n".join(lines).strip(), marks


def split_into_questions(text: str, filename: str = "") -> list[Question]:
    """Split raw paper text into a list of numbered questions."""
    if not text.strip():
        return []

    lines = text.splitlines()
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    current_number: int | None = None

    for line in lines:
        if _NOISE.match(line):
            continue

        marker = _QUESTION_MARKER.match(line)
        is_new_question = False
        number = None

        if marker:
            number = int(marker.group("qnum") or marker.group("plain"))
            # A marker that continues the running sequence starts a new question.
            # Anything else (out-of-order numbers, sub-parts) is body text.
            if current_number is None or number == current_number + 1 or number <= current_number:
                is_new_question = True
            else:
                is_new_question = False

        if is_new_question:
            if current:
                blocks.append((current_number, current))
            current = [line]
            current_number = number
        else:
            if current_number is None:
                # Text before the first question marker (paper header).
                current = []
            else:
                current.append(line)

    if current:
        blocks.append((current_number, current))

    questions: list[Question] = []
    for number, block in blocks:
        body = "\n".join(block)
        # Remove the marker from the very first line of the body.
        first, _, rest = body.partition("\n")
        first = _QUESTION_MARKER.sub("", first, count=1)
        body = "\n".join([first, rest]).strip()

        cleaned, marks = _clean_body(body)
        if len(cleaned) < 12:
            continue

        questions.append(
            Question(
                number=number if number is not None else len(questions) + 1,
                text=cleaned,
                marks=marks,
                source_file=filename,
            )
        )

    # Fall back to sub-part splitting when the paper never used Q markers.
    if len(questions) < 3:
        questions = _split_by_subparts(text, filename)

    return questions


def _split_by_subparts(text: str, filename: str) -> list[Question]:
    """Second-chance splitter for papers numbered with (a), (b), (c)."""
    chunks: list[list[str]] = []
    current: list[str] = []

    for line in text.splitlines():
        if _NOISE.match(line):
            continue
        if _SUBPART_MARKER.match(line) and current:
            chunks.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append(current)

    questions: list[Question] = []
    for index, chunk in enumerate(chunks, start=1):
        body = "\n".join(chunk).strip()
        if len(body) < 25:
            continue
        cleaned, marks = _clean_body(body)
        if len(cleaned) < 12:
            continue
        questions.append(
            Question(number=index, text=cleaned, marks=marks, source_file=filename)
        )
    return questions
