"""Small structural repairs for OCR/LLM markup before it enters metadata.

The pipeline receives OCR text that often mixes Markdown, LaTeX, and small
HTML fragments. A single missing ``$$`` or ``\\end{aligned}`` can make every
downstream renderer treat the rest of the field as math. These helpers only
repair clearly structural balance problems; they do not rewrite content.
"""

from html.parser import HTMLParser
import re
from typing import Dict, List, Optional, Tuple


_LATEX_ENV_RE = re.compile(r"\\(begin|end)\{([^}]+)\}")
_DISPLAY_DOLLAR_RE = re.compile(r"(?<!\\)\$\$")
_SINGLE_DOLLAR_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)")
_BRACKET_OPEN_RE = re.compile(r"(?<!\\)\\\[")
_BRACKET_CLOSE_RE = re.compile(r"(?<!\\)\\\]")
_PAREN_OPEN_RE = re.compile(r"(?<!\\)\\\(")
_PAREN_CLOSE_RE = re.compile(r"(?<!\\)\\\)")

_LATEX_COMPLETE_PATTERNS = [
    re.compile(r"\$\$[^$]+?\$\$", re.DOTALL),
    re.compile(r"(?<!\$)\$(?!\$)[^$]+?(?<!\$)\$(?!\$)", re.DOTALL),
    re.compile(r"\\\(.+?\\\)", re.DOTALL),
    re.compile(r"\\\[.+?\\\]", re.DOTALL),
    re.compile(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.DOTALL),
]

_BOUNDARY_RE = re.compile(
    r"""^\s*(
        (?:Q\.?\s*)?\d{1,3}[\.\)]\s+
        | Sol(?:ution)?\.?\s*\d{0,3}\b
        | \(\d{1,3}\)\s*Answer\b
        | Answer\s*:
        | \\section\*\{(?:Q\.?|Sol|Solution|Answers?)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_VOID_HTML_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}
_TRACKED_HTML_TAGS = {
    "b", "strong", "i", "em", "u", "sub", "sup", "span", "p",
    "div", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tbody",
    "thead", "tr", "td", "th", "ul", "ol", "li", "math", "mrow",
    "msup", "msub", "msubsup", "mfrac", "msqrt", "mroot", "mtable",
    "mtr", "mtd", "mstyle",
}


class _HTMLBalanceParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in _TRACKED_HTML_TAGS and tag not in _VOID_HTML_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs):
        return

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag not in _TRACKED_HTML_TAGS:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
            return
        if tag in self.stack:
            while self.stack:
                popped = self.stack.pop()
                if popped == tag:
                    break


def latex_balance_issues(text: str) -> Dict[str, List[str]]:
    """Return missing LaTeX closers by category."""
    issues = {"envs": [], "delimiters": []}
    if not text:
        return issues

    env_stack: List[str] = []
    for match in _LATEX_ENV_RE.finditer(text):
        kind, env = match.group(1), match.group(2).strip()
        if not env:
            continue
        if kind == "begin":
            env_stack.append(env)
        elif env_stack and env_stack[-1] == env:
            env_stack.pop()
        elif env in env_stack:
            while env_stack:
                popped = env_stack.pop()
                if popped == env:
                    break

    issues["envs"] = list(reversed(env_stack))

    if len(_DISPLAY_DOLLAR_RE.findall(text)) % 2:
        issues["delimiters"].append("$$")
    if len(_BRACKET_OPEN_RE.findall(text)) > len(_BRACKET_CLOSE_RE.findall(text)):
        issues["delimiters"].append(r"\]")
    if len(_PAREN_OPEN_RE.findall(text)) > len(_PAREN_CLOSE_RE.findall(text)):
        issues["delimiters"].append(r"\)")
    if len(_SINGLE_DOLLAR_RE.findall(text)) % 2:
        issues["delimiters"].append("$")

    return issues


def latex_is_balanced(text: str) -> bool:
    issues = latex_balance_issues(text)
    return not issues["envs"] and not issues["delimiters"]


def _missing_latex_closers(text: str) -> List[str]:
    issues = latex_balance_issues(text)
    closers = [rf"\end{{{env}}}" for env in issues["envs"]]
    closers.extend(issues["delimiters"])
    return closers


def repair_latex_markup(text: str) -> str:
    """Append missing LaTeX closing tags/delimiters when clearly unbalanced."""
    if not text:
        return text or ""
    closers = _missing_latex_closers(text)
    if not closers:
        return text
    suffix = "".join(closers)
    trailing_html = re.search(r"(\s*(?:</[A-Za-z][\w:-]*>\s*)+)$", text)
    if trailing_html:
        insert_at = trailing_html.start(1)
        return text[:insert_at] + suffix + text[insert_at:]
    if text.endswith("\n"):
        return text + suffix
    return text + "\n" + suffix


def close_unclosed_html_tags(html: str) -> str:
    """Close common unclosed HTML tags to prevent visual leakage."""
    if not html or "<" not in html:
        return html or ""
    parser = _HTMLBalanceParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return html
    if not parser.stack:
        return html
    return html + "".join(f"</{tag}>" for tag in reversed(parser.stack))


def repair_text_markup(text: str) -> str:
    """Repair LaTeX delimiter/environment balance and common HTML tag balance."""
    if not text:
        return text or ""
    repaired = repair_latex_markup(text)
    repaired = close_unclosed_html_tags(repaired)
    return repaired


def count_latex_segments(text: str) -> int:
    """Count complete LaTeX blocks plus one marker for unbalanced raw LaTeX."""
    if not text:
        return 0
    count = sum(len(pattern.findall(text)) for pattern in _LATEX_COMPLETE_PATTERNS)
    if not latex_is_balanced(text):
        count += 1
    return count


def _line_has_needed_closer(line: str, closers: List[str]) -> bool:
    if not line or not closers:
        return False
    for closer in closers:
        if closer == "$$" and "$$" in line:
            return True
        if closer == "$" and "$" in line.replace("$$", ""):
            return True
        if closer in line:
            return True
    return False


def slice_with_markup_repair(
    raw_lines: List[str],
    start: Optional[int],
    end: Optional[int],
    *,
    max_extra_lines: int = 12,
) -> Tuple[Optional[str], Optional[int], bool]:
    """Slice raw lines, extending briefly if the slice cuts off markup closers.

    Returns ``(content, effective_end, repaired_or_extended)``.
    """
    if start is None or end is None:
        return None, None, False
    if not raw_lines:
        return None, None, False

    start = max(0, int(start))
    end = min(int(end), len(raw_lines) - 1)
    if start > end:
        return None, None, False

    effective_end = end
    content = "".join(raw_lines[start:effective_end + 1]).strip()
    changed = False

    for _ in range(max_extra_lines):
        closers = _missing_latex_closers(content)
        if not closers:
            break
        next_idx = effective_end + 1
        if next_idx >= len(raw_lines):
            break
        next_line = raw_lines[next_idx]
        if _BOUNDARY_RE.match(next_line) and not _line_has_needed_closer(next_line, closers):
            break
        effective_end = next_idx
        content = "".join(raw_lines[start:effective_end + 1]).strip()
        changed = True

    repaired = repair_text_markup(content)
    if repaired != content:
        changed = True
        content = repaired

    return content, effective_end, changed
