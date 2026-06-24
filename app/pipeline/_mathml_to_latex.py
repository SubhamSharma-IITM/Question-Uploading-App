#!/usr/bin/env python3
"""
mathml_to_latex.py
------------------

Utility helpers to replace MathML blocks inside HTML with LaTeX strings.
Only inline MathML (<math>...</math>) snippets are converted; all other
HTML is left untouched (except that the <math> element itself is replaced
with '\\( ... \\)' containing a LaTeX representation).
"""

from __future__ import annotations

from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup

MATH_NS = "http://www.w3.org/1998/Math/MathML"

GREEK_MAP = {
    "\u03b1": "\\alpha",
    "\u03b2": "\\beta",
    "\u03b3": "\\gamma",
    "\u03b4": "\\delta",
    "\u03b5": "\\varepsilon",
    "\u03b6": "\\zeta",
    "\u03b7": "\\eta",
    "\u03b8": "\\theta",
    "\u03b9": "\\iota",
    "\u03ba": "\\kappa",
    "\u03bb": "\\lambda",
    "\u03bc": "\\mu",
    "\u03bd": "\\nu",
    "\u03be": "\\xi",
    "\u03bf": "o",
    "\u03c0": "\\pi",
    "\u03c1": "\\rho",
    "\u03c3": "\\sigma",
    "\u03c4": "\\tau",
    "\u03c5": "\\upsilon",
    "\u03c6": "\\varphi",
    "\u03c7": "\\chi",
    "\u03c8": "\\psi",
    "\u03c9": "\\omega",
    "\u0391": "A",
    "\u0392": "B",
    "\u0393": "\\Gamma",
    "\u0394": "\\Delta",
    "\u0395": "E",
    "\u0396": "Z",
    "\u0397": "H",
    "\u0398": "\\Theta",
    "\u0399": "I",
    "\u039a": "K",
    "\u039b": "\\Lambda",
    "\u039c": "M",
    "\u039d": "N",
    "\u039e": "\\Xi",
    "\u039f": "O",
    "\u03a0": "\\Pi",
    "\u03a1": "P",
    "\u03a3": "\\Sigma",
    "\u03a4": "T",
    "\u03a5": "\\Upsilon",
    "\u03a6": "\\Phi",
    "\u03a7": "X",
    "\u03a8": "\\Psi",
    "\u03a9": "\\Omega",
}

MO_MAP = {
    "\u2212": "-",  # minus sign
    "-": "-",
    "+": "+",
    "\u00b1": "\\pm",
    "\u2213": "\\mp",
    "\u00d7": "\\times",
    "\u22c5": "\\cdot",
    "\u22a0": "\\boxtimes",
    "/": "/",
    "=": "=",
    "<": "<",
    ">": ">",
    "\u2264": "\\leq",
    "\u2265": "\\geq",
    "\u2260": "\\neq",
    "\u2227": "\\land",
    "\u2228": "\\lor",
    "\u2229": "\\cap",
    "\u222a": "\\cup",
    "\u2208": "\\in",
    "\u2209": "\\notin",
    "\u2286": "\\subseteq",
    "\u2287": "\\supseteq",
    ",": ",",
    ";": ";",
    ":": ":",
    "\u2219": "\\cdot",
    "\u00b7": "\\cdot",
}


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _escape_text(text: str | None) -> str:
    if not text:
        return ""
    replacements = {
        "\\": "\\\\",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for src, target in replacements.items():
        text = text.replace(src, target)
    return text


def _convert_text_token(token: str | None) -> str:
    if not token:
        return ""
    token = token.strip()
    if not token:
        return ""
    if token in GREEK_MAP:
        return GREEK_MAP[token]
    return token


def _join_parts(parts) -> str:
    return " ".join(filter(None, parts))


def _convert_children(el) -> str:
    parts = []
    if el.text:
        parts.append(_convert_text_token(el.text))
    for child in el:
        parts.append(_convert_element(child))
        if child.tail:
            parts.append(_convert_text_token(child.tail))
    return _join_parts(parts)


def _convert_element(el) -> str:
    tag = _strip_ns(el.tag)

    if tag in {"math", "mrow"}:
        return _convert_children(el)

    if tag == "mi":
        return _convert_text_token(el.text)

    if tag == "mn":
        return _escape_text(el.text)

    if tag == "mo":
        text = (el.text or "").strip()
        return MO_MAP.get(text, _escape_text(text))

    if tag == "mtext":
        return _escape_text(el.text)

    if tag == "msup":
        children = list(el)
        if len(children) >= 2:
            base = _convert_element(children[0])
            sup = _convert_element(children[1])
            return f"{base}^{{{sup}}}"
        return _convert_children(el)

    if tag == "msub":
        children = list(el)
        if len(children) >= 2:
            base = _convert_element(children[0])
            sub = _convert_element(children[1])
            return f"{base}_{{{sub}}}"
        return _convert_children(el)

    if tag == "msubsup":
        children = list(el)
        if len(children) >= 3:
            base = _convert_element(children[0])
            sub = _convert_element(children[1])
            sup = _convert_element(children[2])
            return f"{base}_{{{sub}}}^{{{sup}}}"
        return _convert_children(el)

    if tag in {"msqrt", "mroot"}:
        children = list(el)
        if tag == "msqrt" and children:
            radicand = _convert_children(el)
            return f"\\sqrt{{{radicand}}}"
        if tag == "mroot" and len(children) == 2:
            base = _convert_element(children[0])
            degree = _convert_element(children[1])
            return f"\\sqrt[{degree}]{{{base}}}"
        return _convert_children(el)

    if tag == "mfrac":
        children = list(el)
        if len(children) >= 2:
            num = _convert_element(children[0])
            den = _convert_element(children[1])
            return f"\\frac{{{num}}}{{{den}}}"
        return _convert_children(el)

    if tag == "mfenced":
        open_ch = el.attrib.get("open", "(")
        close_ch = el.attrib.get("close", ")")
        separators = el.attrib.get("separators", ",")
        inner = []
        for idx, child in enumerate(el):
            inner.append(_convert_element(child))
            if idx < len(el) - 1 and separators:
                inner.append(separators.strip() or ",")
        return f"{open_ch}{' '.join(filter(None, inner))}{close_ch}"

    if tag == "mtable":
        rows = []
        for row in el.findall(f"{{{MATH_NS}}}mtr") + el.findall("mtr"):
            cells = []
            for cell in row.findall(f"{{{MATH_NS}}}mtd") + row.findall("mtd"):
                cells.append(_convert_children(cell))
            rows.append(" & ".join(filter(None, cells)))
        matrix_body = " \\\\ ".join(filter(None, rows))
        return f"\\begin{{matrix}} {matrix_body} \\end{{matrix}}"

    if tag == "mtr":
        cells = [_convert_children(child) for child in el]
        return " & ".join(filter(None, cells))

    if tag == "mtd":
        return _convert_children(el)

    if tag == "menclose":
        notation = el.attrib.get("notation", "")
        content = _convert_children(el)
        if "circle" in notation:
            return f"\\boxed{{{content}}}"
        return content

    if tag == "munderover":
        children = list(el)
        if len(children) == 3:
            base = _convert_element(children[0])
            under = _convert_element(children[1])
            over = _convert_element(children[2])
            return f"\\overset{{{over}}}{{\\underset{{{under}}}{{{base}}}}}"
        return _convert_children(el)

    if tag == "munder":
        children = list(el)
        if len(children) == 2:
            base = _convert_element(children[0])
            under = _convert_element(children[1])
            return f"\\underset{{{under}}}{{{base}}}"
        return _convert_children(el)

    if tag == "mover":
        children = list(el)
        if len(children) == 2:
            base = _convert_element(children[0])
            over = _convert_element(children[1])
            return f"\\overset{{{over}}}{{{base}}}"
        return _convert_children(el)

    if tag == "mpadded":
        return _convert_children(el)

    if tag == "semantics":
        for child in el:
            return _convert_element(child)
        return ""

    return _convert_children(el)


def _convert_math_fragment(fragment: str) -> str:
    try:
        element = ET.fromstring(fragment)
    except ET.ParseError:
        return fragment
    latex = _convert_element(element).strip()
    if not latex:
        return fragment
    return f"\\({latex}\\)"


def convert_mathml_to_latex(html: str) -> str:
    if "<math" not in html.lower():
        return html
    fixed_html = _ensure_closed_math_tags(html)
    soup = BeautifulSoup(fixed_html, "html.parser")
    changed = False
    for math_tag in soup.find_all("math"):
        fragment = str(math_tag)
        converted = _convert_math_fragment(fragment)
        if converted != fragment:
            math_tag.replace_with(soup.new_string(converted))
            changed = True
    if not changed:
        return fixed_html
    return str(soup)


def _ensure_closed_math_tags(html: str) -> str:
    lower = html.lower()
    result = []
    idx = 0
    while True:
        start = lower.find("<math", idx)
        if start == -1:
            result.append(html[idx:])
            break
        result.append(html[idx:start])
        fragment, next_idx = _extract_math_fragment(html, start)
        result.append(fragment)
        idx = next_idx
    return "".join(result)


def _extract_math_fragment(html: str, start: int) -> tuple[str, int]:
    length = len(html)
    # find end of opening tag
    open_end = html.find(">", start)
    if open_end == -1:
        return html[start:] + "</math>", length
    open_end += 1
    depth = 1
    i = open_end
    while i < length:
        if html.startswith("</math", i):
            close_end = html.find(">", i)
            if close_end == -1:
                return html[start:] + "</math>", length
            depth -= 1
            i = close_end + 1
            if depth == 0:
                return html[start:i], i
            continue
        if html.startswith("<math", i):
            depth += 1
            next_gt = html.find(">", i)
            if next_gt == -1:
                return html[start:] + "</math>", length
            i = next_gt + 1
            continue
        if html[i] == "<":
            next_gt = html.find(">", i)
            if next_gt == -1:
                return html[start:] + "</math>", length
            tag_name = html[i + 1 : next_gt].split()[0].lower().lstrip("/")
            if not tag_name.startswith("m"):
                # encountered a non-MathML tag; close before it
                return html[start:i] + "</math>", i
            i = next_gt + 1
            continue
        i += 1
    # ran to end without closing tag
    return html[start:] + "</math>", length


__all__ = ["convert_mathml_to_latex"]
