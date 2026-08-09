"""PDF → 文本 加载器（pymupdf 提取 + 扫描页检测 + 清理管线）。

许可说明：PyMuPDF 为 AGPL-3.0 双授权。本项目用于学习/开源演示；若未来
闭源商用，需购买 PyMuPDF 商业许可或改用 MIT 许可的 pdfplumber。
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

SCANNED_PAGE_MIN_CHARS = 10


class PdfOcrUnavailableError(ValueError):
    """扫描页需要 OCR 但 OCR 引擎不可用（消息含 PDF_OCR_UNAVAILABLE）。"""


def _get_ocr_engine():
    """懒加载 RapidOCR：文本型 PDF 不触发；导入失败抛 PdfOcrUnavailableError。"""
    try:
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()
    except Exception as exc:  # pragma: no cover - 依赖缺失路径
        raise PdfOcrUnavailableError(f"PDF_OCR_UNAVAILABLE: {exc}") from exc


def _sort_ocr_items(result: list) -> list:
    """OCR 结果按 y 坐标（再按 x）排序，保证阅读顺序。"""
    return sorted(result, key=lambda item: (item[0][0][1], item[0][0][0]))


def _ocr_page(page: "pymupdf.Page", engine) -> str:
    """单页 OCR：渲染为图片传给引擎，按 y 坐标排序组装文本。"""
    pix = page.get_pixmap(dpi=200)
    result, _elapse = engine(pix.tobytes())
    if not result:
        return ""
    items = _sort_ocr_items(result)
    return "\n".join(item[1].strip() for item in items if item[1].strip())


def _page_text(page: "pymupdf.Page") -> str:
    """提取单页文本层；多栏时按 blocks 阅读顺序拼接。"""
    blocks = page.get_text("blocks")
    blocks = [b for b in blocks if b[4].strip()]
    blocks.sort(key=lambda b: (round(b[1], 1), b[0]))
    return "\n".join(b[4].strip() for b in blocks).strip()


def is_scanned_page(text: str) -> bool:
    """字符密度检测：去除空白后字符数 < 阈值 → 判定无文本层（扫描页）。"""
    return len(re.sub(r"\s+", "", text)) < SCANNED_PAGE_MIN_CHARS


def extract_pdf_pages(path: str | Path) -> list[str]:
    document_path = Path(path)
    if not document_path.is_file():
        raise FileNotFoundError(document_path)
    try:
        doc = pymupdf.open(document_path)
    except Exception as exc:  # 损坏/加密/非 PDF
        raise ValueError(f"failed to open pdf: {document_path}: {exc}") from exc
    try:
        pages: list[str] = []
        ocr_engine = None  # 懒加载：只有出现扫描页才初始化
        for page in doc:
            text = _page_text(page)
            if is_scanned_page(text):
                if ocr_engine is None:
                    ocr_engine = _get_ocr_engine()
                text = _ocr_page(page, ocr_engine)
            pages.append(text)
        return pages
    finally:
        doc.close()


# ---- 页级文本清理管线（借鉴 MIT 项目 M1ck4/pdfmd 的 transform 算法）----

_PAGE_NUMBER_PATTERNS = (
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^第\s*\d+\s*页$"),
    re.compile(r"^page\s*\d+$", re.IGNORECASE),
    re.compile(r"^[-–]\s*\d+\s*[-–]$"),
)


def _is_page_number_line(line: str) -> bool:
    return any(p.match(line.strip()) for p in _PAGE_NUMBER_PATTERNS)


def _is_metadata_line(line: str) -> bool:
    stripped = line.strip().lstrip(">").strip()
    return any(stripped.startswith(key) for key in ("文档类型", "业务领域", "权限组"))


def _is_heading_or_list_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if stripped.startswith(("-", "*", "•")):
        return True
    return bool(re.match(r"^\d+[.、．]", stripped))


def remove_page_numbers(pages: list[str]) -> list[str]:
    """删除独立成行的页码（纯数字 / 第 X 页 / Page N / - N -）。"""
    cleaned: list[str] = []
    for page in pages:
        lines = [ln for ln in page.splitlines() if not _is_page_number_line(ln)]
        cleaned.append("\n".join(lines).strip())
    return cleaned


def _page_edge_lines(page: str, *, top_lines: int = 2, bottom_lines: int = 2):
    lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
    top = lines[:top_lines]
    bottom = lines[-bottom_lines:] if len(lines) > top_lines else []
    return top, bottom


def remove_repeated_headers_footers(
    pages: list[str], *, min_pages: int = 3
) -> list[str]:
    """页眉页脚去除：规范化文本行出现在页顶/底 且 在 >= min_pages 页重复 → 删除。

    跨页重复是页眉页脚的强证据，避免误删正文中仅出现一次的短行。
    """
    if len(pages) < 2:
        return pages
    top_counts: dict[str, int] = {}
    bottom_counts: dict[str, int] = {}
    for page in pages:
        top, bottom = _page_edge_lines(page)
        for line in top:
            top_counts[line] = top_counts.get(line, 0) + 1
        for line in bottom:
            bottom_counts[line] = bottom_counts.get(line, 0) + 1

    cleaned: list[str] = []
    for page in pages:
        lines = page.splitlines()
        top, bottom = _page_edge_lines(page)
        keep: list[str] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if i < 2 and top_counts.get(stripped, 0) >= min_pages:
                continue
            if i >= len(lines) - 2 and bottom_counts.get(stripped, 0) >= min_pages:
                continue
            keep.append(line)
        cleaned.append("\n".join(keep).strip())
    return cleaned


def merge_hyphenated_words(pages: list[str]) -> list[str]:
    """英文连字符断词合并：docu-\\nment → document（仅英文）。"""
    return [re.sub(r"([a-zA-Z])-\n([a-zA-Z])", r"\1\2", page) for page in pages]


_CHINESE_SENTENCE_END = "。！？；：」』）】…"


def merge_chinese_wrapped_lines(pages: list[str]) -> list[str]:
    """中文断行合并：行尾不是句读标点 → 与下行拼接（中文行尾无空格）。"""
    merged: list[str] = []
    for page in pages:
        lines = page.splitlines()
        output: list[str] = []
        for line in lines:
            stripped = line.strip()
            if (
                output
                and stripped
                and not output[-1].endswith(tuple(_CHINESE_SENTENCE_END))
                and not stripped.startswith(("#", "-", "*", "•"))
                and not _is_metadata_line(stripped)
                and not _is_page_number_line(stripped)
            ):
                output[-1] = output[-1] + stripped
            else:
                output.append(line)
        merged.append("\n".join(output).strip())
    return merged


def merge_orphan_lines(pages: list[str], *, max_chars: int = 30) -> list[str]:
    """孤儿行合并：短行（< max_chars）且非标题/列表/元数据/页码 → 并入上一行。"""
    merged: list[str] = []
    for page in pages:
        lines = page.splitlines()
        output: list[str] = []
        for line in lines:
            stripped = line.strip()
            if (
                output
                and stripped
                and len(stripped) < max_chars
                and not _is_heading_or_list_line(stripped)
                and not _is_metadata_line(stripped)
                and not _is_page_number_line(stripped)
            ):
                output[-1] = output[-1] + stripped
            else:
                output.append(line)
        merged.append("\n".join(output).strip())
    return merged


_OCR_NOISE_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PURE_SYMBOL_LINE = re.compile(r'^[\s|—–•·*\-_=~`\'"（）()\[\]【】]+$')


def clean_ocr_noise(pages: list[str]) -> list[str]:
    """OCR 后处理：控制字符、纯符号孤行、中文字符间误插空格。"""
    cleaned: list[str] = []
    for page in pages:
        page = _OCR_NOISE_CHARS.sub("", page)
        # lookaround 一次删除所有中文字符间的空格（不消费字符）
        page = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", page)
        lines = [ln for ln in page.splitlines() if not _PURE_SYMBOL_LINE.match(ln.strip())]
        cleaned.append("\n".join(lines).strip())
    return cleaned


def clean_pdf_pages(pages: list[str]) -> list[str]:
    """组合清理管线（顺序固定）：页码 → 页眉页脚 → 断词 → 中文断行 → 孤儿行 → OCR 噪声。"""
    return clean_ocr_noise(
        merge_orphan_lines(
            merge_chinese_wrapped_lines(
                merge_hyphenated_words(
                    remove_repeated_headers_footers(remove_page_numbers(pages))
                )
            )
        )
    )
