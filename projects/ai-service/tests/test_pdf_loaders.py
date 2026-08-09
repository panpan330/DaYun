"""PDF 加载器测试：文本提取、扫描页检测、清理管线、OCR 兜底、loaders 接入。"""

import pymupdf
import pytest

from app.rag.pdf_loaders import extract_pdf_pages, is_scanned_page


def make_text_pdf(path, pages_text):
    """用 pymupdf 生成文本型 PDF fixture（内置中文字体 china-s）。"""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11, fontname="china-s")
    doc.save(path)
    doc.close()


def test_extract_pdf_pages_text_pdf(tmp_path):
    pdf_path = tmp_path / "policy.pdf"
    make_text_pdf(pdf_path, ["第一页内容", "第二页内容"])
    pages = extract_pdf_pages(pdf_path)
    assert len(pages) == 2
    assert "第一页内容" in pages[0]
    assert "第二页内容" in pages[1]


def test_extract_pdf_pages_empty_page_detected(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    doc.new_page()  # 空白页 = 模拟扫描页（无文本层）
    doc.save(pdf_path)
    doc.close()
    pages = extract_pdf_pages(pdf_path)
    assert len(pages) == 1
    assert is_scanned_page(pages[0]) is True


def test_extract_pdf_pages_normal_text_not_scanned():
    assert is_scanned_page("正常文本内容足够长可以被识别为非扫描页") is False


def test_extract_pdf_pages_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(tmp_path / "nope.pdf")


# ---- Task 2: 页级清理管线 ----


from app.rag.pdf_loaders import (  # noqa: E402
    clean_ocr_noise,
    clean_pdf_pages,
    merge_chinese_wrapped_lines,
    merge_hyphenated_words,
    merge_orphan_lines,
    remove_page_numbers,
    remove_repeated_headers_footers,
)


def test_remove_repeated_headers_footers():
    pages = [
        "公司内部文档\n第一页正文\n机密文件",
        "公司内部文档\n第二页正文\n机密文件",
        "公司内部文档\n第三页正文\n机密文件",
    ]
    cleaned = remove_repeated_headers_footers(pages)
    assert "公司内部文档" not in cleaned[0]
    assert "机密文件" not in cleaned[0]
    assert "第二页正文" in cleaned[1]


def test_remove_page_numbers_alone():
    pages = ["正文开始", "1", "第 2 页", "正文继续"]
    cleaned = remove_page_numbers(pages)
    assert "1" not in cleaned[1]
    assert "第 2 页" not in cleaned[2]
    assert "正文开始" in cleaned[0]


def test_merge_hyphenated_words():
    pages = ["This is a docu-\nment about PDFs."]
    merged = merge_hyphenated_words(pages)
    assert "document" in merged[0]
    assert "docu-\nment" not in merged[0]


def test_merge_chinese_wrapped_lines():
    pages = ["这是一段很长的中文\n文本跨行显示。"]
    merged = merge_chinese_wrapped_lines(pages)
    assert "中文文本跨行显示。" in merged[0]


def test_chinese_wrapped_line_ends_with_period_stays():
    pages = ["第一句结束。\n第二句开始。"]
    merged = merge_chinese_wrapped_lines(pages)
    lines = merged[0].splitlines()
    assert lines[0].endswith("。")


def test_merge_orphan_lines():
    pages = ["第一段完整内容\n孤行短文本\n第二段"]
    merged = merge_orphan_lines(pages)
    assert "内容\n孤行短文本" in merged[0] or "内容孤行短文本" in merged[0]


def test_clean_ocr_noise():
    pages = ["中 文 之 间 有空格\n\u0002控制字符\u0003\n|—|"]
    cleaned = clean_ocr_noise(pages)
    assert "中文之间" in cleaned[0]
    assert "\u0002" not in cleaned[0]
    assert "|—|" not in cleaned[0]


def test_clean_pdf_pages_compose_pipeline():
    pages = ["页眉\n第一页正文内容\n1", "页眉\n第二页正文内容\n2", "页眉\n第三页正文内容\n3"]
    cleaned = clean_pdf_pages(pages)
    assert "页眉" not in cleaned[0]
    assert "1" not in cleaned[0]
    assert "第一页正文内容" in cleaned[0]


# ---- Task 3: OCR 兜底 ----


from app.rag.pdf_loaders import PdfOcrUnavailableError, _ocr_page  # noqa: E402


class FakeOcrEngine:
    def __call__(self, img_bytes):
        return ([[[[0, 0], [100, 0], [100, 20], [0, 20]], "扫描识别文本", 0.98]], None)


def test_extract_pdf_pages_scanned_page_uses_ocr(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "X", fontsize=11)  # 少于阈值字符 → 扫描页
    doc.save(pdf_path)
    doc.close()

    from app.rag import pdf_loaders as pl

    monkeypatch.setattr(pl, "_get_ocr_engine", lambda: FakeOcrEngine())
    pages = extract_pdf_pages(pdf_path)
    assert "扫描识别文本" in pages[0]


def test_extract_pdf_pages_ocr_unavailable_raises(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    from app.rag import pdf_loaders as pl

    def broken_engine():
        raise PdfOcrUnavailableError("PDF_OCR_UNAVAILABLE: rapidocr not installed")

    monkeypatch.setattr(pl, "_get_ocr_engine", broken_engine)
    with pytest.raises(PdfOcrUnavailableError, match="PDF_OCR_UNAVAILABLE"):
        extract_pdf_pages(pdf_path)


def test_ocr_page_sorts_by_y():
    from app.rag.pdf_loaders import _sort_ocr_items

    result = [[[[0, 30], [100, 30], [100, 50], [0, 50]], "第二行", 0.9],
              [[[0, 0], [100, 0], [100, 20], [0, 20]], "第一行", 0.9]]
    items = _sort_ocr_items(result)
    assert items[0][1] == "第一行"
    assert items[1][1] == "第二行"


# ---- Task 4: loaders.py 接入 ----


from app.rag.loaders import load_document  # noqa: E402


def test_load_document_pdf_end_to_end(tmp_path):
    pdf_path = tmp_path / "policy.pdf"
    make_text_pdf(
        pdf_path,
        ["公司政策\n退款政策第一段内容\n1", "公司政策\n退款政策第二段内容\n2", "公司政策\n退款政策第三段内容\n3"],
    )
    doc = load_document(pdf_path)
    assert doc.metadata["file_extension"] == ".pdf"
    assert doc.metadata["source"] == pdf_path.name
    assert "退款政策" in doc.content
    # 3 页重复的页眉 "公司政策" 应被清理，正文保留
    assert doc.content.count("公司政策") <= 1
    assert doc.metadata["title"]
