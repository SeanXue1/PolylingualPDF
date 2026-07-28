from src.cluster.cluster_builder import _merge_overlapping_paragraphs
from src.image_filter import filter_image_text
from src.models import TextBlock


def test_merge_overlapping_paragraphs_handles_small_cross_overlap():
    para_a = {
        "bbox": (0, 0, 100, 100),
        "boxes": [TextBlock(bbox=(0, 0, 100, 100), text="A", source="native", page_num=1)],
        "text": "A",
        "lines": [((0, 0, 100, 100), "A", 12.0)],
    }
    para_b = {
        "bbox": (80, 30, 180, 70),
        "boxes": [TextBlock(bbox=(80, 30, 180, 70), text="B", source="native", page_num=1)],
        "text": "B",
        "lines": [((80, 30, 180, 70), "B", 12.0)],
    }

    merged = _merge_overlapping_paragraphs([para_a, para_b])

    assert len(merged) == 1
    assert merged[0]["bbox"] == (0, 0, 180, 100)
    assert merged[0]["text"] == "AB"


def test_merge_overlapping_paragraphs_merges_fragmented_same_column_lines():
    para_a = {
        "bbox": (20, 0, 220, 40),
        "boxes": [TextBlock(bbox=(20, 0, 220, 40), text="Line 1", source="ocr", page_num=1)],
        "text": "Line 1",
        "lines": [((20, 0, 220, 40), "Line 1", 14.0)],
    }
    para_b = {
        "bbox": (22, 46, 218, 86),
        "boxes": [TextBlock(bbox=(22, 46, 218, 86), text="Line 2", source="ocr", page_num=1)],
        "text": "Line 2",
        "lines": [((22, 46, 218, 86), "Line 2", 14.0)],
    }

    merged = _merge_overlapping_paragraphs([para_a, para_b])

    assert len(merged) == 1
    assert merged[0]["text"] == "Line 1Line 2"
    assert len(merged[0]["lines"]) == 2


def test_filter_image_text_removes_blocks_inside_photo_but_keeps_caption():
    blocks = [
        TextBlock(bbox=(450, 450, 520, 500), text="731", source="ocr", page_num=1),
        TextBlock(bbox=(40, 40, 100, 70), text="Caption", source="ocr", page_num=1),
    ]
    image_bboxes = [(100, 100, 220, 220)]

    filtered = filter_image_text(
        blocks,
        image_bboxes,
        source_dpi=300,
        page_width=595,
        page_height=842,
    )

    assert len(filtered) == 1
    assert filtered[0].text == "Caption"
