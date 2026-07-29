from src.cluster.cluster_builder import _merge_overlapping_paragraphs
from src.cluster.layout_cluster import LayoutClusterAlgorithm
from src.cluster.paragraph_detector import detect_paragraphs
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


def test_detect_paragraphs_splits_numbered_items_by_text():
    line_groups = [
        [(0, 0, 100, 20)],
        [(0, 24, 100, 44)],
        [(0, 60, 100, 80)],
        [(0, 84, 100, 104)],
    ]
    line_texts = [
        "1. First item",
        "continuation",
        "2. Second item",
        "continuation",
    ]

    paras = detect_paragraphs(line_groups, line_texts=line_texts)

    assert paras == [[0, 1], [2, 3]]


def test_layout_cluster_keeps_numbered_items_separate():
    blocks = [
        TextBlock(bbox=(0, 0, 100, 20), text="1. First item", source="ocr", page_num=1),
        TextBlock(bbox=(0, 24, 100, 44), text="continuation", source="ocr", page_num=1),
        TextBlock(bbox=(0, 60, 100, 80), text="2. Second item", source="ocr", page_num=1),
        TextBlock(bbox=(0, 84, 100, 104), text="continuation", source="ocr", page_num=1),
    ]

    algo = LayoutClusterAlgorithm(column_gap=50.0, line_spacing_ratio=1.5, min_region_size=0)
    paras = algo.build_clusters(blocks, page_width=200, page_height=200)

    assert len(paras) == 2
    assert paras[0].text.startswith("1. First item")
    assert paras[1].text.startswith("2. Second item")


def test_filter_image_text_removes_blocks_inside_photo_but_keeps_caption():
    blocks = [
        TextBlock(bbox=(450, 2700, 520, 2750), text="731", source="ocr", page_num=1),
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
