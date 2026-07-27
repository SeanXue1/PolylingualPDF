from src.merger import merge_paragraphs
from src.models import TextBlock


def test_single_block():
    blocks = [TextBlock(bbox=(0, 0, 100, 20), text="Hello", source="native", page_num=1)]
    paras = merge_paragraphs(blocks)
    assert len(paras) == 1
    assert paras[0].text == "Hello"


def test_vertical_merge():
    blocks = [
        TextBlock(bbox=(0, 0, 100, 20), text="Line 1", source="native", page_num=1),
        TextBlock(bbox=(0, 25, 100, 45), text="Line 2", source="native", page_num=1),
    ]
    paras = merge_paragraphs(blocks)
    assert len(paras) == 1
    assert paras[0].text == "Line 1Line 2"
    assert len(paras[0].line_bboxes) == 2
    assert paras[0].line_bboxes[0][1] == "Line 1"


def test_separate_paragraphs():
    blocks = [
        TextBlock(bbox=(0, 0, 100, 20), text="Para 1", source="native", page_num=1),
        TextBlock(bbox=(0, 100, 100, 120), text="Para 2", source="native", page_num=1),
    ]
    paras = merge_paragraphs(blocks)
    assert len(paras) == 2


def test_two_columns():
    blocks = [
        TextBlock(bbox=(0, 0, 100, 20), text="Left 1", source="native", page_num=1),
        TextBlock(bbox=(0, 25, 100, 45), text="Left 2", source="native", page_num=1),
        TextBlock(bbox=(200, 0, 300, 20), text="Right 1", source="native", page_num=1),
    ]
    paras = merge_paragraphs(blocks)
    assert len(paras) == 2
    assert "Left" in paras[0].text
    assert "Right" in paras[1].text


def test_three_column_reading_order():
    blocks = [
        TextBlock(bbox=(400, 0, 500, 20), text="Right column", source="native", page_num=1),
        TextBlock(bbox=(200, 0, 300, 20), text="Middle column", source="native", page_num=1),
        TextBlock(bbox=(0, 0, 100, 20), text="Left column", source="native", page_num=1),
    ]
    paras = merge_paragraphs(blocks)
    assert len(paras) == 3
    assert paras[0].text == "Left column"
    assert paras[1].text == "Middle column"
    assert paras[2].text == "Right column"


def test_empty():
    assert merge_paragraphs([]) == []
