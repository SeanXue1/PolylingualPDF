from src.models import Paragraph, TextBlock, TextCluster
from src.renderer import _split_cluster_translation
from src.translator import _BaseTranslator


class DummyTranslator(_BaseTranslator):
    def __init__(self):
        self.cache = None
        self.json_called = False
        self.paragraph_called = False

    def _call_paragraph_api(self, text: str) -> str:
        self.paragraph_called = True
        raise AssertionError("freeform paragraph API should not be used")

    def _call_json_api(self, input_json: str, n_items: int) -> str:
        self.json_called = True
        return '{"items":[{"id":0,"text":"中文翻译"}]}'


def test_translate_paragraph_uses_json_path():
    translator = DummyTranslator()
    result = translator.translate_paragraph("日本語の段落")

    assert result == "中文翻译"
    assert translator.json_called is True
    assert translator.paragraph_called is False


def test_split_cluster_translation_prefers_paragraph_count():
    para1 = Paragraph(blocks=[TextBlock(bbox=(0, 0, 10, 10), text="one", source="ocr", page_num=1)])
    para2 = Paragraph(blocks=[TextBlock(bbox=(0, 20, 10, 30), text="two", source="ocr", page_num=1)])
    cluster = TextCluster(paragraphs=[para1, para2], translation="第一段落译文\n第二段落译文")

    segments = _split_cluster_translation(cluster.translation, cluster.paragraphs)

    assert segments == ["第一段落译文", "第二段落译文"]
