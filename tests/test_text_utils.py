from text_utils import clean_text, tokens


def test_lowercases_and_strips_punctuation():
    assert clean_text("Hello, World!!") == "hello world"


def test_removes_redaction_tokens():
    assert clean_text("I paid XXXX on XX/XX/2020") == "i paid on 2020"


def test_collapses_whitespace():
    assert clean_text("a   b \n c") == "a b c"


def test_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_idempotent():
    once = clean_text("Debt Collector!!! XXXX called")
    assert clean_text(once) == once


def test_tokens():
    assert tokens("Hi there!") == ["hi", "there"]
