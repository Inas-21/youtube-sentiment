from src.utils.text import clean_text


def test_removes_urls_and_mentions():
    assert clean_text("@john check https://youtu.be/abc www.site.com now") == "check now"
    assert clean_text("see r/india and u/someone") == "see and"


def test_emojis_become_words():
    out = clean_text("Love it 😂❤️")
    assert "joy" in out and "heart" in out
    assert "_" not in out


def test_lowercase_and_punctuation():
    assert clean_text("GREAT video!!! Don't stop...") == "great video don t stop"


def test_html_entities():
    assert clean_text("Tom &amp; Jerry &#39;rocks&#39;") == "tom jerry rocks"


def test_non_latin_letters_kept():
    assert clean_text("बहुत अच्छा!") != ""


def test_empty_and_invalid_inputs():
    assert clean_text("") == ""
    assert clean_text("   !!! ") == ""
    assert clean_text(None) == ""
