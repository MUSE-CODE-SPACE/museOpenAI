import pytest

from muselm.tokenizer import ByteBPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 50
    + "hello world, this is a byte level tokenizer test. " * 50
    + "유니코드 텍스트도 문제없이 처리합니다. " * 20
)


@pytest.fixture(scope="module")
def tokenizer():
    return ByteBPETokenizer.train(CORPUS, vocab_size=512)


def test_roundtrip_ascii(tokenizer):
    text = "the quick brown fox"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_unicode(tokenizer):
    text = "유니코드 텍스트 😀 emoji"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_arbitrary_bytes(tokenizer):
    # Byte-level tokenizers must round-trip anything, even unseen text.
    text = "\x00\x01 raw \t\n bytes ~!@#$%^&*()"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_size(tokenizer):
    # Training may stop early once no byte pair repeats, so the realized
    # vocab is at most the requested size but always covers the 256 bytes.
    assert 256 < tokenizer.vocab_size <= 512
    ids = tokenizer.encode("hello world")
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


def test_special_tokens(tokenizer):
    ids = tokenizer.encode("hi <|eos|> bye")
    assert tokenizer.eos_id in ids
    # Special tokens can be disabled.
    ids_plain = tokenizer.encode("<|eos|>", allow_special=False)
    assert tokenizer.eos_id not in ids_plain


def test_merges_reduce_length(tokenizer):
    # A trained tokenizer should compress repeated text below raw byte length.
    text = "the quick brown fox " * 10
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))


def test_save_load(tmp_path, tokenizer):
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    text = "roundtrip after reload 유니코드"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_vocab_too_small():
    with pytest.raises(ValueError):
        ByteBPETokenizer.train("abc", vocab_size=100)
