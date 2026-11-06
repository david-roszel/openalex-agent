import pytest

from openalex_agent.freeze_pilot import snapshot, verify


def test_snapshot_preserves_bytes_and_detects_changes(tmp_path):
    source = tmp_path / "payload"
    original = b'{"prompt":"exact\\nmessage","temperature":null}\n'
    source.write_bytes(original)
    output = tmp_path / "frozen"
    receipt = snapshot({"requests.jsonl": source}, output)
    assert (output / "requests.jsonl").read_bytes() == original
    assert verify(receipt)["verified"]
    with pytest.raises(ValueError, match="already exists"):
        snapshot({"requests.jsonl": source}, output)
    (output / "requests.jsonl").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        verify(receipt)


@pytest.mark.parametrize("change", ["extra", "missing", "manifest", "symlink"])
def test_snapshot_inventory_and_manifest_are_pinned(tmp_path, change):
    source = tmp_path / "source"
    source.write_text("exact source")
    output = tmp_path / "frozen"
    receipt = snapshot({"file": source}, output)
    if change == "extra":
        (output / "extra").write_text("unexpected")
    elif change == "missing":
        (output / "file").unlink()
    elif change == "manifest":
        manifest = output / "manifest.json"
        manifest.write_text(manifest.read_text() + " ")
    else:
        (output / "file").unlink()
        (output / "file").symlink_to(source)
    with pytest.raises(ValueError):
        verify(receipt)


def test_credentials_and_unsafe_paths_cancel_without_partial_snapshot(tmp_path):
    source = tmp_path / "source"
    source.write_text("a configured TEST-KEY was accidentally logged")
    output = tmp_path / "frozen"
    with pytest.raises(ValueError, match="credential"):
        snapshot({"safe/file": source}, output, forbidden=("TEST-KEY",))
    assert not output.exists()
    with pytest.raises(ValueError, match="Unsafe"):
        snapshot({"../escape": source}, output)
    assert not output.exists()
