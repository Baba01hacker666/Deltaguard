from pathlib import Path
from delta_rich import discover_files

def test_discover_files(tmp_path):
    (tmp_path / "file1.txt").write_text("content")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir/file2.txt").write_text("content")
    
    files = discover_files(tmp_path)
    assert len(files) == 2
    assert any(f.name == "file1.txt" for f in files)
