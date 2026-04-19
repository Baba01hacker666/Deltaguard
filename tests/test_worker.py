from pathlib import Path
from delta_rich import Stats, should_copy_file, process_single_file
import shutil

def test_should_copy_file(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("content")
    
    # Target doesn't exist
    assert should_copy_file(src, dst) == True
    
    # Target exists, same content
    shutil.copy2(src, dst)
    assert should_copy_file(src, dst) == False
    
    # Target exists, different content
    dst.write_text("different")
    assert should_copy_file(src, dst) == True

def test_process_single_file(tmp_path):
    source_root = tmp_path / "source"
    dst_root = tmp_path / "target"
    source_root.mkdir()
    dst_root.mkdir()
    
    src = source_root / "test.txt"
    src.write_text("content")
    
    stats = Stats()
    process_single_file(src, dst_root, source_root, stats)
    
    assert stats.get("copied") == 1
    assert stats.get("skipped") == 0
    assert (dst_root / "test.txt").exists()
    
    # Process again, should skip
    process_single_file(src, dst_root, source_root, stats)
    assert stats.get("copied") == 1
    assert stats.get("skipped") == 1
