import logging
import threading
import hashlib
import os
import shutil
from rich.logging import RichHandler
from pathlib import Path
from typing import List

def compute_file_hash(filepath: Path, partial: bool = False, size: int = 64*1024) -> str:
    hasher = hashlib.sha256()
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        if partial and file_size > size * 2:
            hasher.update(f.read(size))
            f.seek(-size, os.SEEK_END)
            hasher.update(f.read(size))
        else:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
    return hasher.hexdigest()

def should_copy_file(src: Path, dst: Path) -> bool:
    if not dst.exists(): return True
    if src.stat().st_size != dst.stat().st_size: return True
    if src.stat().st_mtime != dst.stat().st_mtime: return True
    
    # Partial hash for large files (>10MB)
    if src.stat().st_size > 10 * 1024 * 1024:
        if compute_file_hash(src, partial=True) != compute_file_hash(dst, partial=True):
            return True
            
    return compute_file_hash(src) != compute_file_hash(dst)

def process_single_file(src: Path, dst_root: Path, source_root: Path, stats: 'Stats'):
    try:
        rel_path = src.relative_to(source_root)
        dst = dst_root / rel_path
        
        if not should_copy_file(src, dst):
            stats.increment("skipped")
            return

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        stats.increment("copied")
    except Exception as e:
        logging.getLogger("deltaguard").error(f"Failed {src}: {e}")
        stats.increment("failed")

def discover_files(source_dir: Path) -> List[Path]:
    return [f for f in source_dir.rglob("*") if f.is_file()]

class Stats:
    def __init__(self):
        self._stats = {"copied": 0, "skipped": 0, "failed": 0, "fixed": 0, "discovered": 0}
        self._lock = threading.Lock()

    def increment(self, key):
        with self._lock:
            if key in self._stats:
                self._stats[key] += 1

    def get(self, key):
        with self._lock:
            return self._stats.get(key, 0)

    def all(self):
        with self._lock:
            return self._stats.copy()

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor
import argparse

def run_backup(source: Path, target: Path) -> 'Stats':
    files = discover_files(source)
    stats = Stats()
    stats._stats["discovered"] = len(files)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        main_task = progress.add_task("[green]Backing up files...", total=len(files))
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_single_file, f, target, source, stats) for f in files]
            for future in futures:
                future.add_done_callback(lambda _: progress.advance(main_task))
                
    return stats

def print_report(stats: 'Stats'):
    console = Console()
    table = Table(title="Deltaguard Backup Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="magenta")
    
    all_stats = stats.all()
    for key, value in all_stats.items():
        table.add_row(key.capitalize(), str(value))
    
    console.print(table)

def setup_logging():
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger("deltaguard")

if __name__ == "__main__":
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Deltaguard: Fast Concurrent File Backup")
    parser.add_argument("source", type=Path, help="Source directory")
    parser.add_argument("target", type=Path, help="Target directory")
    
    args = parser.parse_args()
    
    if not args.source.exists() or not args.source.is_dir():
        logger.error(f"Source directory does not exist: {args.source}")
        exit(1)
        
    stats = run_backup(args.source, args.target)
    print_report(stats)
