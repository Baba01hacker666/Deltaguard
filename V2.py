#!/usr/bin/env python3
import os
import sys
import json
import shutil
import hashlib
import logging
import argparse
from pathlib import Path
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# ========================
# CONFIG & GLOBALS
# ========================
STATE_FILE = "backup_state.json"
LOG_FILE = "backup.log"
CONFIG_FILE = "backup_config.json"

# Default configuration
DEFAULT_CONFIG = {
    "partial_hash_size": 64 * 1024,  # 64KB for partial hashing
    "full_hash_threshold": 10 * 1024 * 1024,  # 10MB - files larger than this will use partial hashing first
    "use_mtime": True,  # Use modification time as a quick check
    "max_retries": 3,
    "retry_delay": 1.0,  # seconds
    "checkpoint_interval": 50,  # Save state every N files
    "log_level": "INFO"
}

# Setup logging
def setup_logging(log_level: str = "INFO"):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ========================
# UTILITY FUNCTIONS
# ========================

def load_json(filepath: str, default: Any = None) -> Any:
    """Safely load JSON file. Returns default if fails."""
    try:
        if not Path(filepath).exists():
            return default
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ Failed to load {filepath}: {e}. Using default.")
        return default

def save_json(filepath: str, data: Any) -> bool:
    """Safely save data to JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save {filepath}: {e}")
        return False

def compute_file_hash(filepath: str, partial: bool = False, size: int = 64*1024) -> Optional[str]:
    """
    Compute SHA-256 hash of file.
    If partial=True, only hash the first and last 'size' bytes.
    Returns None if fails.
    """
    try:
        hasher = hashlib.sha256()
        file_size = os.path.getsize(filepath)
        
        with open(filepath, 'rb') as f:
            if partial and file_size > size * 2:
                # Hash first chunk
                hasher.update(f.read(size))
                # Seek to end and hash last chunk
                f.seek(-size, os.SEEK_END)
                hasher.update(f.read(size))
            else:
                # Hash entire file
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"⚠️ Could not hash {filepath}: {e}")
        return None

def ensure_dir(path: str) -> bool:
    """Ensure directory exists. Auto-fix by creating if missing."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        # Fix permissions (rwx for user)
        if os.name != 'nt':  # Skip on Windows
            os.chmod(path, 0o700)
        return True
    except Exception as e:
        logger.error(f"❌ Could not create/access directory {path}: {e}")
        return False

def is_file_readable(filepath: str) -> bool:
    """Check if file is readable."""
    try:
        with open(filepath, 'rb') as f:
            f.read(1)
        return True
    except Exception:
        return False

# ========================
# AUTO-FIXER MODULE
# ========================

class AutoFixer:
    @staticmethod
    def fix_file_permissions(filepath: str) -> bool:
        """Attempt to fix file permissions."""
        try:
            if os.name != 'nt':
                os.chmod(filepath, 0o600)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not fix permissions for {filepath}: {e}")
            return False

    @staticmethod
    def fix_directory(path: str) -> bool:
        """Attempt to fix/create directory."""
        return ensure_dir(path)

    @staticmethod
    def skip_and_log_corrupt_file(filepath: str, reason: str = "unknown") -> None:
        """Log and skip corrupt/unreadable file."""
        logger.error(f"🚫 Skipping corrupt file: {filepath} (reason: {reason})")
        # Record in state for user review
        state = load_json(STATE_FILE, {})
        corrupted = state.get("corrupted_files", [])
        corrupted.append({"file": filepath, "reason": reason, "timestamp": datetime.now().isoformat()})
        state["corrupted_files"] = corrupted
        save_json(STATE_FILE, state)

# ========================
# MAIN BACKUP ENGINE
# ========================

class BackupEngine:
    def __init__(self, source_dir: str, target_dir: str, resume: bool = True, config: Dict[str, Any] = None):
        self.source_dir = Path(source_dir).resolve()
        self.target_dir = Path(target_dir).resolve()
        self.config = config or DEFAULT_CONFIG
        self.state = self.load_state(resume)
        self.stats = {"copied": 0, "skipped": 0, "failed": 0, "fixed": 0}
        self.file_cache = {}  # Cache for file metadata to avoid repeated stat calls

    def load_state(self, resume: bool) -> Dict[str, Any]:
        """Load or initialize state."""
        default_state = {
            "last_run": None,
            "completed_files": [],
            "failed_files": [],
            "corrupted_files": [],
            "version": "1.0"
        }
        if not resume:
            return default_state
        state = load_json(STATE_FILE, default_state)
        # Validate structure
        for key in default_state:
            if key not in state:
                state[key] = default_state[key]
        return state

    def save_state(self) -> None:
        """Save current state."""
        self.state["last_run"] = datetime.now().isoformat()
        self.state["stats"] = self.stats
        save_json(STATE_FILE, self.state)

    def get_file_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Get file metadata with caching to avoid repeated stat calls."""
        str_path = str(filepath)
        if str_path in self.file_cache:
            return self.file_cache[str_path]
        
        try:
            stat = filepath.stat()
            metadata = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "exists": True
            }
            self.file_cache[str_path] = metadata
            return metadata
        except Exception as e:
            logger.warning(f"⚠️ Could not get metadata for {filepath}: {e}")
            return {"exists": False}

    def should_copy_file(self, src_file: Path) -> bool:
        """
        Determine if file needs to be copied using a multi-stage comparison:
        1. Check if target exists
        2. Compare file sizes
        3. Compare modification times (if enabled)
        4. Compare partial hashes (for large files)
        5. Compare full hashes (if necessary)
        """
        rel_path = src_file.relative_to(self.source_dir)
        target_file = self.target_dir / rel_path
        
        # Get source metadata
        src_meta = self.get_file_metadata(src_file)
        if not src_meta["exists"]:
            AutoFixer.skip_and_log_corrupt_file(str(src_file), "unreadable source")
            self.stats["failed"] += 1
            return False
        
        # Check if target exists
        target_meta = self.get_file_metadata(target_file)
        if not target_meta["exists"]:
            return True
        
        # 1. Compare file sizes
        if src_meta["size"] != target_meta["size"]:
            logger.debug(f"Size differs for {src_file.name}: {src_meta['size']} vs {target_meta['size']}")
            return True
        
        # 2. Compare modification times (if enabled)
        if self.config.get("use_mtime", True) and src_meta["mtime"] != target_meta["mtime"]:
            logger.debug(f"Modification time differs for {src_file.name}")
            return True
        
        # 3. For large files, use partial hashing first
        if src_meta["size"] > self.config.get("full_hash_threshold", 10*1024*1024):
            partial_hash_size = self.config.get("partial_hash_size", 64*1024)
            src_partial_hash = compute_file_hash(str(src_file), partial=True, size=partial_hash_size)
            if src_partial_hash is None:
                AutoFixer.skip_and_log_corrupt_file(str(src_file), "unreadable source")
                self.stats["failed"] += 1
                return False
                
            target_partial_hash = compute_file_hash(str(target_file), partial=True, size=partial_hash_size)
            if target_partial_hash is None:
                logger.warning(f"⚠️ Target file corrupt: {target_file}. Will overwrite.")
                return True
            
            if src_partial_hash != target_partial_hash:
                logger.debug(f"Partial hash differs for {src_file.name}")
                return True
        
        # 4. Full hash comparison (final check)
        src_hash = compute_file_hash(str(src_file))
        if src_hash is None:
            AutoFixer.skip_and_log_corrupt_file(str(src_file), "unreadable source")
            self.stats["failed"] += 1
            return False
            
        target_hash = compute_file_hash(str(target_file))
        if target_hash is None:
            logger.warning(f"⚠️ Target file corrupt: {target_file}. Will overwrite.")
            return True
        
        if src_hash != target_hash:
            logger.debug(f"Full hash differs for {src_file.name}")
            return True
        
        # All checks passed, file is the same
        return False

    def copy_file_with_retries(self, src: Path, dst: Path) -> bool:
        """Copy file with retries and auto-fix attempts."""
        max_retries = self.config.get("max_retries", 3)
        retry_delay = self.config.get("retry_delay", 1.0)
        
        for attempt in range(1, max_retries + 1):
            try:
                # Ensure target directory exists
                if not AutoFixer.fix_directory(str(dst.parent)):
                    self.stats["failed"] += 1
                    return False

                # Try copying
                shutil.copy2(src, dst)
                logger.info(f"✅ Copied: {src} -> {dst}")
                self.stats["copied"] += 1
                
                # Update file cache for the new file
                self.file_cache[str(dst)] = self.get_file_metadata(dst)
                return True

            except PermissionError:
                logger.warning(f"🔒 Permission denied on attempt {attempt} for {dst}. Trying to fix...")
                if not AutoFixer.fix_file_permissions(str(dst.parent)):
                    if attempt == max_retries:
                        logger.error(f"❌ Perm fix failed after {max_retries} attempts: {dst}")
                        self.stats["failed"] += 1
                        return False
                time.sleep(retry_delay)

            except Exception as e:
                logger.warning(f"⚠️ Copy failed (attempt {attempt}): {e}")
                if attempt == max_retries:
                    logger.error(f"❌ Gave up after {max_retries} attempts: {src}")
                    self.stats["failed"] += 1
                    return False
                time.sleep(retry_delay)

        return False

    def run(self) -> bool:
        """Run backup process."""
        logger.info(f"🚀 Starting backup from '{self.source_dir}' to '{self.target_dir}'")

        if not self.source_dir.exists():
            logger.error(f"❌ Source directory does not exist: {self.source_dir}")
            return False

        if not ensure_dir(str(self.target_dir)):
            logger.error(f"❌ Cannot access/create target directory: {self.target_dir}")
            return False

        try:
            # Use generator to avoid loading all files into memory
            all_files = self.source_dir.rglob("*")
            total_files = sum(1 for f in all_files if f.is_file())
            logger.info(f"📁 Found {total_files} files to process.")
            
            # Reset the generator
            all_files = self.source_dir.rglob("*")
            
            processed_files = 0
            for src_file in all_files:
                if not src_file.is_file():
                    continue

                if not is_file_readable(str(src_file)):
                    AutoFixer.skip_and_log_corrupt_file(str(src_file), "unreadable")
                    self.stats["failed"] += 1
                    continue

                if not self.should_copy_file(src_file):
                    self.stats["skipped"] += 1
                    continue

                rel_path = src_file.relative_to(self.source_dir)
                target_file = self.target_dir / rel_path

                if self.copy_file_with_retries(src_file, target_file):
                    # Record success
                    completed = self.state.get("completed_files", [])
                    completed.append(str(src_file))
                    self.state["completed_files"] = list(set(completed))  # dedupe
                else:
                    # Record failure
                    failed = self.state.get("failed_files", [])
                    failed.append(str(src_file))
                    self.state["failed_files"] = list(set(failed))

                processed_files += 1
                
                # Save state at intervals
                checkpoint_interval = self.config.get("checkpoint_interval", 50)
                if processed_files % checkpoint_interval == 0:
                    self.save_state()
                    logger.info(f"📊 Processed {processed_files}/{total_files} files...")

            # Final save
            self.save_state()

            logger.info("🎉 Backup completed!")
            logger.info(f"📊 Stats: {self.stats}")
            if self.stats["failed"] > 0:
                logger.warning(f"❗ {self.stats['failed']} files failed. See {STATE_FILE} for details.")

            return self.stats["failed"] == 0

        except Exception as e:
            logger.exception(f"💥 Unexpected error during backup: {e}")
            self.save_state()
            return False

# ========================
# CLI ENTRY POINT
# ========================

def main():
    parser = argparse.ArgumentParser(description="Self-Healing File Backup Tool")
    parser.add_argument("source", help="Source directory to backup")
    parser.add_argument("target", help="Target backup directory")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore previous state)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()

    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        user_config = load_json(args.config, {})
        config.update(user_config)
    
    # Setup logging with configured level
    global logger
    logger = setup_logging(config.get("log_level", "INFO"))
    
    if args.quiet:
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]

    engine = BackupEngine(args.source, args.target, resume=not args.fresh, config=config)
    success = engine.run()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
