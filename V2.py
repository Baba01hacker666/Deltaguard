1#!/usr/bin/env python3
import os
import sys
import json
import shutil
import hashlib
import logging
import argparse
import fnmatch
from pathlib import Path
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple

# ========================
# CONFIG & GLOBALS
# ========================

# --- FIX: Store state and logs in a user-specific directory for consistency ---
APP_NAME = "self_healing_backup"
USER_DATA_DIR = Path.home() / ".local" / "share" / APP_NAME
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_STATE_FILE = USER_DATA_DIR / "backup_state.json"
DEFAULT_LOG_FILE = USER_DATA_DIR / "backup.log"
DEFAULT_CONFIG_FILE = USER_DATA_DIR / "backup_config.json"

# --- FIX: Implemented a proper configuration system ---
DEFAULT_CONFIG = {
    "state_file": str(DEFAULT_STATE_FILE),
    "log_file": str(DEFAULT_LOG_FILE),
    "comparison_mode": "mtime",  # "mtime" or "hash". "mtime" is much faster.
    "hash_algorithm": "sha256",
    "retry_count": 3,
    "checkpoint_interval": 50, # Save state every N files
    "exclude_patterns": [".git", "__pycache__", "*.tmp", "node_modules"],
    "symlinks": "preserve" # "follow", "preserve", or "skip"
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        # We will add the file handler after loading the config
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
            # --- FIX: JSON can't save sets, so convert them to lists before saving ---
            if isinstance(data, dict):
                data_to_save = {k: list(v) if isinstance(v, set) else v for k, v in data.items()}
            else:
                data_to_save = data
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save {filepath}: {e}")
        return False

def compute_file_hash(filepath: str, algorithm: str = 'sha256') -> Optional[str]:
    """Compute hash of file. Returns None if fails."""
    try:
        hasher = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
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
        return True
    except Exception as e:
        logger.error(f"❌ Could not create/access directory {path}: {e}")
        return False

# --- FIX: Replaced is_file_readable with a more robust check ---
def is_file_accessible(filepath: str) -> bool:
    """Check if file can be opened for reading."""
    try:
        with open(filepath, 'rb'):
            pass
        return True
    except Exception:
        return False

# ========================
# AUTO-FIXER MODULE
# ========================

class AutoFixer:
    # --- FIX: Removed dangerous fix_file_permissions. The tool should not alter security settings. ---
    # The "healing" is now limited to retries and creating missing directories.

    @staticmethod
    def fix_directory(path: str) -> bool:
        """Attempt to fix/create directory."""
        return ensure_dir(path)

    @staticmethod
    def skip_and_log_issue(filepath: str, reason: str, issue_type: str = "corrupted") -> None:
        """Log and skip problematic file."""
        logger.error(f"🚫 Skipping {issue_type} file: {filepath} (reason: {reason})")
        # This is now handled within the main engine for better state management

# ========================
# MAIN BACKUP ENGINE
# ========================

class BackupEngine:
    def __init__(self, source_dir: str, target_dir: str, config: Dict[str, Any], dry_run: bool = False):
        self.source_dir = Path(source_dir).resolve()
        self.target_dir = Path(target_dir).resolve()
        self.config = config
        self.dry_run = dry_run
        self.state_file = Path(config["state_file"])
        self.state = self.load_state()
        # --- FIX: Use sets for efficient in-memory operations ---
        self.state["completed_files"] = set(self.state.get("completed_files", []))
        self.state["failed_files"] = set(self.state.get("failed_files", []))
        self.stats = {"copied": 0, "skipped": 0, "failed": 0, "fixed": 0, "total_files": 0}

    def load_state(self) -> Dict[str, Any]:
        """Load or initialize state."""
        default_state = {
            "last_run": None,
            "completed_files": [],
            "failed_files": [],
            "version": "2.0"
        }
        state = load_json(str(self.state_file), default_state)
        for key in default_state:
            if key not in state:
                state[key] = default_state[key]
        return state

    def save_state(self) -> None:
        """Save current state."""
        self.state["last_run"] = datetime.now().isoformat()
        self.state["stats"] = self.stats
        save_json(str(self.state_file), self.state)

    def is_excluded(self, path: Path) -> bool:
        """Check if a path matches any exclusion pattern."""
        for pattern in self.config.get("exclude_patterns", []):
            if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern):
                return True
        return False

    def should_copy_file(self, src_file: Path) -> bool:
        """Determine if file needs to be copied."""
        rel_path = src_file.relative_to(self.source_dir)
        target_file = self.target_dir / rel_path

        if not target_file.exists():
            return True

        # --- FIX: Use fast mtime/size comparison by default ---
        if self.config["comparison_mode"] == "mtime":
            try:
                src_mtime = src_file.stat().st_mtime
                src_size = src_file.stat().st_size
                tgt_mtime = target_file.stat().st_mtime
                tgt_size = target_file.stat().st_size
                return src_mtime != tgt_mtime or src_size != tgt_size
            except OSError as e:
                logger.warning(f"⚠️ Could not stat file for comparison: {e}. Will recopy.")
                return True
        
        # --- FIX: Hashing is now an optional, slower mode ---
        elif self.config["comparison_mode"] == "hash":
            src_hash = compute_file_hash(str(src_file), self.config["hash_algorithm"])
            if src_hash is None:
                self.stats["failed"] += 1
                self.state["failed_files"].add(str(src_file))
                return False

            target_hash = compute_file_hash(str(target_file), self.config["hash_algorithm"])
            if target_hash is None:
                logger.warning(f"⚠️ Target file corrupt or unreadable: {target_file}. Will overwrite.")
                return True
            
            return src_hash != target_hash
        
        return False

    def copy_file_with_retries(self, src: Path, dst: Path) -> bool:
        """Copy file with retries."""
        max_retries = self.config["retry_count"]
        for attempt in range(1, max_retries + 1):
            try:
                if not AutoFixer.fix_directory(str(dst.parent)):
                    raise IOError(f"Could not create target directory {dst.parent}")

                if self.dry_run:
                    logger.info(f"🔎 [DRY RUN] Would copy: {src} -> {dst}")
                    self.stats["copied"] += 1
                    return True

                shutil.copy2(src, dst)
                logger.info(f"✅ Copied: {src} -> {dst}")
                self.stats["copied"] += 1
                return True

            except PermissionError as e:
                # --- FIX: Removed dangerous chmod. Just log and retry. ---
                logger.warning(f"🔒 Permission denied on attempt {attempt} for {dst}. Error: {e}. Retrying...")
            except Exception as e:
                logger.warning(f"⚠️ Copy failed (attempt {attempt}): {e}")

            if attempt < max_retries:
                time.sleep(1)

        logger.error(f"❌ Gave up after {max_retries} attempts: {src}")
        self.stats["failed"] += 1
        self.state["failed_files"].add(str(src))
        return False

    def run(self) -> bool:
        """Run backup process."""
        logger.info(f"🚀 Starting backup from '{self.source_dir}' to '{self.target_dir}'")
        if self.dry_run:
            logger.info("🔎 DRY RUN MODE: No files will be copied.")

        if not self.source_dir.exists():
            logger.error(f"❌ Source directory does not exist: {self.source_dir}")
            return False

        if not ensure_dir(str(self.target_dir)):
            logger.error(f"❌ Cannot access/create target directory: {self.target_dir}")
            return False

        try:
            # --- FIX: Iterate directly over the generator to save memory ---
            file_generator = self.source_dir.rglob("*")
            processed_files = 0
            for src_path in file_generator:
                self.stats["total_files"] += 1
                
                # --- FIX: Handle symlinks based on configuration ---
                if src_path.is_symlink():
                    action = self.config.get("symlinks", "preserve")
                    if action == "skip":
                        logger.debug(f"⏭️ Skipping symlink: {src_path}")
                        continue
                    # For 'follow' or 'preserve', rglob will give us the link itself.
                    # shutil.copy2 will preserve the symlink if it points to a valid target.
                    # If it's a broken symlink, copy2 will fail, which is acceptable.
                
                if not src_path.is_file():
                    continue

                if self.is_excluded(src_path):
                    logger.debug(f"🚫 Excluded by pattern: {src_path}")
                    self.stats["skipped"] += 1
                    continue

                if not is_file_accessible(str(src_path)):
                    logger.error(f"🚫 Skipping unreadable file: {src_path}")
                    self.stats["failed"] += 1
                    self.state["failed_files"].add(str(src_path))
                    continue
                
                if str(src_path) in self.state["completed_files"] and not self.should_copy_file(src_path):
                    self.stats["skipped"] += 1
                    continue

                rel_path = src_path.relative_to(self.source_dir)
                target_file = self.target_dir / rel_path

                if self.copy_file_with_retries(src_path, target_file):
                    self.state["completed_files"].add(str(src_path))
                else:
                    # Failure is already logged and counted in copy_file_with_retries
                    pass

                # --- FIX: Checkpointing is now configurable ---
                processed_files += 1
                if processed_files % self.config["checkpoint_interval"] == 0:
                    self.save_state()
                    logger.info(f"📊 Checkpoint: {processed_files} files processed...")

            # Final save
            self.save_state()

            logger.info("🎉 Backup completed!")
            logger.info(f"📊 Stats: {self.stats}")
            if self.stats["failed"] > 0:
                logger.warning(f"❗ {self.stats['failed']} files failed. See {self.state_file} for details.")

            return self.stats["failed"] == 0

        except Exception as e:
            logger.exception(f"💥 Unexpected error during backup: {e}")
            self.save_state()
            return False

# ========================
# CLI ENTRY POINT
# ========================

def main():
    parser = argparse.ArgumentParser(description="Improved Self-Healing File Backup Tool")
    parser.add_argument("source", help="Source directory to backup")
    parser.add_argument("target", help="Target backup directory")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore previous state)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    # --- FIX: Added new CLI arguments for new features ---
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without doing anything")
    parser.add_argument("--config", help=f"Path to config file (default: {DEFAULT_CONFIG_FILE})")
    parser.add_argument("--symlinks", choices=["follow", "preserve", "skip"], default="preserve", help="How to handle symbolic links")

    args = parser.parse_args()

    # --- FIX: Load configuration and setup logging based on it ---
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_FILE
    config = load_json(str(config_path), DEFAULT_CONFIG)
    
    # Override config with CLI args
    if args.symlinks:
        config["symlinks"] = args.symlinks

    # Setup file logger now that we have the path
    file_handler = logging.FileHandler(config["log_file"], encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    if args.quiet:
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]

    if args.fresh and config["state_file"].exists():
        os.remove(config["state_file"])
        logger.info("🗑️ Fresh start: removed previous state file.")

    engine = BackupEngine(args.source, args.target, config, dry_run=args.dry_run)
    success = engine.run()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
