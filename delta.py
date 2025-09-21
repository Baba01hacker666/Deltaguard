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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
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
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save {filepath}: {e}")
        return False

def compute_file_hash(filepath: str) -> Optional[str]:
    """Compute SHA-256 hash of file. Returns None if fails."""
    try:
        hasher = hashlib.sha256()
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
    def __init__(self, source_dir: str, target_dir: str, resume: bool = True):
        self.source_dir = Path(source_dir).resolve()
        self.target_dir = Path(target_dir).resolve()
        self.state = self.load_state(resume)
        self.stats = {"copied": 0, "skipped": 0, "failed": 0, "fixed": 0}

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

    def should_copy_file(self, src_file: Path) -> bool:
        """Determine if file needs to be copied (based on hash or existence)."""
        rel_path = src_file.relative_to(self.source_dir)
        target_file = self.target_dir / rel_path

        if not target_file.exists():
            return True

        src_hash = compute_file_hash(str(src_file))
        if src_hash is None:
            AutoFixer.skip_and_log_corrupt_file(str(src_file), "unreadable source")
            self.stats["failed"] += 1
            return False

        target_hash = compute_file_hash(str(target_file))
        if target_hash is None:
            logger.warning(f"⚠️ Target file corrupt: {target_file}. Will overwrite.")
            return True

        return src_hash != target_hash

    def copy_file_with_retries(self, src: Path, dst: Path, max_retries: int = 3) -> bool:
        """Copy file with retries and auto-fix attempts."""
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
                return True

            except PermissionError:
                logger.warning(f"🔒 Permission denied on attempt {attempt} for {dst}. Trying to fix...")
                if not AutoFixer.fix_file_permissions(str(dst.parent)):
                    if attempt == max_retries:
                        logger.error(f"❌ Perm fix failed after {max_retries} attempts: {dst}")
                        self.stats["failed"] += 1
                        return False
                time.sleep(0.5)  # wait before retry

            except Exception as e:
                logger.warning(f"⚠️ Copy failed (attempt {attempt}): {e}")
                if attempt == max_retries:
                    logger.error(f"❌ Gave up after {max_retries} attempts: {src}")
                    self.stats["failed"] += 1
                    return False
                time.sleep(1)

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
            all_files = list(self.source_dir.rglob("*"))
            total_files = len([f for f in all_files if f.is_file()])
            logger.info(f"📁 Found {total_files} files to process.")

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

                # Save state every 10 files (checkpoint)
                if (self.stats["copied"] + self.stats["failed"]) % 10 == 0:
                    self.save_state()

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

    args = parser.parse_args()

    if args.quiet:
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]

    engine = BackupEngine(args.source, args.target, resume=not args.fresh)
    success = engine.run()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
