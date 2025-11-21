#!/usr/bin/env python3
"""
scripts/repair_notebooks.py

Scan .ipynb files under learning_modules and repair files where the notebook
JSON was embedded as a single JSON string inside a raw cell (common when
notebooks are generated incorrectly). For each such file, replace the file
contents with the parsed JSON object so the notebook renders properly.

Usage:
  python scripts/repair_notebooks.py [--dry-run]

This script will backup the original file to file.bak before overwriting.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_GLOB = (
    "notebooks/*.ipynb"  # Matches all notebooks; filter to specific ones if needed
)


def is_embedded_json(nb_path: Path) -> bool:
    try:
        j = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    # Check if first cell exists and is raw with a source that appears to be a JSON string
    cells = j.get("cells")
    if not cells or not isinstance(cells, list):
        return False
    first = cells[0]
    if first.get("cell_type") != "raw":
        return False
    source = first.get("source")
    if not source or not isinstance(source, list):
        return False

    # Try multiple approaches to parse the source
    # Approach 1: Direct join (strings in source already have proper escapes from JSON parsing)
    joined = "".join(source).strip()
    if not joined:
        return False

    # Try direct parsing
    try:
        parsed = json.loads(joined)
        if isinstance(parsed, dict) and "cells" in parsed:
            return True
    except json.JSONDecodeError:
        pass  # Try next approach

    # Approach 2: Try with raw string decoding (in case of double escaping)
    try:
        # Encode and decode to handle escape sequences
        decoded = joined.encode("utf-8").decode("unicode_escape")
        parsed = json.loads(decoded)
        if isinstance(parsed, dict) and "cells" in parsed:
            return True
    except Exception:
        pass  # Try next approach

    # Approach 3: Try with latin-1 escape decoding
    try:
        decoded = joined.encode("utf-8").decode("raw_unicode_escape")
        parsed = json.loads(decoded)
        if isinstance(parsed, dict) and "cells" in parsed:
            return True
    except Exception:
        pass

    # Approach 4: Try with regex-based quote escaping
    try:
        import re

        escaped = joined
        # Find all f-string dict accesses and escape the quotes
        # Pattern: {something["..."] inside the string
        escaped = re.sub(
            r'\{([^}]*)\["([^"]*?)"\]',
            lambda m: "{" + m.group(1) + '[\\"' + m.group(2) + '\\"]',
            escaped,
        )
        parsed = json.loads(escaped)
        if isinstance(parsed, dict) and "cells" in parsed:
            return True
    except Exception:
        pass

    return False


def repair_file(nb_path: Path, dry_run: bool = False) -> bool:
    """
    Repair a notebook by extracting embedded JSON from the first raw cell's source.
    Handles malformed JSON by reconstructing it properly from the source array.
    """
    text = nb_path.read_text(encoding="utf-8")
    obj = json.loads(text)
    first = obj["cells"][0]
    source_array = first.get("source", [])

    # The source_array contains strings that form a JSON object when joined
    # But since they're stored as JSON strings, quotes inside them are escaped
    # We need to reconstruct the actual JSON by treating each element as a real string

    # Approach 1: Try direct joining and parsing
    joined = "".join(source_array).strip()
    parsed = None
    approach_used = None

    try:
        parsed = json.loads(joined)
        approach_used = "direct"
    except json.JSONDecodeError:
        pass

    # Approach 2: If direct parse failed, the JSON might have unescaped quotes
    # inside string values. We need to reconstruct it.
    # The source_array elements are already JSON-valid Python strings.
    # The problem is they may contain unescaped quotes in f-strings, etc.
    # Try encoding/decoding to handle escape sequences
    if parsed is None:
        try:
            # In case of double escaping or other issues, try various codecs
            decoded = joined.encode("utf-8").decode("unicode_escape")
            parsed = json.loads(decoded)
            approach_used = "unicode_escape"
        except Exception:
            pass

    if parsed is None:
        try:
            decoded = joined.encode("utf-8").decode("raw_unicode_escape")
            parsed = json.loads(decoded)
            approach_used = "raw_unicode_escape"
        except Exception:
            pass

    # Approach 3: If it still fails, try to parse and reconstruct the JSON
    # by properly escaping the strings
    if parsed is None:
        try:
            # Re-escape the problematic quotes
            # Look for patterns like {best["key"]} and escape the inner quotes
            import re

            escaped = joined
            # Find all f-string dict accesses and escape the quotes
            # Pattern: {something["..."] inside the string
            escaped = re.sub(
                r'\{([^}]*)\["([^"]*?)"\]',
                lambda m: "{" + m.group(1) + '[\\"' + m.group(2) + '\\"]',
                escaped,
            )
            parsed = json.loads(escaped)
            approach_used = "quote_escaping"
        except Exception:
            pass

    # If we still couldn't parse, return False
    if parsed is None:
        print(f"Failed to parse embedded JSON in: {nb_path}")
        return False

    if dry_run:
        print(f"Would repair: {nb_path} (using {approach_used})")
        return True

    bak = nb_path.with_suffix(nb_path.suffix + ".bak")
    print(f"Repairing: {nb_path} -> backup {bak} (approach: {approach_used})")
    nb_path.replace(bak)
    nb_path.write_text(
        json.dumps(parsed, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    nbs = list(ROOT.glob(NB_GLOB))
    print(f"Found {len(nbs)} files matching {NB_GLOB}")

    repaired = 0
    for nb in nbs:
        if args.verbose:
            print(f"Checking: {nb}")
        if is_embedded_json(nb):
            if args.verbose:
                print("  -> Detected as embedded JSON")
            if repair_file(nb, dry_run=args.dry_run):
                repaired += 1
        elif args.verbose:
            print("  -> Not detected as embedded JSON")

    print(f"Done. Repaired: {repaired} files.")


if __name__ == "__main__":
    main()
