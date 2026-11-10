"""Pinned control entry point; safe from any working directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openalex_agent.multihorizon_supervisor import main
if __name__ == "__main__":
    raise SystemExit(main())
