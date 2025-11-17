"""Minimal KataGo analysis engine client stubs.

These are lightweight helpers and docstrings to get development started.
They intentionally do not run KataGo here; instead they provide small
call signatures and parsing helpers to be expanded when integrating with
the actual KataGo binary or the JSON analysis engine.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_katago_analysis_cmd(katago_binary: str, model_path: str, config_path: str, extra_args: Optional[List[str]] = None) -> List[str]:
    """Build a command line for running the KataGo JSON analysis engine.

    This returns the command as a list suitable for subprocess calls. It
    does not execute the command.
    """
    cmd = [katago_binary, "analysis", "-model", model_path, "-config", config_path]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


class AnalysisClient:
    """A tiny stub client to outline interactions with the analysis engine.

    Methods here should be expanded to actually spawn processes and stream
    JSON messages to/from KataGo. For now they return simple example data
    structures so downstream code can be developed and unit-tested.
    """

    def __init__(self, katago_binary: str = "katago", model: str = "default_model.bin.gz", config: str = "analysis.cfg"):
        self.katago_binary = katago_binary
        self.model = model
        self.config = config

    def analyze_once(self, sgf_position: Optional[str] = None) -> Dict[str, Any]:
        """Run a single analysis and return parsed JSON-like dict.

        This is a stub. The real implementation should invoke the command
        produced by `build_katago_analysis_cmd` and parse the JSON output.
        """
        # Example of a parsed JSON structure that KataGo might return
        example = {
            "root": {
                "policy": [0.05, 0.9, 0.05],
                "value": 0.12,
                "entropy": 1.02,
            }
        }
        return example

    @staticmethod
    def parse_analysis_json(raw: str) -> Dict[str, Any]:
        """Parse a raw JSON string from KataGo into a Python dict.

        Raises ValueError if parsing fails.
        """
        return json.loads(raw)


__all__ = ["build_katago_analysis_cmd", "AnalysisClient"]
