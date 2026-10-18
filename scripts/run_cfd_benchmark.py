"""Compatibility entry point for the CFD-only complementary evaluation.

The implementation lives in run_integrated_evaluation.py so the academic
notebook, final report and this standalone command use one canonical path.
"""
from __future__ import annotations

import json

from run_integrated_evaluation import CFD_RESULTS_PATH, evaluate_cfd


def main() -> None:
    result = evaluate_cfd(create_figures=True)
    CFD_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFD_RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rows={len(result['rows'])} results={CFD_RESULTS_PATH}")


if __name__ == "__main__":
    main()
