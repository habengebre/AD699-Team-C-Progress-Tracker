
import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

notebook_path = Path("/workspace/input/notebook.ipynb")
report_path = Path("/workspace/output/execution_report.json")

report = {
    "status": "not_run",
    "executed_code_cells": 0,
    "error_count": 0,
    "errors": []
}

try:
    notebook = nbformat.read(notebook_path, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": "/workspace/input"}}
    )
    client.execute()

    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue

        report["executed_code_cells"] += 1

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                report["errors"].append({
                    "cell": index,
                    "type": output.get("ename", "Error"),
                    "message": output.get("evalue", "")
                })

    report["error_count"] = len(report["errors"])
    report["status"] = (
        "passed" if report["error_count"] == 0 else "failed"
    )

except Exception as error:
    report["status"] = "failed"
    report["error_count"] = 1
    report["errors"].append({
        "cell": None,
        "type": type(error).__name__,
        "message": str(error)
    })

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2))

print(json.dumps(report, indent=2))
sys.exit(0)
