#!/usr/bin/env python3
"""Smoke test: dqm-ml-core + dqm-ml-job (metrics via YAML pipeline).

Verifies that metrics can be executed through dqm-ml-job CLI with a YAML config.
"""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def test_metrics_via_yaml():
    from dqm_ml_job.cli import execute

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        test_data = {
            "col_a": [1, 2, None, 4, 5, 6, 7, None, 9, 10],
            "col_b": [1, 2, 3, None, 5, 6, None, 8, 9, 10],
            "feature": np.random.default_rng().normal(0, 1, 10),
        }
        table = pa.table(test_data)
        parquet_path = tmpdir / "test_data.parquet"
        pq.write_table(table, parquet_path)

        yaml_content = f"""\
dataloaders:
  loaders:
    - name: test_data
      type: parquet
      path: {parquet_path}
      batch_size: 5

metrics:
  processors:
    - name: completeness
      type: completeness
      columns:
        input: ["col_a", "col_b"]
      include_per_column: true
      include_overall: true

  outputs:
    path: {tmpdir}/output_metrics.parquet
"""
        yaml_path = tmpdir / "config.yaml"
        yaml_path.write_text(yaml_content)
        execute(["-p", str(yaml_path)])

        output_file = tmpdir / "output_metrics.parquet"
        assert output_file.exists(), f"Output file not found: {output_file}"

        output_df = pq.read_table(output_file).to_pandas()
        assert "completeness_col_a" in output_df.columns, "Missing completeness_col_a in output"
        assert "completeness_col_b" in output_df.columns, "Missing completeness_col_b in output"
        assert "completeness_overall" in output_df.columns, "Missing completeness_overall in output"


if __name__ == "__main__":
    try:
        test_metrics_via_yaml()
        print("dqm-ml-core + dqm-ml-job smoke test PASSED")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
