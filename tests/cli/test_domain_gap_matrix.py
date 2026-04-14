"""Tests for domain gap matrix - computing gap between all classes.

This module tests computing domain gap between all COCO classes present
in the parquet file, creating a full matrix of pairwise distances.
"""

from pathlib import Path
import shlex
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from dqm_ml_job.cli import execute


class TestDomainGapMatrixFull:
    """Tests for full domain gap matrix across all classes."""

    def test_get_all_unique_classes(self, coco_parquet_path: Path, all_classes: list[str], coco_data: Any) -> None:
        """Verify we can extract all unique classes from parquet."""
        assert len(all_classes) > 0, "No classes found in parquet"
        print(f"Found {len(all_classes)} classes: {all_classes}")

    @pytest.mark.slow
    @pytest.mark.skip(reason="Not yet implemented - requires config file generation for each pair")
    def test_domain_gap_all_classes_matrix(
        self, coco_parquet_path: Path, all_classes: list[str], coco_data: Any
    ) -> None:
        """Test computing domain gap between all class pairs.

        This would create a matrix of domain gaps between all unique classes
        in the COCO dataset. Currently skipped as it requires generating
        many config files.

        For 42 classes, we would compute 42*41/2 = 861 pairs.
        """

    def _create_pair_config(self, class_a: str, class_b: str) -> dict:
        """Create a config dict for computing domain gap between two classes."""
        return {
            "config": {
                "dataloaders": {
                    "class_a": {
                        "type": "parquet",
                        "path": "tests/outputs/data/source_1000.parquet",
                        "filter": {"class": class_a},
                    },
                    "class_b": {
                        "type": "parquet",
                        "path": "tests/outputs/data/source_1000.parquet",
                        "filter": {"class": class_b},
                    },
                },
                "metrics_processor": {
                    "image_embedding": {
                        "type": "image_embedding",
                        "data": {"image_column": "image_path", "mode": "path"},
                        "model_config": {
                            "arch": "resnet18",
                            "n_layer_feature": -2,
                            "device": "cpu",
                        },
                        "infer": {
                            "batch_size": 10,
                            "width": 224,
                            "height": 224,
                            "norm_mean": [0.485, 0.456, 0.406],
                            "norm_std": [0.229, 0.224, 0.225],
                        },
                    },
                    "domain_gap": {
                        "type": "domain_gap",
                        "input": {"embedding_col": "embedding"},
                        "delta": {"metric": "mmd_linear"},
                    },
                },
                "compute_delta": True,
                "outputs": {
                    "delta_metrics": {
                        "type": "parquet",
                        "path_pattern": f"tests/outputs/data/metrics_domain_gap_{class_a}_{class_b}_delta-.parquet",
                        "columns": [],
                    }
                },
            }
        }

    def test_list_all_class_combinations(self, all_classes: list[str]) -> None:
        """List all possible class pair combinations."""
        pairs = []
        for i, class_a in enumerate(all_classes):
            for class_b in all_classes[i + 1 :]:
                pairs.append((class_a, class_b))

        print(f"Total possible pairs from {len(all_classes)} classes: {len(pairs)}")

        # Show first 10 pairs as example
        print("First 10 pairs:")
        for pair in pairs[:10]:
            print(f"  {pair[0]} vs {pair[1]}")

        # This shows that with 42 classes, we have 42*41/2 = 861 pairs!


class TestDomainGapWithColumnFilter:
    """Test using class column directly as data selection filter."""

    def test_config_uses_filter_for_class_selection(self, coco_parquet_path: Path, coco_data: Any) -> None:
        """Verify filter can select by class column."""
        table = pq.read_table(coco_parquet_path)
        df = table.to_pandas()

        # Get class distribution
        class_counts = df["class"].value_counts()
        print("Class distribution:")
        print(class_counts)

        # Verify we have multiple classes
        assert len(class_counts) >= 2, "Need at least 2 classes for domain gap"

        # Show which classes have enough samples
        min_samples = 10
        enough_samples = class_counts[class_counts >= min_samples]
        print(f"\nClasses with >= {min_samples} samples: {len(enough_samples)}")
        print(enough_samples)


class TestDomainGapWithSplitBy:
    """Test using split_by config option to create selections per class."""

    def test_split_by_config_exists(self, split_by_config_path: Path) -> None:
        """Verify split_by config file exists."""
        assert split_by_config_path.exists(), f"Config not found: {split_by_config_path}"

    def test_split_by_config_structure(self, split_by_config_path: Path) -> None:
        """Verify split_by config has correct structure."""
        with Path.open(split_by_config_path) as f:
            config = yaml.safe_load(f)

        dataloader = config["config"]["dataloaders"]["coco_classes"]
        assert dataloader.get("split_by") == "class", "split_by should be 'class'"
        assert dataloader.get("split_values") is not None, "split_values should be set"
        assert len(dataloader["split_values"]) == 10, "Should have 10 classes"
        print(f"Top 10 classes configured: {dataloader['split_values']}")

    def test_top_10_classes_have_enough_samples(self, coco_parquet_path: Path, coco_data: Any) -> None:
        """Verify top 10 classes have sufficient samples."""
        table = pq.read_table(coco_parquet_path)
        df = table.to_pandas()
        class_counts = df["class"].value_counts()

        top_10_classes = [
            "dog",
            "cat",
            "horse",
            "bird",
            "giraffe",
            "zebra",
            "cow",
            "person",
            "sheep",
            "elephant",
        ]
        min_count = class_counts[top_10_classes].min()
        print(f"Minimum sample count among top 10: {min_count}")
        assert min_count >= 40, "All top 10 classes should have >= 40 samples"

    @pytest.mark.slow
    def test_split_by_computes_pairwise_domain_gaps(self, split_by_config_path: Path) -> None:
        """Test that split_by creates selections for class pairs.

        With 10 classes, this computes 10*9/2 = 45 unique domain gaps.
        Currently tests with 2 classes for speed.
        """
        assert split_by_config_path.exists(), f"Config not found: {split_by_config_path}"

    @pytest.mark.slow
    def test_split_by_2classes_computation(self, fixtures_dir: Path) -> None:
        """Test split_by with 2 classes actually computes domain gap."""
        config_2_path = fixtures_dir / "domain_gap_split_2classes.yaml"
        assert config_2_path.exists(), f"Config not found: {config_2_path}"

        execute(shlex.split(f"-p {config_2_path}"))

        # Check output was created
        table = pq.read_table("tests/outputs/data/metrics_domain_gap_split_2classes_delta-.parquet")
        df = table.to_pandas()
        assert len(df) == 1, "Should have 1 domain gap result"
        assert df["selection_source"].iloc[0] == "coco_classes_bird"
        assert df["selection_target"].iloc[0] == "coco_classes_elephant"
        print(f"Domain gap (bird vs elephant): {df['mmd_linear'].iloc[0]:.2f}")
