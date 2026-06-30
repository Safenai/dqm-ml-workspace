# Tutorials and Examples

## Fictive Scenario

You are assembling a 4-class animal image dataset (elephant, lion, giraffe, zebra) from three acquisition environments:

- **Zoo** — enclosure photos with bars, fences, and artificial backgrounds. Closer shots at consistent distances. Zoo keepers track quality, so null rates are low.
- **Safari** — drive-through safari park. Vehicle-based photos in open, managed landscapes. Overhead sunlight and vehicle movement produce higher variability and motion blur.
- **Reserve** — game reserve / conservancy. Tourist photos in fully wild conditions. Long distances, heat haze, camera shake, and variable light create the least consistent quality and the highest null rate.

Each environment has a distinct visual profile that the pipeline measures and compares. Each source contributes metadata (quality scores, class labels, source tags) and image data. You need to:

1. [**Completeness**](scenario/completeness.md) — verify every sample has non-null quality scores
2. [**Visual Features**](scenario/visual_features.md) — extract per-image luminosity, contrast, blur, entropy
3. [**Embeddings**](scenario/embeddings.md) — compute deep feature vectors for distribution analysis
4. [**Diversity**](scenario/diversity.md) — check class balance and category spread across sources
5. [**Representativeness**](scenario/representativeness.md) — validate that visual features follow expected distributions
6. [**Domain Gap (Split)**](scenario/domain_gap_split.md) — measure distribution shift between zoo, safari, and reserve subsets
7. [**Domain Gap (Filter)**](scenario/domain_gap_filter.md) — compare safari vs reserve images directly
8. [**Domain Gap (Split + Filter)**](scenario/domain_gap_split_filter.md) — combine both strategies for per-class analysis
9. [**Full Pipeline**](scenario/full_story.md) — run all metrics end-to-end in a single pipeline

Each example page shows the relevant config with inline explanations and a ready-to-run CLI command.

## Prerequisites

```bash
# Generate the required data files (run once):
python examples/script/generate_data.py
```

This creates two files in `examples/data/`:

| File | Rows | Content |
|------|------|---------|
| `large_test_2m.parquet` | 2,000,000 | Tabular: quality scores, class labels, ~5 % nulls |
| `samples_with_images.parquet` | 1,200 | Small synthetic images (32×32 PNG bytes) + metadata |

## Running a Config

```bash
dqm-ml process -p examples/config/scenario/<config_name>.yaml
```

Or run the full pipeline:

```bash
dqm-ml process -p examples/config/scenario/full_story.yaml
```
