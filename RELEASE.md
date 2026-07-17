# Release Notes

## v2.0.0 (2026-07-17)

Major release introducing a new processor architecture with three specialized
interfaces (FeaturesProcessor, MetricsProcessor, GapProcessor), Pydantic-based
configuration, a processor DAG with topological execution, new domain gap
algorithms, S3 support, and comprehensive documentation.

### Processor Interfaces

- Split monolithic `DatametricProcessor` into three specialized interfaces:
  - `FeaturesProcessor` — feature extraction (visual features, embeddings)
  - `MetricsProcessor` — tabular metrics (completeness, diversity, representativeness)
  - `GapProcessor` — domain gap analysis (FID, MMD-RBF, Wasserstein-1D)
- `ProcessorRunner.run_gap()` accepts raw images directly — no manual embedding
  extraction needed when paired with `ImageEmbeddingProcessor`.

### Configuration

- Pydantic models for processor, output, dataloader, and global configs with
  validation and type safety.
- `StorageConfig` model for S3 with retry strategies, role-based access, and
  env-var fallback.

### Pipeline

- Topological sort ensures generators run before consumers in `DatasetJob.execute()`.
- Accumulate-then-flush mode for single-path output patterns.
- Configurable `compute_max_memory` with memory-threshold parsing.

### Metrics

- **Domain Gap**: MMD-RBF, MMD-Poly, PAD, CMD with multi-layer support.
- **Domain Gap**: KLMVN variance-eps dampening, FID epsilon regularization,
  PAD with `CalibratedClassifierCV`, CMD multi-channel spatial moments.
- **Diversity**: New `DiversityProcessor` with Simpson, Gini-Simpson, Shannon,
  and Richness indices.
- **Representativeness**: Mean-std estimation, expected-counts method,
  per-column distribution parameters, path-prefix for sample paths.
- **Completeness**: Improved validation and error handling.

### Image Processing

- Multi-input-column support for embeddings.
- Lazy model loading with auto device resolution.
- Column prefix/suffix and S3 per-column path prefixes.
- Configurable failure handling with rate thresholds.

### S3 Support

- Shared S3 utility (`dqm_ml_job.utils.s3.get_s3_filesystem`) with env-var
  and dict-based configuration.
- S3 support in Parquet data loading, output writing, image embedding,
  and visual features processors.
- `region` parameter for providers like OVH S3.

### Documentation

- Restructured docs: `docs/configuration/*.md`, metrics documentation,
  formal concepts page.
- New scenario example configs and example notebooks for each metric.
- Updated user guide with installation, metrics, configuration, and usage.
- Google-format docstrings across source packages.

### Testing

- New integration tests: batch invariance, pipeline ordering, path prefix,
  output columns, data flow, full story, features embeddings.
- New unit tests: matching, registry, representativeness, error policy,
  output configs, job, pandas loader, domain gap processor.
- New property-based tests: diversity, domain gap, representativeness,
  visual features.
- Package isolation smoke tests for all 5 packages.
- Configurable test seeds via `DQM_ML_TEST_SEED` env var (default 42).
- tqdm output suppressed in CI environments.
- Synthetic data generated at runtime, replacing LFS-tracked files.

### Python Compatibility

- Python 3.10, 3.11, 3.12, 3.13 supported.

### Breaking Changes

- Package renamed: `dqm-ml-v2` → `dqm-ml`
- CLI command: `dqm-ml-v2` → `dqm-ml`
