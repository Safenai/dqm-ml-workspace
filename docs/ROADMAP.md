# Roadmap & Limitations

This page documents where DQM-ML is headed and current limitations. We believe in transparency about what works well and what needs improvement.

## Current Limitations

V2 represents a major architectural improvement, but it's still evolving. Here's what you should know:


### What's Working Great ✅

- Streaming architecture handles large datasets efficiently
- Core metrics (Completeness, Representativeness) are solid
- Plugin system makes adding new metrics straightforward
- Memory usage stays constant regardless of dataset size

### Known Limitations ⚠️

| Area | Current State | Notes |
|------|---------------|-------|
| **Beta packages** | `dqm-ml-job`, `dqm-ml-images` | Config schemas may change based on feedback |
| **V1 metrics** | Some not yet ported | Gini-Simpson, Simpson indices, Relative Diversity, PAD, CMD, Diversity (pending scientific discussion) |
| **Result variations** | Minor differences in FID/KLMVN | Being investigated for mathematical equivalence |
| **Single-column focus** | Most metrics work per-column | Multi-dimensional feature support coming |
| ~~Legacy dependency~~ | ~~Submodule still present~~ | ~~Legacy `dqm-ml` for comparison (to be phased out)~~ |

For more information see * **[Why a dqm-ml V2](./dqm-ml-v2.md)**: The "why" and "how" of V2.

> 📝 **Your feedback matters!** If you encounter issues or have suggestions, please [open an issue](https://github.com/Safenai/dqm-ml-workspace/issues).

## Roadmap

Here's our vision for DQM-ML, organized into phases:

### Phase 0: Complete V2.0.0-rc (Now - open for comment dqm-ml v2)

Usable version of dqm-ml v2 open for comment before official release.
- [x] **Standalone release** - Finalize V2.0.0 as a proper package


### Phase 1: Complete V2.0.0 (Near term - Q2 T2026)

What's coming in the next release:
- [ ] **Configuration consitency** - make configuration metrics consistent, and check configuration validity
- [ ] **Comminuty feedback** - implement user feedback quick correction and upgrade roadmap with others
- [ ] **Feature parity** - Port remaining V1 metrics to V2 API
- [ ] **API freeze** - Lock down `dqm-ml-core` for stability

### Phase 2: New Domains 

Expanding what DQM-ML can analyze:

- [ ] **Time series** - New package for sequential data quality
- [ ] **Multi-modal** - Support for text + image datasets
- [ ] **SQL integration** - Compute metrics directly via DuckDB

### Phase 3: Performance & Scale 

Improving for larger workloads:

- [ ] **Advanced streaming** - Disk-backed accumulators for very large datasets
- [ ] **Parallelization** - Multi-core processing for image features and deep learning metrics
- [ ] **Database support** - Read directly from databases, not just files



## How We Prioritize

We decide what to build next based on:

1. **Community needs** - Issues and discussions from users
2. **Technical feasibility** - What's achievable with current architecture
3. **Resource availability** - Who can help build it

Want to influence the roadmap? Here's how:

- 🐛 [Report bugs](https://github.com/Safenai/dqm-ml-workspace/issues) - Help us prioritize fixes
- 💡 [Suggest features](https://github.com/Safenai/dqm-ml-workspace/discussions) - Share your use case
- 👩‍💻 [Contribute](contributing.md) - Help build V2.0.0

## Priorities for Contributors

Looking to contribute? Here's what needs help most:

### High Priority

- **Porting V1 metrics**: Gini-Simpson, Simpson indices, Relative Diversity, PAD, CMD
- **Test coverage**: More edge cases and integration tests

### Medium Priority

- **Documentation**: Examples, use cases, better explanations
- **Performance optimizations**: Batch processing improvements

### Lower Priority

- **New features**: Time series support, database connectors
- **Tooling**: IDE integrations, IDE plugins

### How to Start

1. Check [open issues](https://github.com/Safenai/dqm-ml-workspace/issues) tagged `good first issue`
2. Read the [contributing guide](contributing.md) for setup instructions
3. Join discussions to propose new features

## Version History

| Version | Release Date | Highlights |
|---------|-------------|------------|
| 2.0.0-rc | 2026 | V2 architecture, streaming, plugins, `dqm-ml` CLI (renamed from dqm-ml-v2) |
| 1.1.x | Q1 2026 | V2 release candidate series |
| 1.0.x | Earlier | Original library (V1) |


