# Documentation

## Building Docs

Generate documentation in HTML format using mkdocs:

```bash
# Build documentation (outputs to docs/site/)
uv run nox -s docs

# Build for offline use
uv run nox -s docs_offline

# Serve documentation locally (live reload)
uv run nox -s docs_serve
```

The documentation site is automatically built and deployed by the CI/CD pipeline to GitHub Pages.

## Documentation Guidelines

When editing documentation (README, docs/, etc.), follow these guidelines.

### Audience

The audience is the open source community, grouped by role:

#### Technical Users (primary)
- **Data Scientists** — ML practitioners checking dataset quality
- **ML Engineers / MLOps** — Building and monitoring data pipelines
- **Data Engineers** — Building reliable ETL pipelines
- **Software Engineers** — Integrating metrics into applications

#### Researchers
- **Research Scientists** — Academic papers on data quality methodology
- **Academics / Students** — Learning and teaching data quality concepts

#### Decision Makers
- **Tech Leads / Architects** — Deciding on data infrastructure
- **Product Managers** — Defining data quality requirements
- **Startup Founders** — Building AI products

#### Specialized Roles
- **Domain Experts** — Healthcare, finance — validating domain-specific data
- **AI Ethics / Governance** — Checking for bias, ensuring fairness
- **Enterprise Users** — Compliance, governance, audit

#### Community
- **Open Source Contributors** — Integrating metrics into other tools
- **Python Enthusiasts** — Exploring data quality metrics

### Tone and Wording

- **Welcoming and friendly** — Write as if explaining to a colleague
- **Accessible** — Don't assume deep technical knowledge of DQM-ML internals
- **Practical** — Focus on "how to" and "why" before details
- **Inclusive** — Avoid jargon; explain technical terms briefly
- **Respect technical levels** — Some know ML, others know Python, some neither

### Technical Level Guidelines

| Context | Example | Approach |
|---------|---------|----------|
| **Quick Start** | "Install and run in 2 minutes" | Keep simple |
| **API docs** | `CompletenessProcessor` | Explain, show minimal example |
| **Architecture** | Streaming pipeline | Explain "why" before "how" |
| **Configuration** | YAML examples | Copy-paste friendly |

### Best Practices

1. **Lead with the goal** — Tell readers what they'll learn/do
2. **Use concrete examples** — "Run this command"
3. **Link for depth** — README overview, docs/ details
4. **Explain abbreviations** — First use: "Maximum Mean Discrepancy (MMD)"
5. **Keep it scannable** — Tables, bullet points, code blocks
6. **Respect all levels** — Don't assume expertise

### What to Avoid

- **Assuming expertise** — New users may not know uv, Docker, or ML
- **Being patronizing** — Explain once, not repeatedly
- **Over-simplifying** — Respect intelligence
- **Inconsistent terminology** — Same term throughout

### Structure for Markdown Files

- **README.md**: High-level overview, quick start, key references
- **docs/*.md**: Detailed explanations, full examples, background
- **docs/metrics/*.md**: Detailed metric docs (configuration, parameters, usage)
- **docs/metrics.md**: Overview table, links to detailed metric pages
