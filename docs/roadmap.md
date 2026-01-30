# Known limitation and roadmap

## Known limitation

In this version we focus on code restructuration, deplendency management, proposition of a unified interface for all metrics computation as well as introduction of ability to compute sample feature on which metrics are computed.

We identify during refactoring several limitation for an industrial usage that might be adressed in futur version, but can not be implemented in this version as we what to remain as close as possible from original implementation in order to be able to compare results, see functional limitation identified

### known issues

- not all V1 metrics are re-implemented with the following API, the following metrics are missing 
  - diversity
  - domain gap
    - PAD
    - CMD

- Several variation between V1 and V2 need more investigations
  - KLMVN and FID Domain gap metrics provide different results between V1 and V2, see FT #4

- dqm-ml-pipeline and dqm-ml-images are in beta version, and still subject to change before V2, configuration structure, and command line parameter are still subject of change. But except bug / feedback from community API to derived in order to implement new metrics based on dqm-ml-core might not change, and will be frozen in V2.0.0 to enable community to propose new metrics

### functional limitations

- Scientific discussion need to be perform to adapt target content of diversity metric to allow implementation with the new API, as well as the position of entropy computation

- Extension of proposed metric to compute representativity of a set of feature, not only one column
  - need to exetend computation on other type of data
  - need scientific contribution to define target and reference scientific communication associated.

## ROADMAP

### Finalize dqm-ml upgrade to V2.0.0

- doc integration in this repository
- mark original repository as deprecated and reference this one
- integrate the last metrics not implemented
- put version as V2.0.0

### Improve identified limit in current metrics implementation

- Rationalize level of parametrization between metrics,
- request with scientific commity an improvement in allocation of method between
  - diversity (specific to class problems ?, clarity outputs)
  - representativness
    - use entropy for diversity
    - extend current distribution propose to other kind of expected real life distribution
    - use based metrics to improve precision of batch implementation (min,max,mean,std)

- Improve performance of image feature computation
- Integrate feedback from community, and architectural concept document realize in Q4 2025

### Extend the proposed content of the library

- extend dqm-ml-workspace with a package dedicated to time series features
- Propose a performance improvement, and document how to configure dqml usage

- Support of other format as inputs and outputs than parquet
  - write output in json / yaml format for metrics
  - read from csv
  - read / write to/from database

- Create a documentation "How to create a new metric"

- extend in pipeline the experimental usage of duckdb to allow computation of metric directly on groups in the input without the need of creating several configuratons / parquet files.

- New metrics, and improvement of current metrics configurations, base on next month experimentations.
