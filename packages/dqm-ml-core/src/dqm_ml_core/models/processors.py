"""Processor configuration models for DQM-ML pipelines.

Defines configuration classes for all supported processor types:
- Image feature extraction (luminosity, contrast, blur, entropy)
- Neural network embedding extraction
- Completeness metrics
- Representativeness evaluation (chi-square, GRTE, KS, Shannon entropy)
- Diversity metrics (Simpson, Gini, Shannon, richness)
- Domain gap measurement (MMD, discriminative, etc.)

Also includes supporting configuration for models, inference, kernels,
distance metrics, and summary statistics.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dqm_ml_core.models.columns import ColumnsConfig

_LUMINOSITY_STANDARDS: dict[str, tuple[float, float, float]] = {
    "bt601": (0.299, 0.587, 0.114),
    "bt709": (0.2126, 0.7152, 0.0722),
    "bt2020": (0.2627, 0.6780, 0.0593),
}


class _ProcessorBase(BaseModel):
    """Base processor configuration with common fields.

    Attributes:
        name: Unique processor name.
        type: Processor type discriminator for serialization.
        columns: Column input/output configuration.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique processor name.")
    type: str = Field(description="Processor type discriminator.")
    columns: ColumnsConfig | None = None


class HistogramConfig(BaseModel):
    """Histogram parameters for feature extraction.

    Attributes:
        bins: Number of histogram bins (must be positive).
    """

    model_config = ConfigDict(extra="forbid")

    bins: int = Field(default=256, gt=0, description="Number of histogram bins.")


class ImageFeaturesProcessorConfig(_ProcessorBase):
    """Configuration for low-level image feature extraction.

    Extracts luminosity, contrast, blur, and entropy features from images.

    Attributes:
        type: Processor type discriminator ("image_features").
        features: List of image features to compute.
        batch_size: Batch size for image processing.
        grayscale: Whether to convert images to grayscale.
        normalize: Whether to normalize pixel values to [0, 1].
        laplacian_kernel: Laplacian kernel size for blur detection ("3x3" or "5x5").
        clip_percentiles: Percentile clipping for extreme pixel values, e.g. (1, 99).
        histogram: Histogram configuration for feature computation.
        luminosity_weights: Luminosity weights for grayscale conversion.
            Standard name ('bt601', 'bt709', 'bt2020') or [R, G, B] list/tuple.
            Defaults to BT.709 when None.
    """

    type: Literal["image_features"] = "image_features"
    features: list[str] = Field(
        default=["luminosity", "contrast", "blur", "entropy"],
        description="List of image features to compute.",
    )
    batch_size: int = Field(default=64, gt=0, description="Batch size for image processing.")
    grayscale: bool = Field(default=True, description="Convert images to grayscale.")
    normalize: bool = Field(default=True, description="Normalise pixel values to [0, 1].")
    laplacian_kernel: str = Field(default="3x3", description="Laplacian kernel size for blur detection.")
    clip_percentiles: tuple[int, int] | None = Field(
        default=None,
        description="Percentile clipping for extreme pixel values, e.g. (1, 99).",
    )
    histogram: HistogramConfig | None = None
    luminosity_weights: str | tuple[float, float, float] | None = Field(
        default=None,
        description="Luminosity weights for grayscale conversion. "
        "Standard name ('bt601', 'bt709', 'bt2020') or [R, G, B] list. "
        "Defaults to BT.709 when None.",
    )

    @field_validator("luminosity_weights", mode="before")
    @classmethod
    def _normalize_luminosity_weights(cls, v: Any) -> Any:
        """Normalize luminosity weights input to standard key or tuple.

        Args:
            v: Input value - None, standard name string, or [R, G, B] list/tuple.

        Returns:
            Normalized value: None, standard key (e.g. "bt709"), or tuple of 3 floats.

        Raises:
            ValueError: If input is not a recognized standard, list/tuple of length 3, or None.
        """
        if v is None:
            return v
        if isinstance(v, str):
            key = v.lower().replace(".", "")
            if key not in _LUMINOSITY_STANDARDS:
                raise ValueError(f"Unknown luminosity standard '{v}'. Use one of {list(_LUMINOSITY_STANDARDS)}.")
            return key
        if isinstance(v, (list, tuple)):
            if len(v) != 3:
                raise ValueError(f"luminosity_weights must have exactly 3 elements, got {len(v)}.")
            return tuple(v)
        raise ValueError(f"luminosity_weights must be a standard name, [R,G,B] list, or None, got {type(v).__name__}.")


class ModelConfig(BaseModel):
    """Neural network model configuration for embedding extraction.

    Attributes:
        arch: Model architecture name (e.g., "resnet18", "resnet50").
        n_layer_feature: Layer index (negative for reverse) or list of layer names
            for feature extraction. Default -2 (second to last layer).
        device: Device for model inference ("auto", "cpu", "cuda").
    """

    model_config = ConfigDict(extra="forbid")

    arch: str = Field(default="resnet18", description="Model architecture name.")
    n_layer_feature: int | list[str] = Field(
        default=-2,
        description="Layer index or list of layer names for feature extraction.",
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Device for model inference.",
    )


class InferConfig(BaseModel):
    """Inference pre-processing settings for image embeddings.

    Attributes:
        batch_size: Inference batch size.
        width: Resize width for input images.
        height: Resize height for input images.
        norm_mean: Per-channel mean used for normalisation (ImageNet defaults).
        norm_std: Per-channel std used for normalisation (ImageNet defaults).
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=32, gt=0, description="Inference batch size.")
    width: int = Field(default=224, gt=0, description="Resize width for input images.")
    height: int = Field(default=224, gt=0, description="Resize height for input images.")
    norm_mean: list[float] = Field(
        default=[0.485, 0.456, 0.406],
        description="Per-channel mean used for normalisation.",
    )
    norm_std: list[float] = Field(
        default=[0.229, 0.224, 0.225],
        description="Per-channel std used for normalisation.",
    )


class FeaturesEmbeddingsProcessorConfig(_ProcessorBase):
    """Configuration for neural-network embedding feature extraction.

    Extracts deep learning embeddings from images using a configured model.

    Attributes:
        type: Processor type discriminator ("features_embeddings").
        model: Neural network model configuration.
        infer: Inference pre-processing settings.
    """

    type: Literal["features_embeddings"] = "features_embeddings"
    model: ModelConfig = Field(default_factory=ModelConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    # luminosity_weights: TODO


class CompletenessProcessorConfig(_ProcessorBase):
    """Configuration for completeness metric computation.

    Computes per-column and overall completeness (non-null) metrics.

    Attributes:
        type: Processor type discriminator ("completeness").
        include_per_column: Include per-column completeness scores in output.
        include_overall: Include overall completeness score in output.
        include_metadata: Include metadata (total rows, null counts) in output.
    """

    type: Literal["completeness"] = "completeness"
    include_per_column: bool = Field(default=True, description="Include per-column completeness scores.")
    include_overall: bool = Field(default=True, description="Include overall completeness score.")
    include_metadata: bool = Field(default=False, description="Include metadata in output.")


class InterpretationConfig(BaseModel):
    """Human-readable labels for representativeness and diversity results.

    Attributes:
        follows_distribution: Label when data follows the expected distribution.
        does_not_follow_distribution: Label when data diverges from expected distribution.
        high_diversity: Label for high diversity results.
        low_diversity: Label for low diversity results.
        high_representativeness: Label for high representativeness results.
        low_representativeness: Label for low representativeness results.
    """

    model_config = ConfigDict(extra="forbid")

    follows_distribution: str = Field(
        default="fits target",
        description="Label when data follows the distribution.",
    )
    does_not_follow_distribution: str = Field(default="diverges from target", description="Label when data diverges.")
    high_diversity: str = Field(default="varied", description="Label for high diversity.")
    low_diversity: str = Field(default="uniform", description="Label for low diversity.")
    high_representativeness: str = Field(
        default="representative",
        description="Label for high representativeness.",
    )
    low_representativeness: str = Field(
        default="under-represented",
        description="Label for low representativeness.",
    )


class HistogramsConfig(BaseModel):
    """Histogram parameters for representativeness evaluation.

    Attributes:
        bins: Number of histogram bins (must be positive).
    """

    model_config = ConfigDict(extra="forbid")

    bins: int = Field(default=10, gt=0, description="Number of histogram bins.")


class ShannonConfig(BaseModel):
    """Shannon-entropy threshold configuration for representativeness.

    Attributes:
        threshold: Entropy threshold for flagging under-represented values.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=2.0, description="Entropy threshold for flagging.")


class GrteConfig(BaseModel):
    """Gini-ratio-threshold-entropy (GRTE) configuration for representativeness.

    Attributes:
        threshold: Gini ratio threshold for flagging.
        scaling_factor: Scaling applied to the Gini ratio (negative inverts).
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.5, description="Gini ratio threshold.")
    scaling_factor: float = Field(default=-2.0, description="Scaling applied to the Gini ratio.")


class KsConfig(BaseModel):
    """Kolmogorov-Smirnov test configuration for representativeness.

    Attributes:
        sample_size: Number of samples for KS testing.
        min_sample_size: Minimum samples required for KS test.
        sample_divisor: Divisor for automatic sample-size calculation.
    """

    model_config = ConfigDict(extra="forbid")

    sample_size: int = Field(default=500, gt=0, description="Number of samples for KS testing.")
    min_sample_size: int = Field(default=50, gt=0, description="Minimum samples required for KS test.")
    sample_divisor: int = Field(
        default=20,
        gt=0,
        description="Divisor for automatic sample-size calculation.",
    )


class ColumnDistributionParams(BaseModel):
    """Per-column distribution parameters for user_provided mean_std_estimation."""

    column: str
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None


class RepresentativenessProcessorConfig(_ProcessorBase):
    """Configuration for representativeness evaluation against a reference distribution.

    Evaluates how well a dataset represents a target distribution using
    multiple statistical metrics.

    Attributes:
        type: Processor type discriminator ("representativeness").
        metrics: List of representativeness metrics to compute.
        alpha: Significance level for statistical tests.
        epsilon: Small constant to avoid division by zero.
        distribution: Expected reference distribution ("normal" or "uniform").
        interpretation: Human-readable labels for results.
        histogram: Histogram configuration for evaluation.
        shannon: Shannon entropy threshold configuration.
        grte: GRTE configuration.
        ks: Kolmogorov-Smirnov test configuration.
    """

    type: Literal["representativeness"] = "representativeness"
    metrics: list[str] = Field(
        default=["chi-square", "grte", "kolmogorov-smirnov", "shannon-entropy"],
        description="List of representativeness metrics to compute.",
    )
    alpha: float = Field(default=0.05, description="Significance level for statistical tests.")
    epsilon: float = Field(
        default=1e-9,
        gt=0,
        description="Small constant to avoid division by zero.",
    )
    distribution: Literal["normal", "uniform"] = Field(
        default="normal",
        description="Expected reference distribution.",
    )
    mean_std_estimation: Literal["from_first_batch", "per_batch", "from_all_data", "user_provided"] = Field(
        default="from_first_batch",
        description=(
            "How distribution parameters (mean/std for normal, min/max for uniform) are estimated. "
            "'from_first_batch' — estimate from the first batch and reuse (consistent with bin edges). "
            "'per_batch' — re-estimate on each batch (use with high-variance data, risks insufficient_bins). "
            "'user_provided' — use explicit parameters from distribution_params. "
            "'from_all_data' — estimate from full dataset (not yet implemented)."
        ),
    )
    expected_counts_method: Literal["cdf", "monte_carlo"] = Field(
        default="cdf",
        description=(
            "Method for computing expected bin counts. "
            "'cdf' — exact expected counts via CDF (deterministic). "
            "'monte_carlo' — Monte Carlo sampling via RNG (stochastic)."
        ),
    )
    distribution_params: list[ColumnDistributionParams] | None = Field(
        default=None,
        description=(
            "Per-column explicit distribution parameters for 'user_provided' strategy. "
            "Example: [{'column': 'col1', 'mean': 0.0, 'std': 1.0}]"
        ),
    )
    interpretation: InterpretationConfig | None = None
    histogram: HistogramsConfig | None = None
    shannon: ShannonConfig | None = None
    grte: GrteConfig | None = None
    ks: KsConfig | None = None


class DiversityProcessorConfig(_ProcessorBase):
    """Configuration for diversity metric computation.

    Computes diversity metrics: Simpson, Gini, Shannon entropy, and richness.

    Attributes:
        type: Processor type discriminator ("diversity").
        metrics: List of diversity metrics to compute.
    """

    type: Literal["diversity"] = "diversity"
    metrics: list[str] = Field(
        default=["simpson", "gini", "shannon", "richness"],
        description="List of diversity metrics to compute.",
    )


class KernelParamsRbf(BaseModel):
    """RBF (Radial Basis Function) kernel parameters for distance metrics.

    Attributes:
        gamma: RBF kernel gamma parameter (inverse kernel width).
    """

    model_config = ConfigDict(extra="forbid")

    gamma: float = Field(default=1.0, description="RBF kernel gamma parameter.")


class KernelParamsPoly(BaseModel):
    """Polynomial kernel parameters for distance metrics.

    Attributes:
        degree: Polynomial degree.
        gamma: Polynomial kernel gamma parameter.
        coefficient0: Polynomial kernel coefficient offset (constant term).
    """

    model_config = ConfigDict(extra="forbid")

    degree: float = Field(default=3.0, description="Polynomial degree.")
    gamma: float = Field(default=1.0, description="Polynomial kernel gamma.")
    coefficient0: float = Field(default=1.0, description="Polynomial kernel coefficient offset.")


class DistanceConfig(BaseModel):
    """Distance metric configuration for domain-gap computation.

    Attributes:
        metric: Distance metric name (e.g., "mmd", "discriminative", "klmvn_diag").
        evaluator: Optional evaluator type for discriminative metrics.
        k: Number of nearest neighbours (for k-NN based metrics).
        feature_weights: Per-feature weights for weighted distance computation.
        kernel_params: Kernel parameters (RBF or Polynomial) for kernel-based metrics.
        epsilon: Regularization epsilon for numerical stability of covariance-based metrics.
        klmvn_var_eps: Variance regularization for KL divergence numerical stability.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(description="Distance metric name (e.g. 'mmd', 'discriminative').")
    evaluator: str | None = Field(default=None, description="Optional evaluator type.")
    k: int | None = Field(
        default=None,
        description="Number of nearest neighbours (if applicable).",
    )
    feature_weights: list[float] | None = Field(
        default=None,
        description="Per-feature weights for weighted distance computation.",
    )
    kernel_params: KernelParamsRbf | KernelParamsPoly | None = None
    epsilon: float = Field(
        default=1e-6,
        ge=0,
        description="Regularization epsilon for numerical stability of covariance-based metrics (e.g. FID).",
    )
    klmvn_var_eps: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Variance regularization for klmvn_diag numerical stability "
            "(default 0.0). When > 0, source and target variances are "
            "replaced by var + klmvn_var_eps * mean(var) before computing "
            "KL divergence, preventing blow-up from near-zero variance "
            "dimensions."
        ),
    )


class HistogramSummaryConfig(BaseModel):
    """Histogram settings for embedding summary statistics.

    Attributes:
        dims: Number of dimensions to histogram.
        bins: Number of bins per dimension (must be positive).
        range: Histogram range [min, max] for each dimension.
    """

    model_config = ConfigDict(extra="forbid")

    dims: int = Field(default=64, gt=0, description="Number of dimensions to histogram.")
    bins: int = Field(default=32, gt=0, description="Number of bins per dimension.")
    range: list[float] = Field(default=[-3.0, 3.0], description="Histogram range [min, max].")


class SummaryConfig(BaseModel):
    """Embedding summary configuration for domain-gap computation.

    Attributes:
        collect_sum_outer: Collect sum-of-outer-products for covariance estimation.
        store_embeddings: Store full embedding vectors in output.
        histogram: Histogram configuration for embedding summaries.
    """

    model_config = ConfigDict(extra="forbid")

    collect_sum_outer: bool | None = Field(
        default=None,
        description="Collect sum-of-outer-products for covariance estimation.",
    )
    store_embeddings: bool | None = Field(default=None, description="Store full embedding vectors.")
    histogram: HistogramSummaryConfig | None = None


class DomainGapProcessorConfig(_ProcessorBase):
    """Configuration for domain-gap (distribution shift) measurement.

    Measures distribution shift between datasets using a configured distance metric.

    Attributes:
        type: Processor type discriminator ("domain_gap").
        columns: Column input configuration (required).
        distance: Distance metric configuration.
        summary: Embedding summary configuration.
    """

    type: Literal["domain_gap"] = "domain_gap"
    columns: ColumnsConfig = Field(description="Column input configuration (required).")
    distance: DistanceConfig = Field(description="Distance metric configuration.")
    summary: SummaryConfig | None = None

    @model_validator(mode="after")
    def _validate_domain_gap(self) -> "DomainGapProcessorConfig":
        """Validate domain-gap configuration constraints."""
        if self.columns and not self.columns.input:
            raise ValueError("'columns.input' is required and must not be empty for domain_gap processors")
        if self.distance and self.distance.feature_weights is not None and self.columns and self.columns.input:
            n_cols = len(self.columns.input)
            if len(self.distance.feature_weights) != n_cols:
                raise ValueError(
                    f"feature_weights length ({len(self.distance.feature_weights)}) "
                    f"must match columns.input length ({n_cols})"
                )
        return self


ProcessorConfig = Annotated[
    ImageFeaturesProcessorConfig
    | FeaturesEmbeddingsProcessorConfig
    | CompletenessProcessorConfig
    | RepresentativenessProcessorConfig
    | DiversityProcessorConfig
    | DomainGapProcessorConfig,
    Field(discriminator="type"),
]
