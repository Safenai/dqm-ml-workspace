"""Global configuration models for storage, compute, and error handling.

Defines shared configuration classes used across pipeline interfaces:
- RetryConfig: Retry policies for storage operations
- StorageConfig: S3/local storage with credential management
- ComputeConfig: Runtime settings (device, memory, threads, logging)
- ImageErrorsConfig/TabularErrorsConfig/ErrorsConfig: Error handling policies
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetryConfig(BaseModel):
    """Retry policy for storage operations."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["default", "standard"] = Field(
        default="standard",
        description="Retry mode: 'default' uses exponential backoff, 'standard' uses fixed intervals.",
    )
    max_attempts: int = Field(default=3, gt=0, description="Maximum number of retry attempts.")


class StorageConfig(BaseModel):
    """Remote or local storage configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["s3", "local"] = Field(description="Storage backend type.")
    bucket: str | None = Field(default=None, description="S3 bucket name (required when type='s3').")

    access_key: str | None = Field(default=None, description="AWS access key ID.")
    secret_key: str | None = Field(default=None, description="AWS secret access key.")
    session_token: str | None = Field(default=None, description="AWS session token.")
    anonymous: bool = Field(default=False, description="Use anonymous (unsigned) requests.")
    role_arn: str | None = Field(default=None, description="ARN of IAM role to assume.")
    session_name: str | None = Field(default=None, description="Name for the assumed role session.")
    external_id: str | None = Field(default=None, description="External ID for role assumption.")
    load_frequency: int = Field(default=900, gt=0, description="Frequency (seconds) to refresh credentials / role.")

    region: str | None = Field(default=None, description="AWS region (e.g. 'us-east-1').")
    endpoint: str | None = Field(default=None, description="Custom S3 endpoint URL.")
    request_timeout: float | None = Field(default=None, description="Request timeout in seconds.")
    connect_timeout: float | None = Field(default=None, description="Connection timeout in seconds.")
    scheme: str | None = Field(default=None, description="URI scheme (e.g. 'https').")
    proxy_options: dict[str, Any] | str | None = Field(
        default=None,
        description="Proxy configuration as a dict or URL string.",
    )
    tls_ca_file_path: str | None = Field(default=None, description="Path to a custom TLS CA bundle.")

    retry: RetryConfig | None = None

    checksum_validation: Literal["when_required", "always", "never"] = Field(
        default="when_required",
        description="Controls S3 checksum validation behaviour.",
    )

    @model_validator(mode="after")
    def _require_bucket_for_s3(self) -> "StorageConfig":
        """Validate that S3 storage configuration includes a bucket name.

        Returns:
            The validated StorageConfig instance.

        Raises:
            ValueError: If storage type is 's3' but no bucket is provided.
        """
        if self.type == "s3" and self.bucket is None:
            raise ValueError("StorageConfig with type='s3' requires a 'bucket'")
        return self


class ComputeConfig(BaseModel):
    """Global compute / runtime settings."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, description="Random seed for reproducibility.")
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="warning",
        description="Logging verbosity.",
    )
    max_memory: str | None = Field(default=None, description="Maximum memory per worker (e.g. '4Gi').")
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Compute device: 'auto' picks cuda if available.",
    )
    progress_bar: bool = Field(default=True, description="Show tqdm progress bars.")
    threads: int = Field(default=4, gt=0, description="Number of worker threads.")


class ImageErrorsConfig(BaseModel):
    """Error-handling policy for image-processing failures."""

    model_config = ConfigDict(extra="forbid")

    on_decode_failure: Literal["silent_fail", "fail_fast"] = Field(
        default="silent_fail",
        description="Action when an image cannot be decoded.",
    )
    on_transform_error: Literal["silent_fail", "fail_fast"] = Field(
        default="silent_fail",
        description="Action when an image transform fails.",
    )
    on_unsupported_format: Literal["silent_fail", "fail_fast"] = Field(
        default="fail_fast",
        description="Action on unsupported image format.",
    )


class TabularErrorsConfig(BaseModel):
    """Error-handling policy for tabular-data failures."""

    model_config = ConfigDict(extra="forbid")

    on_missing_column: Literal["silent_fail", "fail_fast"] = Field(
        default="fail_fast",
        description="Action when a required column is missing.",
    )
    on_file_not_found: Literal["silent_fail", "fail_fast"] = Field(
        default="fail_fast",
        description="Action when a data file is not found.",
    )


class ErrorsConfig(BaseModel):
    """Aggregate error-handling configuration."""

    model_config = ConfigDict(extra="forbid")

    default: Literal["silent_fail", "fail_fast"] = Field(
        default="silent_fail",
        description="Default error action when no specific policy is set.",
    )
    images: ImageErrorsConfig | None = None
    tabular: TabularErrorsConfig | None = None
    max_failure_rate: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Maximum tolerated failure rate before the job aborts (from 0.0 to 1.0).",
    )
