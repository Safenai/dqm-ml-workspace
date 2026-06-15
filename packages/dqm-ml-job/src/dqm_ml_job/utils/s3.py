"""S3 utilities for DQM-ML job."""

import os

import pyarrow as pa


def get_s3_filesystem(
    access_key: str | None = None,
    secret_key: str | None = None,
    endpoint: str | None = None,
    region: str | None = None,
) -> pa.fs.S3FileSystem | None:
    """Create and return an S3 filesystem instance.

    Uses provided credentials or falls back to environment variables:
    - S3_ACCESS_KEY
    - S3_SECRET_KEY
    - S3_ENDPOINT
    - S3_REGION

    Args:
        access_key: Optional AWS access key (overrides env if provided)
        secret_key: Optional AWS secret key (overrides env if provided)
        endpoint: Optional S3 endpoint URL (overrides env if provided)
        region: Optional AWS region (overrides env if provided)

    Returns:
        Configured S3 filesystem or None if credentials not available.
    """
    os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"

    access_key = access_key or os.getenv("S3_ACCESS_KEY")
    secret_key = secret_key or os.getenv("S3_SECRET_KEY")
    endpoint = endpoint or os.getenv("S3_ENDPOINT", "")
    region = region or os.getenv("S3_REGION")

    if not access_key or not secret_key:
        return None

    config = {
        "access_key": access_key,
        "secret_key": secret_key,
        "endpoint_override": endpoint,
        "region": region,
    }

    # Mock environment: configure to avoid multipart upload issues
    if os.getenv("ENVIRONMENT") == "mock":
        config["background_writes"] = False  # type: ignore[assignment]
        config["retry_strategy"] = pa.fs.AwsStandardS3RetryStrategy(max_attempts=10)

    return pa.fs.S3FileSystem(**config)
