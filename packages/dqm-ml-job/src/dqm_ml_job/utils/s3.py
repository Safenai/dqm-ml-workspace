"""S3 utilities for DQM-ML job."""

import os
from typing import Any

from dqm_ml_core.models.global_ import StorageConfig
import pyarrow as pa

_SIMPLE_KWARGS: list[tuple[str, str]] = [
    ("region", "region"),
    ("session_token", "session_token"),
    ("anonymous", "anonymous"),
    ("request_timeout", "request_timeout"),
    ("connect_timeout", "connect_timeout"),
    ("scheme", "scheme"),
    ("proxy_options", "proxy_options"),
    ("tls_ca_file_path", "tls_ca_file_path"),
]

_ROLE_KWARGS: list[tuple[str, str]] = [
    ("session_name", "session_name"),
    ("external_id", "external_id"),
]


def _build_retry_strategy(
    retry_cfg: Any,
) -> pa.fs.S3RetryStrategy | None:
    """Build an S3 retry strategy from configuration.

    Args:
        retry_cfg: RetryConfig instance or dict with mode and max_attempts.

    Returns:
        Configured PyArrow S3 retry strategy, or None if no config.
    """
    from dqm_ml_core.models.global_ import RetryConfig

    if retry_cfg is None:
        return None
    if isinstance(retry_cfg, dict):
        retry_cfg = RetryConfig.model_validate(retry_cfg)
    max_attempts = retry_cfg.max_attempts
    if retry_cfg.mode == "default":
        return pa.fs.AwsDefaultS3RetryStrategy(max_attempts=max_attempts)
    return pa.fs.AwsStandardS3RetryStrategy(max_attempts=max_attempts)


def _apply_simple_kwargs(storage_config: StorageConfig, kwargs: dict[str, Any]) -> None:
    """Apply simple 1-to-1 config-attribute-to-kwargs mappings."""
    for attr, kwarg in _SIMPLE_KWARGS:
        val = getattr(storage_config, attr, None)
        if val is not None:
            kwargs[kwarg] = val if kwarg != "anonymous" else True


def _apply_role_kwargs(storage_config: StorageConfig, kwargs: dict[str, Any]) -> None:
    """Apply role-based access kwargs."""
    kwargs["role_arn"] = storage_config.role_arn
    for attr, kwarg in _ROLE_KWARGS:
        val = getattr(storage_config, attr, None)
        if val is not None:
            kwargs[kwarg] = val
    kwargs["load_frequency"] = storage_config.load_frequency


def get_s3_filesystem(
    storage_config: StorageConfig,
) -> pa.fs.S3FileSystem | None:
    """Create and return an S3 filesystem instance from StorageConfig.

    Credentials fall back to environment variables:
      S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT, S3_REGION

    Args:
        storage_config: Validated StorageConfig instance.

    Returns:
        Configured S3 filesystem or None if credentials not available.
    """
    os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = storage_config.checksum_validation

    access_key = storage_config.access_key or os.getenv("S3_ACCESS_KEY")
    secret_key = storage_config.secret_key or os.getenv("S3_SECRET_KEY")

    if not access_key or not secret_key:
        return None

    kwargs: dict[str, Any] = {
        "access_key": access_key,
        "secret_key": secret_key,
        "endpoint_override": storage_config.endpoint or os.getenv("S3_ENDPOINT", ""),
    }

    _apply_simple_kwargs(storage_config, kwargs)

    if storage_config.role_arn:
        _apply_role_kwargs(storage_config, kwargs)

    if storage_config.retry:
        kwargs["retry_strategy"] = _build_retry_strategy(storage_config.retry)

    if os.getenv("ENVIRONMENT") == "mock":
        kwargs.setdefault("background_writes", False)
        kwargs.setdefault("retry_strategy", pa.fs.AwsStandardS3RetryStrategy(max_attempts=10))

    return pa.fs.S3FileSystem(**kwargs)
