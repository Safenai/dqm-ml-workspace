## 4. Global Configuration

Global configuration sets the execution environment. These settings can be overridden at the interface level or per processor.

```yaml
storage:     # where are files located?
  ...
compute:     # what hardware / parallelism?
  ...
errors:      # what happens on failures?
  ...
```

### 4.1 Storage

Configure the storage platform for datasets and outputs:

```yaml
storage:
  type: s3                              # or local
  bucket: datasets                      # required when type=s3
  # Auth — choose one strategy
  access_key: AKIAIOSFODNN7EXAMPLE
  secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
  session_token: <token>                # temporary credentials
  anonymous: false                      # anonymous access
  role_arn: arn:aws:iam::123456789012:role/MyRole
  session_name: dqm-ml                  # session name for role assumption
  external_id: <external_id>
  load_frequency: 900                   # credential reload frequency (seconds)
  # Connection
  region: us-east-1
  endpoint: https://s3.amazonaws.com    # S3-compatible endpoint
  request_timeout: 60                   # seconds
  connect_timeout: 5                    # seconds
  scheme: https
  proxy_options: <url>
  tls_ca_file_path: /path/to/ca.pem
  # Retry
  retry:
    mode: standard                      # default | standard
    max_attempts: 3
  # Security
  checksum_validation: when_required    # when_required | always | never
```

### 4.2 Compute

Configure computation device, performance, and reproducibility:

```yaml
compute:
  seed: 42                              # reproducibility seed
  log_level: warning                    # debug | info | warning | error
  device: cuda                          # or cpu
  progress_bar: true                    # show tqdm progress bar during processing
  threads: 4                            # parallel reading threads
```

### 4.3 Errors (Global)

```yaml
errors:
  default: silent_fail                  # or fail_fast
  max_failure_rate: 0.05                # auto-switch to fail_fast when exceeded
  images:
    on_decode_failure: silent_fail      # corrupt bytes, missing file, S3 error
    on_transform_error: silent_fail     # resize / normalize / tensor conversion failure
    on_unsupported_format: fail_fast    # image format outside PIL's supported list
  tabular:
    on_missing_column: fail_fast
    on_file_not_found: fail_fast
```

> Interface-level and processor-level `errors` override global settings for their scope.
