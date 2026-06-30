# YAML Basics

Quick reference for writing YAML configuration files.

## Basic Syntax

| YAML | Meaning |
|------|---------|
| `key: value` | Assign a value |
| `- item` | List item |
| `# comment` | Comment (ignored) |
| `"string"` | Explicit string (use quotes if value has special chars) |
| `-` | Indentation (use spaces, not tabs) |

## Data Types

### Strings

```yaml
name: hello        # No quotes needed
message: "hello world"  # Quotes for special chars
```

### Numbers

```yaml
count: 42         # Integer
price: 19.99      # Float
```

### Booleans

```yaml
enabled: true   # or yes, on
disabled: false # or no, off
```

### Lists

```yaml
# Inline syntax
colors: [red, green, blue]

# Block syntax
colors:
  - red
  - green
  - blue
```

### Dictionaries

```yaml
# Inline syntax
server: {host: localhost, port: 8080}

# Block syntax
server:
  host: localhost
  port: 8080
```

## Indentation Rules

- **Use spaces** (not tabs)
- Consistent indentation (usually 2 spaces)
- Nested items are indented further

```yaml
parent:
  child:
    grandchild: value
```

## Common Patterns in DQM-ML

### Key-Value Pairs

```yaml
dataloaders:
  loaders:
    - name: my_data
      type: parquet
      path: data/train.parquet
```

### Lists of Options

```yaml
metrics:
  processors:
    - name: completeness
      type: completeness
      columns:
        input:
          - name
          - age
          - score
```

### Nested Configuration

```yaml
features:
  processors:
    - name: image_embedding
      type: features_embeddings
      model:
        arch: resnet18
      infer:
        batch_size: 10
        width: 224
```

## Common Mistakes to Avoid

### Wrong: Using Tabs

```yaml
# Wrong — uses tabs (shown as →)
key:→value

# Correct — uses spaces (shown as ·)
key:.value
```

### Wrong: Inconsistent Indentation

```yaml
# Wrong — key2 indented inconsistently
key1: value
··key2: value    # ← 2 extra spaces (shown as ..), wrong level

# Correct — consistent indentation
key1: value
key2: value
```

### Wrong: Missing Colon

```yaml
# Wrong - missing colon after key
key value

# Correct
key: value
```

## Tools for Writing YAML

- [YAML Validator](https://www.yamllint.com/) - Check syntax
- [YAML to JSON](https://www.convertjson.com/yaml-to-json.php) - Convert between formats

## Related Pages

- [Configuration](configuration/overview.md) - Full configuration guide
- [Data Selection](configuration/data_selection.md) - Data source configuration
- [CLI Reference](cli.md) - Command-line usage