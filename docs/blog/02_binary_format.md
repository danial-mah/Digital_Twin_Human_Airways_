# 02 - Understanding The Binary Snapshot Format

The dataset stores large arrays in `.bin` files.

These files are not text files. They are raw binary numeric data.

## What We Discovered

At first, we tested reading the files as `float32`.

That produced wrong values, including `NaN`.

The correct format for this dataset is:

```python
raw_values = np.fromfile(path, dtype=np.float64)
cleaned_values = raw_values[1:]
snapshot = cleaned_values.astype(np.float32)
```

## Why Read As float64?

The original binary files are stored using 64-bit floating point numbers.

If we read them as `float32`, Python interprets every 8-byte number as two separate 4-byte numbers. That corrupts the data.

This caused the PCA error:

```text
ValueError: Input X contains NaN.
```

## Why Remove The First Value?

The first value behaves like a header, not real field data.

So we remove it:

```python
cleaned_values = raw_values[1:]
```

## Why Convert To float32 Afterward?

After reading correctly, we convert to `float32` to reduce memory usage.

So the final matrix is still smaller, but the data is not corrupted.

## Key Lesson

Read the raw data using its true format first. Compress or convert only after the data has been interpreted correctly.
