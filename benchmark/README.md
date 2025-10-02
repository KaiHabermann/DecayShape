# K-Matrix Performance Benchmarks

This folder contains performance benchmarks for the DecayShape K-matrix implementation.

## Files

- `kmatrix_performance.py` - Main benchmark script
- `kmatrix_performance_results.csv` - Benchmark results (generated after running)
- `kmatrix_performance_results.png` - Performance plots (generated after running)

## Running the Benchmark

```bash
cd benchmark
python kmatrix_performance.py
```

## What it Tests

The benchmark tests K-matrix performance across:

- **Array lengths**: 100, 500, 1000, 2000, 5000, 10000 points
- **Number of channels**: 1, 2, 3, 4 channels
- **Number of poles**: 1, 2, 3, 4 poles

This gives a total of 96 different combinations (6 × 4 × 4).

## Output

The benchmark will:
1. Run timing tests for all combinations
2. Report detailed performance statistics
3. Create visualization plots
4. Save results to CSV and PNG files

The results help understand how K-matrix performance scales with:
- Problem size (array length)
- System complexity (number of channels and poles)
