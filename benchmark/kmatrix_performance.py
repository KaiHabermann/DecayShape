"""
K-matrix Performance Benchmark

This script benchmarks the performance of KMatrixAdvanced for different:
- Array lengths: from 100 to 10,000 points
- Number of channels: 1 to 4 channels
- Number of poles: 1 to 4 poles

Results are timed, reported, and plotted for analysis.
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import sys
import os
import numpy as np

# Add the parent directory to the path so we can import decayshape
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decayshape import config, set_backend
set_backend("numpy")
from decayshape.kmatrix_advanced import KMatrixAdvanced
from decayshape.particles import Channel, CommonParticles


def create_test_channels(n_channels):
    """Create test channels for benchmarking."""
    particles = [
        (CommonParticles.PI_PLUS, CommonParticles.PI_MINUS),
        (CommonParticles.K_PLUS, CommonParticles.K_MINUS),
        (CommonParticles.PROTON, CommonParticles.NEUTRON),
        (CommonParticles.PI_PLUS, CommonParticles.K_PLUS),
        (CommonParticles.K_PLUS, CommonParticles.PROTON),
        (CommonParticles.PI_MINUS, CommonParticles.NEUTRON),
    ]
    
    channels = []
    for i in range(n_channels):
        p1, p2 = particles[i % len(particles)]
        channels.append(Channel(particle1=p1, particle2=p2))
    
    return channels


def create_kmatrix(s_values, n_channels, n_poles):
    """Create a KMatrixAdvanced instance for benchmarking."""
    channels = create_test_channels(n_channels)
    
    # Create pole masses spread across the energy range
    s_min, s_max = np.min(s_values), np.max(s_values)
    pole_masses = np.linspace(np.sqrt(s_min) + 0.1, np.sqrt(s_max) - 0.1, n_poles)
    
    # Create production couplings (one per pole)
    production_couplings = [1.0 / (i + 1) for i in range(n_poles)]
    
    # Create decay couplings (n_poles × n_channels)
    decay_couplings = []
    for pole_idx in range(n_poles):
        for channel_idx in range(n_channels):
            # Vary coupling strength based on pole and channel
            coupling = 1.0 / ((pole_idx + 1) * (channel_idx + 1))
            decay_couplings.append(coupling)
    
    return KMatrixAdvanced(
        s=s_values,
        channels=channels,
        pole_masses=pole_masses.tolist(),
        production_couplings=production_couplings,
        decay_couplings=decay_couplings,
        output_channel=0  # Always return first channel
    )


def benchmark_kmatrix(array_length, n_channels, n_poles, n_runs=5):
    """Benchmark a single K-matrix configuration."""
    # Create s values
    s_values = np.linspace(0.1, 4.0, array_length)
    



    if config.backend_name == "jax":
        import jax

        kmat_object = create_kmatrix(s_values, n_channels, n_poles)
        fun = kmat_object.function
        kmat_jitted = jax.jit(fun)
        kmat_jitted(s_values[:1])
        kmat = lambda *args, **kwargs: kmat_jitted(s_values, *args, **kwargs)
        # kmat = jax.jit(lambda *args, **kwargs: kmat_object(*args, **kwargs))
    else:
        # Create K-matrix
        kmat_object = create_kmatrix(s_values, n_channels, n_poles)
        kmat = kmat_object

    # # Warm-up run
    # _ = kmat()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        result = kmat()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    return {
        'array_length': array_length,
        'n_channels': n_channels,
        'n_poles': n_poles,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'result_shape': result.shape,
        'result_dtype': str(result.dtype)
    }


def run_full_benchmark():
    """Run the complete benchmark suite."""
    print("=" * 60)
    print("K-MATRIX PERFORMANCE BENCHMARK")
    print("=" * 60)
    print()
    
    # Define parameter ranges
    array_lengths = [100, 500, 1000, 2000, 5000, 10000]
    n_channels_range = [1, 2, 3, 4, 5, 6]
    n_poles_range = [1, 2, 3, 4, 5, 6]
    
    # Calculate total combinations
    total_combinations = len(array_lengths) * len(n_channels_range) * len(n_poles_range)
    print(f"Testing {total_combinations} combinations:")
    print(f"  Array lengths: {array_lengths}")
    print(f"  Channels: {n_channels_range}")
    print(f"  Poles: {n_poles_range}")
    print()
    
    # Run benchmarks
    results = []
    combination_count = 0
    
    for array_length, n_channels, n_poles in product(array_lengths, n_channels_range, n_poles_range):
        combination_count += 1
        print(f"[{combination_count:2d}/{total_combinations}] "
              f"Array: {array_length:5d}, Channels: {n_channels}, Poles: {n_poles} ... ", end="", flush=True)
        
        try:
            result = benchmark_kmatrix(array_length, n_channels, n_poles)
            results.append(result)
            print(f"{result['mean_time']*1000:.2f} ms")
        except Exception as e:
            raise e
            print(f"FAILED: {e}")
    
    return pd.DataFrame(results)


def analyze_results(df):
    """Analyze and report benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS ANALYSIS")
    print("=" * 60)
    
    # Filter out failed runs
    valid_df = df.dropna(subset=['mean_time'])
    
    if len(valid_df) == 0:
        print("No valid benchmark results!")
        return
    
    print(f"\nValid results: {len(valid_df)}/{len(df)}")
    print(f"Failed results: {len(df) - len(valid_df)}")
    
    # Overall statistics
    print(f"\nOverall timing statistics:")
    print(f"  Fastest run: {valid_df['min_time'].min()*1000:.2f} ms")
    print(f"  Slowest run: {valid_df['mean_time'].max()*1000:.2f} ms")
    print(f"  Average time: {valid_df['mean_time'].mean()*1000:.2f} ms")
    
    # Performance by array length
    print(f"\nPerformance by array length:")
    for length in sorted(valid_df['array_length'].unique()):
        subset = valid_df[valid_df['array_length'] == length]
        avg_time = subset['mean_time'].mean() * 1000
        print(f"  {length:5d} points: {avg_time:8.2f} ms average")
    
    # Performance by channels
    print(f"\nPerformance by number of channels:")
    for n_channels in sorted(valid_df['n_channels'].unique()):
        subset = valid_df[valid_df['n_channels'] == n_channels]
        avg_time = subset['mean_time'].mean() * 1000
        print(f"  {n_channels} channels: {avg_time:8.2f} ms average")
    
    # Performance by poles
    print(f"\nPerformance by number of poles:")
    for n_poles in sorted(valid_df['n_poles'].unique()):
        subset = valid_df[valid_df['n_poles'] == n_poles]
        avg_time = subset['mean_time'].mean() * 1000
        print(f"  {n_poles} poles: {avg_time:8.2f} ms average")
    
    # Top 5 fastest and slowest configurations
    print(f"\nTop 5 fastest configurations:")
    fastest = valid_df.nsmallest(5, 'mean_time')
    for _, row in fastest.iterrows():
        print(f"  Array: {row['array_length']:5d}, Channels: {row['n_channels']}, "
              f"Poles: {row['n_poles']} -> {row['mean_time']*1000:.2f} ms")
    
    print(f"\nTop 5 slowest configurations:")
    slowest = valid_df.nlargest(5, 'mean_time')
    for _, row in slowest.iterrows():
        print(f"  Array: {row['array_length']:5d}, Channels: {row['n_channels']}, "
              f"Poles: {row['n_poles']} -> {row['mean_time']*1000:.2f} ms")


def create_plots(df):
    """Create performance visualization plots."""
    print(f"\nCreating performance plots...")
    
    # Filter out failed runs
    valid_df = df.dropna(subset=['mean_time'])
    
    if len(valid_df) == 0:
        print("No valid data to plot!")
        return
    
    # Convert time to milliseconds for plotting
    valid_df = valid_df.copy()
    valid_df['mean_time_ms'] = valid_df['mean_time'] * 1000
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('K-Matrix Performance Benchmark Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance vs Array Length
    ax1 = axes[0, 0]
    for n_channels in sorted(valid_df['n_channels'].unique()):
        for n_poles in sorted(valid_df['n_poles'].unique()):
            subset = valid_df[(valid_df['n_channels'] == n_channels) & 
                             (valid_df['n_poles'] == n_poles)]
            if len(subset) > 0:
                label = f'{n_channels}ch, {n_poles}p'
                ax1.plot(subset['array_length'], subset['mean_time_ms'], 
                        'o-', label=label, alpha=0.7)
    
    ax1.set_xlabel('Array Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Performance vs Array Length')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance vs Channels (for different array lengths)
    ax2 = axes[0, 1]
    array_lengths_to_plot = [1000, 5000, 10000]
    for array_length in array_lengths_to_plot:
        subset = valid_df[valid_df['array_length'] == array_length]
        if len(subset) > 0:
            grouped = subset.groupby('n_channels')['mean_time_ms'].mean()
            ax2.plot(grouped.index, grouped.values, 'o-', 
                    label=f'{array_length} points', alpha=0.7)
    
    ax2.set_xlabel('Number of Channels')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Performance vs Number of Channels')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance vs Poles (for different array lengths)
    ax3 = axes[1, 0]
    for array_length in array_lengths_to_plot:
        subset = valid_df[valid_df['array_length'] == array_length]
        if len(subset) > 0:
            grouped = subset.groupby('n_poles')['mean_time_ms'].mean()
            ax3.plot(grouped.index, grouped.values, 'o-', 
                    label=f'{array_length} points', alpha=0.7)
    
    ax3.set_xlabel('Number of Poles')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Performance vs Number of Poles')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of performance (channels vs poles for largest array)
    ax4 = axes[1, 1]
    largest_array = valid_df['array_length'].max()
    heatmap_data = valid_df[valid_df['array_length'] == largest_array]
    
    if len(heatmap_data) > 0:
        # Create pivot table for heatmap
        pivot_table = heatmap_data.pivot_table(
            values='mean_time_ms', 
            index='n_poles', 
            columns='n_channels',
            aggfunc='mean'
        )
        
        im = ax4.imshow(pivot_table.values, cmap='viridis', aspect='auto')
        ax4.set_xticks(range(len(pivot_table.columns)))
        ax4.set_yticks(range(len(pivot_table.index)))
        ax4.set_xticklabels(pivot_table.columns)
        ax4.set_yticklabels(pivot_table.index)
        ax4.set_xlabel('Number of Channels')
        ax4.set_ylabel('Number of Poles')
        ax4.set_title(f'Performance Heatmap\n({largest_array} points)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Time (ms)')
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                if not np.isnan(pivot_table.iloc[i, j]):
                    text = ax4.text(j, i, f'{pivot_table.iloc[i, j]:.1f}',
                                   ha="center", va="center", color="white", fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'kmatrix_performance_results.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Performance plots saved to: {plot_filename}")
    
    # Show the plot
    plt.show()


def save_results(df):
    """Save benchmark results to CSV file."""
    csv_filename = 'kmatrix_performance_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Benchmark results saved to: {csv_filename}")


def main():
    """Main benchmark execution."""
    print("Starting K-matrix performance benchmark...")
    print("This may take several minutes to complete.\n")
    
    # Run the benchmark
    start_time = time.time()
    results_df = run_full_benchmark()
    end_time = time.time()
    
    print(f"\nBenchmark completed in {end_time - start_time:.1f} seconds")
    
    # Analyze results
    analyze_results(results_df)
    
    # Create plots
    create_plots(results_df)
    
    # Save results
    save_results(results_df)
    
    print(f"\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - kmatrix_performance_results.csv")
    print("  - kmatrix_performance_results.png")


if __name__ == "__main__":
    main()
