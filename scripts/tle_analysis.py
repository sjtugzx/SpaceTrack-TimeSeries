#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from datetime import datetime
import matplotlib
import argparse
matplotlib.rcParams['font.sans-serif'] = ['Arial']  # For displaying labels correctly
matplotlib.rcParams['axes.unicode_minus'] = False  # For displaying minus sign correctly

# Data paths
data_dir = "../data/Starlink_Tle"
starlink_file = os.path.join(data_dir, "sl_merged_tle_data.csv")
all_satellites_file = os.path.join(data_dir, "ori_merged_tle_data.csv")
starlink_output_dir = os.path.join(data_dir, "starlink_analysis_results")
all_output_dir = os.path.join(data_dir, "all_satellites_analysis_results")

# Create output directories
for directory in [starlink_output_dir, all_output_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_preprocess_data(data_file):
    """
    Load and preprocess TLE data
    
    Args:
        data_file: Path to the CSV file to load
        
    Returns:
        Preprocessed DataFrame
    """
    print(f"Loading data from {os.path.basename(data_file)}...")
    # Read CSV file
    df = pd.read_csv(data_file)
    
    # Check and standardize column names
    column_renames = {
        '1st_Derivative_of_Mean_Motion': 'First_Derivative_of_Mean_Motion',
        '2nd_Derivative_of_Mean_Motion': 'Second_Derivative_of_Mean_Motion'
    }
    
    # Only rename columns that exist in the dataframe
    rename_dict = {k: v for k, v in column_renames.items() if k in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Convert time column to datetime format
    df['datetime'] = pd.to_datetime(df['iso_time'])
    
    # Sort by time
    df = df.sort_values('datetime')
    
    # Remove duplicate satellite-time combinations
    df = df.drop_duplicates(subset=['Satellite_Name', 'datetime'])
    
    return df

def basic_statistics(df, output_dir):
    """
    Calculate basic statistics and generate boxplots for key orbital parameters
    """
    print("\n========== Basic Statistics ==========")
    
    # Satellite count statistics
    unique_satellites = df['Satellite_Name'].nunique()
    print(f"Unique satellite count: {unique_satellites}")
    
    # Time range
    time_range = df['datetime'].max() - df['datetime'].min()
    print(f"Data time range: {time_range.days} days")
    print(f"Start date: {df['datetime'].min().strftime('%Y-%m-%d')}")
    print(f"End date: {df['datetime'].max().strftime('%Y-%m-%d')}")
    
    # Orbital parameter statistics
    for param in ['Inclination', 'Eccentricity', 'Mean_Motion']:
        mean_val = df[param].mean()
        std_val = df[param].std()
        min_val = df[param].min()
        max_val = df[param].max()
        print(f"{param} - Mean: {mean_val:.6f}, Std Dev: {std_val:.6f}, Range: [{min_val:.6f}, {max_val:.6f}]")
    
    # Calculate observations per satellite
    obs_counts = df.groupby('Satellite_Name').size()
    print(f"Average observations per satellite: {obs_counts.mean():.2f}")
    print(f"Most observed satellite: {obs_counts.idxmax()} ({obs_counts.max()} observations)")
    
    # Generate boxplots for key orbital parameters
    print("\nGenerating boxplots for key orbital parameters...")
    key_params = ['Inclination', 'Eccentricity', 'Mean_Motion', 
                  'Right_Ascension_of_Node', 'Argument_of_Perigee', 
                  'Mean_Anomaly', 'BSTAR_Drag_Term']
    
    # Create a figure with subplots arranged in a vertical layout
    fig, axes = plt.subplots(len(key_params), 1, figsize=(12, 16))
    
    # Create boxplots for each parameter
    for i, param in enumerate(key_params):
        # Create boxplot
        boxplot = axes[i].boxplot(df[param], vert=False, patch_artist=True)
        
        # Customize boxplot colors
        for box in boxplot['boxes']:
            box.set(facecolor='lightblue', alpha=0.7)
        
        # Set title and labels
        axes[i].set_title(f'{param} Distribution')
        axes[i].set_xlabel(param)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add statistical annotations
        stats_text = f"Mean: {df[param].mean():.6f}\nStd Dev: {df[param].std():.6f}\n" \
                     f"Min: {df[param].min():.6f}\nMax: {df[param].max():.6f}"
        axes[i].text(0.02, 0.7, stats_text, transform=axes[i].transAxes, 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "orbital_parameters_boxplots.png"), dpi=300)
    plt.close()
    
    # Generate a single combined boxplot for easier comparison
    plt.figure(figsize=(14, 10))
    
    # Normalize data for comparison (z-score)
    normalized_data = pd.DataFrame()
    for param in key_params:
        # Skip BSTAR for combined plot as its scale is very different
        if param != 'BSTAR_Drag_Term':
            normalized_data[param] = (df[param] - df[param].mean()) / df[param].std()
    
    # Create combined boxplot
    sns.boxplot(data=normalized_data)
    plt.title('Normalized Comparison of Orbital Parameters')
    plt.ylabel('Z-score (Standard Deviations from Mean)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "orbital_parameters_combined_boxplot.png"), dpi=300)
    plt.close()
    
    print("Boxplots generated and saved to output directory")
    
    # Save results to file
    with open(os.path.join(output_dir, "basic_statistics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Unique satellite count: {unique_satellites}\n")
        f.write(f"Data time range: {time_range.days} days\n")
        f.write(f"Start date: {df['datetime'].min().strftime('%Y-%m-%d')}\n")
        f.write(f"End date: {df['datetime'].max().strftime('%Y-%m-%d')}\n\n")
        
        for param in ['Inclination', 'Eccentricity', 'Mean_Motion']:
            mean_val = df[param].mean()
            std_val = df[param].std()
            min_val = df[param].min()
            max_val = df[param].max()
            f.write(f"{param} - Mean: {mean_val:.6f}, Std Dev: {std_val:.6f}, Range: [{min_val:.6f}, {max_val:.6f}]\n")
        
        f.write(f"\nAverage observations per satellite: {obs_counts.mean():.2f}\n")
        f.write(f"Most observed satellite: {obs_counts.idxmax()} ({obs_counts.max()} observations)\n")
    
    return obs_counts

def orbital_parameters_distribution(df, output_dir):
    """
    Analyze orbital parameter distributions
    """
    print("\n========== Orbital Parameter Distribution Analysis ==========")
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Create a 2x3 grid of plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Inclination distribution
    sns.histplot(df['Inclination'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Inclination Distribution')
    axes[0, 0].set_xlabel('Inclination (degrees)')
    
    # Eccentricity distribution
    sns.histplot(df['Eccentricity'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Eccentricity Distribution')
    axes[0, 1].set_xlabel('Eccentricity')
    
    # Mean motion distribution
    sns.histplot(df['Mean_Motion'], kde=True, ax=axes[0, 2])
    axes[0, 2].set_title('Mean Motion Distribution')
    axes[0, 2].set_xlabel('Mean Motion (revs/day)')
    
    # Right ascension distribution
    sns.histplot(df['Right_Ascension_of_Node'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Right Ascension of Node Distribution')
    axes[1, 0].set_xlabel('Right Ascension of Node (radians)')
    
    # Argument of perigee distribution
    sns.histplot(df['Argument_of_Perigee'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Argument of Perigee Distribution')
    axes[1, 1].set_xlabel('Argument of Perigee (radians)')
    
    # Mean anomaly distribution
    sns.histplot(df['Mean_Anomaly'], kde=True, ax=axes[1, 2])
    axes[1, 2].set_title('Mean Anomaly Distribution')
    axes[1, 2].set_xlabel('Mean Anomaly (radians)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "orbital_parameters_distribution.png"), dpi=300)
    plt.close()
    
    print("Orbital parameter distribution plots saved")

def orbital_planes_analysis(df, output_dir):
    """
    Analyze orbital plane distributions
    """
    print("\n========== Orbital Plane Analysis ==========")
    
    # Create scatter plot with right ascension and inclination as coordinates
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Right_Ascension_of_Node'], df['Inclination'], 
                         alpha=0.6, c=df['Mean_Motion'], cmap='viridis')
    
    plt.colorbar(scatter, label='Mean Motion (revs/day)')
    plt.xlabel('Right Ascension of Node (radians)')
    plt.ylabel('Inclination (degrees)')
    plt.title('Satellite Orbital Plane Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "orbital_planes.png"), dpi=300)
    plt.close()
    
    # Calculate orbital plane clustering
    from sklearn.cluster import KMeans
    
    # Select features for clustering
    features = df[['Inclination', 'Right_Ascension_of_Node']].copy()
    
    # Try different numbers of clusters
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow chart to find optimal number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('K-means Clustering Elbow Chart')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "kmeans_elbow.png"), dpi=300)
    plt.close()
    
    # Choose optimal number of clusters based on the elbow chart (assuming 4 here)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['orbit_cluster'] = kmeans.fit_predict(features)
    
    # Plot clustering results
    plt.figure(figsize=(12, 8))
    for cluster in range(optimal_k):
        cluster_data = df[df['orbit_cluster'] == cluster]
        plt.scatter(cluster_data['Right_Ascension_of_Node'], 
                   cluster_data['Inclination'], 
                   label=f'Orbital Plane {cluster+1}',
                   alpha=0.7)
    
    plt.xlabel('Right Ascension of Node (radians)')
    plt.ylabel('Inclination (degrees)')
    plt.title('Satellite Orbital Plane Clustering')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "orbital_planes_clusters.png"), dpi=300)
    plt.close()
    
    # Calculate number of satellites in each orbital plane
    plane_counts = df.groupby('orbit_cluster')['Satellite_Name'].nunique()
    print("Satellite count by orbital plane:")
    for plane, count in plane_counts.items():
        print(f"Orbital Plane {plane+1}: {count} satellites")
    
    return df['orbit_cluster']

def time_series_analysis(df, output_dir, sample_satellites=5):
    """
    Time series analysis of parameter changes for specific satellites
    """
    print("\n========== Time Series Analysis ==========")
    
    # Select satellites with the most observations
    obs_counts = df.groupby('Satellite_Name').size()
    top_satellites = obs_counts.nlargest(sample_satellites).index.tolist()
    
    print(f"Selected {sample_satellites} satellites with the most observations for time series analysis:")
    for sat in top_satellites:
        print(f"- {sat}")
    
    # Plot time series for each satellite
    for param in ['Mean_Motion', 'Eccentricity', 'Inclination', 'BSTAR_Drag_Term']:
        plt.figure(figsize=(14, 8))
        
        for sat in top_satellites:
            sat_data = df[df['Satellite_Name'] == sat].sort_values('datetime')
            plt.plot(sat_data['datetime'], sat_data[param], marker='o', linewidth=1, markersize=4, label=sat)
        
        plt.xlabel('Time')
        plt.ylabel(param)
        plt.title(f'{param} Changes Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"time_series_{param}.png"), dpi=300)
        plt.close()
    
    print("Time series analysis plots saved")

def decay_analysis(df, output_dir):
    """
    Orbital decay analysis
    """
    print("\n========== Orbital Decay Analysis ==========")
    
    # Calculate average BSTAR value for each satellite
    bstar_mean = df.groupby('Satellite_Name')['BSTAR_Drag_Term'].mean().reset_index()
    bstar_mean = bstar_mean.sort_values('BSTAR_Drag_Term', ascending=False)
    
    # Plot top 20 satellites with highest BSTAR values
    top_20 = bstar_mean.head(20)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(top_20['Satellite_Name'], top_20['BSTAR_Drag_Term'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4e}',
                ha='center', va='bottom', rotation=90)
    
    plt.xlabel('Satellite Name')
    plt.ylabel('Average BSTAR Drag Term')
    plt.title('Top 20 Satellites with Highest BSTAR Drag Term')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "top_bstar_satellites.png"), dpi=300)
    plt.close()
    
    # Analyze first derivative of mean motion, indicating orbital height change rate
    mm_derivative_mean = df.groupby('Satellite_Name')['First_Derivative_of_Mean_Motion'].mean().reset_index()
    mm_derivative_mean = mm_derivative_mean.sort_values('First_Derivative_of_Mean_Motion', ascending=False)
    
    # Plot top 20 satellites with highest mean motion derivative
    top_20_mm = mm_derivative_mean.head(20)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(top_20_mm['Satellite_Name'], top_20_mm['First_Derivative_of_Mean_Motion'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4e}',
                ha='center', va='bottom', rotation=90)
    
    plt.xlabel('Satellite Name')
    plt.ylabel('Average First Derivative of Mean Motion')
    plt.title('Top 20 Satellites with Highest Mean Motion Change Rate')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "top_mm_derivative_satellites.png"), dpi=300)
    plt.close()
    
    print("Orbital decay analysis plots saved")

def correlation_analysis(df, output_dir):
    """
    Analyze correlations between orbital parameters
    """
    print("\n========== Parameter Correlation Analysis ==========")
    
    # Select orbital parameters of interest
    params = ['Inclination', 'Eccentricity', 'Mean_Motion', 'Right_Ascension_of_Node', 
              'Argument_of_Perigee', 'Mean_Anomaly', 'BSTAR_Drag_Term']
    
    # Calculate correlation coefficient matrix
    corr = df[params].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Orbital Parameter Correlation Matrix')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "parameter_correlation.png"), dpi=300)
    plt.close()
    
    print("Parameter correlation analysis plot saved")

def analyze_dataset(data_file, output_dir, dataset_name):
    """
    Perform complete analysis on a dataset
    
    Args:
        data_file: Path to the data file
        output_dir: Directory to save results
        dataset_name: Name of the dataset for display purposes
    """
    print(f"\n\n{'='*20} ANALYZING {dataset_name.upper()} DATASET {'='*20}\n")
    
    # Load data
    df = load_and_preprocess_data(data_file)
    
    # Perform analyses
    basic_statistics(df, output_dir)
    orbital_parameters_distribution(df, output_dir)
    df['orbit_cluster'] = orbital_planes_analysis(df, output_dir)
    time_series_analysis(df, output_dir)
    decay_analysis(df, output_dir)
    correlation_analysis(df, output_dir)
    
    print(f"\nAnalysis of {dataset_name} complete! Results saved to: {output_dir}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze satellite TLE data.')
    parser.add_argument('--dataset', type=str, choices=['starlink', 'all', 'both'], 
                       default='both', help='Dataset to analyze (default: both)')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    if args.dataset == 'starlink' or args.dataset == 'both':
        analyze_dataset(starlink_file, starlink_output_dir, "Starlink")
    
    if args.dataset == 'all' or args.dataset == 'both':
        analyze_dataset(all_satellites_file, all_output_dir, "All Satellites")
    
    print("\nAll analyses complete!")

if __name__ == "__main__":
    main()