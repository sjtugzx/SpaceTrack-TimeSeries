#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Satellite Ephemeris Data Analysis Module

This module provides comprehensive analysis tools for satellite ephemeris data,
including orbital parameter calculations, statistical analysis, and visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from datetime import datetime
import matplotlib
import argparse
from scipy.spatial.transform import Rotation
import math
from multiprocessing import Pool, cpu_count
from functools import partial

# Configure matplotlib for better visualization
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# Set global font style for consistent visualization
plt.rcParams.update({
    'font.size': 24,  # Base font size
    'axes.titlesize': 28,  # Title font size
    'axes.labelsize': 26,  # Axis label font size
    'xtick.labelsize': 24,  # X-axis tick label font size
    'ytick.labelsize': 24,  # Y-axis tick label font size
    'legend.fontsize': 28,  # Legend font size
    'font.weight': 'bold',  # Font weight
    'axes.titleweight': 'bold',  # Title weight
    'axes.labelweight': 'bold',  # Axis label weight
    'legend.title_fontsize': 30  # Legend title font size
})

# Configure seaborn for enhanced visualization
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Define data paths
data_dir = "../data/ephemeris"
output_dir = os.path.join(data_dir, "analysis_results")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def calculate_orbital_parameters_chunk(chunk_data):
    """
    Helper function to calculate orbital parameters for a data chunk.
    Used for multiprocessing to improve performance.
    
    Args:
        chunk_data (pd.DataFrame): A subset of the main DataFrame containing position and velocity data
        
    Returns:
        pd.DataFrame: DataFrame subset with calculated orbital parameters
    """
    # Earth gravitational constant (km³/s²)
    mu = 398600.4418
    
    # Calculate orbital parameters for each row
    for idx, row in chunk_data.iterrows():
        try:
            # Extract position and velocity vectors
            r = np.array([row['X'], row['Y'], row['Z']])
            v = np.array([row['VX'], row['VY'], row['VZ']])
            
            # Calculate angular momentum vector
            h = np.cross(r, v)
            # Calculate node line vector
            n = np.cross([0, 0, 1], h)
            
            # Calculate eccentricity vector
            e = np.cross(v, h) / mu - r / np.linalg.norm(r)
            
            # Store eccentricity
            chunk_data.at[idx, 'Eccentricity'] = np.linalg.norm(e)
            
            # Calculate and store inclination (radians)
            inclination = np.arccos(h[2] / np.linalg.norm(h))
            chunk_data.at[idx, 'Inclination'] = min(inclination, np.pi - inclination)
            
            # Calculate and store RAAN (Right Ascension of Ascending Node)
            raan = np.arctan2(n[1], n[0])
            if raan < 0:
                raan += 2 * np.pi
            chunk_data.at[idx, 'Right_Ascension_of_Node'] = raan
            
            # Calculate and store argument of perigee
            if np.linalg.norm(n) > 0 and np.linalg.norm(e) > 0:
                argp = np.arctan2(np.dot(e, np.cross(n, h)), np.dot(e, n))
                if argp < 0:
                    argp += 2 * np.pi
                chunk_data.at[idx, 'Argument_of_Perigee'] = argp
            
            # Calculate orbital period and mean motion
            r_mag = np.linalg.norm(r)
            v_mag = np.linalg.norm(v)
            
            # Calculate semi-major axis
            a = 1 / (2/r_mag - v_mag**2/mu)
            
            # Calculate mean motion (rad/s)
            n_instant = np.sqrt(mu / a**3)
            
            # Convert to revs/day (consistent with TLE data)
            chunk_data.at[idx, 'Mean_Motion'] = n_instant * 86400 / (2 * np.pi)
            
            # Calculate true anomaly
            if np.linalg.norm(e) > 0:
                cos_f = np.dot(e, r) / (np.linalg.norm(e) * r_mag)
                sin_f = np.dot(np.cross(e, r), h) / (np.linalg.norm(e) * r_mag * np.linalg.norm(h))
                true_anomaly = np.arctan2(sin_f, cos_f)
                if true_anomaly < 0:
                    true_anomaly += 2 * np.pi
                chunk_data.at[idx, 'True_Anomaly'] = true_anomaly
            
            # Calculate mean anomaly
            if np.linalg.norm(e) > 0:
                E = 2 * np.arctan(np.sqrt((1 - np.linalg.norm(e)) / (1 + np.linalg.norm(e))) * 
                                np.tan(true_anomaly / 2))
                chunk_data.at[idx, 'Mean_Anomaly'] = E - np.linalg.norm(e) * np.sin(E)
            
        except Exception as e:
            print(f"Error calculating orbital parameters for {row['Satellite_Name']} at {row['datetime']}: {str(e)}")
            continue
    
    return chunk_data

def calculate_orbital_parameters(df):
    """
    Calculate orbital parameters using multiprocessing
    
    Args:
        df: DataFrame containing position and velocity data
        
    Returns:
        DataFrame with added orbital parameters
    """
    # Determine number of processes
    n_cores = max(1, cpu_count() - 1)  # Reserve one core for the system
    
    # Split data into multiple chunks
    chunk_size = len(df) // n_cores
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Use multiprocessing pool to process in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(calculate_orbital_parameters_chunk, chunks)
    
    # Merge results
    df = pd.concat(results, ignore_index=True)
    
    # Print data range check
    print("\nOrbital parameters range check:")
    for param in ['Inclination', 'Mean_Motion', 'Eccentricity', 'Right_Ascension_of_Node']:
        if param in df.columns:
            print(f"\n{param}:")
            print(f"Range: [{df[param].min():.6f}, {df[param].max():.6f}]")
            print(f"Mean: {df[param].mean():.6f}")
            print(f"Standard deviation: {df[param].std():.6f}")
    
    return df

def process_file(file_path):
    """
    Helper function to process a single file, used for multiprocessing
    
    Args:
        file_path: CSV file path
        
    Returns:
        Processed DataFrame
    """
    try:
        print(f"Processing {file_path}...")
        
        # Read the first few rows of the CSV to check column names
        df_sample = pd.read_csv(file_path, nrows=5)
        print(f"File columns: {df_sample.columns.tolist()}")
        
        # Check and standardize column names
        column_mapping = {
            'Vx': 'VX',
            'Vy': 'VY',
            'Vz': 'VZ',
            'x': 'X',
            'y': 'Y',
            'z': 'Z',
            'timestamp': 'Timestamp',
            'Satellite_Number': 'Satellite_Name',
            'Coordinate': 'Coordinate',
            'Ephemeris_Source': 'Ephemeris_Source'
        }
        
        # Rename columns (only rename existing columns)
        rename_dict = {k: v for k, v in column_mapping.items() if k in df_sample.columns}
        if rename_dict:
            print(f"Renaming columns: {rename_dict}")
        
        # Check if required columns exist (consider case sensitivity)
        required_columns = ['Satellite_Name', 'Timestamp', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ']
        required_columns_lower = [col.lower() for col in required_columns]
        
        # Get all column names (convert to lowercase)
        available_columns_lower = [col.lower() for col in df_sample.columns]
        
        # Check required columns
        missing_columns = []
        for req_col in required_columns:
            if req_col.lower() not in available_columns_lower:
                missing_columns.append(req_col)
        
        if missing_columns:
            print(f"Warning: {file_path} is missing required columns: {missing_columns}")
            print(f"Available columns: {df_sample.columns.tolist()}")
            return None
        
        # Use chunksize to read large files in chunks
        chunk_size = 10000  # Read 10000 rows at a time
        chunks = []
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Rename columns
            if rename_dict:
                chunk = chunk.rename(columns=rename_dict)
        
        # If Timestamp column does not exist, try using other time columns
        if 'Timestamp' not in chunk.columns:
            time_columns = [col for col in chunk.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_columns:
                chunk['Timestamp'] = chunk[time_columns[0]]
                print(f"Using {time_columns[0]} as Timestamp column")
            else:
                print(f"Warning: {file_path} does not find Timestamp column")
                return None
        
        # Process Satellite_Name column
        if 'Satellite_Name' in chunk.columns:
            # If Satellite_Name is DataFrame, take the first column
            if isinstance(chunk['Satellite_Name'], pd.DataFrame):
                print(f"Converting Satellite_Name to Series")
                chunk['Satellite_Name'] = chunk['Satellite_Name'].iloc[:, 0]
            # If Series but contains duplicate column names, only keep the first one
            elif chunk.columns.duplicated().any():
                print(f"Removing duplicates")
                chunk = chunk.loc[:, ~chunk.columns.duplicated()]
        
        chunks.append(chunk)
        
        # Merge all chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Print processed column information
        print(f"Processing completed: {file_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Safely print Satellite_Name information
        try:
            if isinstance(df['Satellite_Name'], pd.DataFrame):
                print(f"Satellite_Name type: DataFrame")
                print(f"Satellite_Name column count: {df['Satellite_Name'].shape[1]}")
                print(f"First few Satellite_Name values: {df['Satellite_Name'].iloc[:5, 0].values.tolist()}")
            else:
                print(f"Satellite_Name type: {type(df['Satellite_Name'])}")
                print(f"First few Satellite_Name values: {df['Satellite_Name'].head().values.tolist()}")
        except Exception as e:
            print(f"Error printing Satellite_Name information: {str(e)}")
            print(f"Satellite_Name column type: {type(df['Satellite_Name'])}")
            print(f"Satellite_Name column shape: {df['Satellite_Name'].shape if hasattr(df['Satellite_Name'], 'shape') else 'unknown'}")
        
        return df
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Error stack trace: {traceback.format_exc()}")
        return None

def apply_low_pass_filter(df, window_hours=24):
    """
    Apply low-pass filter to satellite ephemeris data to remove short-period components.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        window_hours (int): Size of sliding window in hours
        
    Returns:
        pd.DataFrame: Processed DataFrame with filtered parameters
    """
    print(f"\nApplying {window_hours} hour low-pass filter...")
    
    try:
        # Create a copy of the DataFrame
        filtered_df = df.copy()
        
        # Ensure Satellite_Name column is Series rather than DataFrame
        if isinstance(filtered_df['Satellite_Name'], pd.DataFrame):
            print("Detected Satellite_Name is DataFrame, converting...")
            filtered_df['Satellite_Name'] = filtered_df['Satellite_Name'].iloc[:, 0]
        elif filtered_df.columns.duplicated().any():
            print("Detected duplicate column names, removing duplicates...")
            filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
        
        # Ensure datetime is datetime type
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['datetime']):
            filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
        
        # Verify Satellite_Name column type
        if not isinstance(filtered_df['Satellite_Name'], pd.Series):
            raise ValueError(f"Satellite_Name column type error: {type(filtered_df['Satellite_Name'])}")
        
        # Parameters to filter
        filter_params = [
            'Inclination', 'Eccentricity', 'Mean_Motion',
            'Right_Ascension_of_Node', 'Argument_of_Perigee', 'Mean_Anomaly'
        ]
        
        def filter_group(group):
            """
            Apply low-pass filter to a group of data points.
            
            Args:
                group (pd.DataFrame): Group of data points for a single satellite
                
            Returns:
                pd.DataFrame: Filtered group data
            """
            if len(group) < 2:
                return group
            # Ensure sorted by time
            group = group.sort_values('datetime')
            # Calculate window size
            time_diff = group['datetime'].diff().mean().total_seconds()
            if pd.isna(time_diff) or time_diff == 0:
                window_size = 3
            else:
                window_size = max(3, int(window_hours * 3600 / time_diff))
            # Apply sliding average to each parameter
            for param in filter_params:
                if param in group.columns:
                    group[param] = group[param].rolling(
                        window=window_size, 
                        center=True,
                        min_periods=1
                    ).mean()
            return group
        
        # Group by satellite and apply filter
        print("Starting to group by satellite and apply filter...")
        filtered_df = filtered_df.groupby('Satellite_Name', group_keys=False).apply(filter_group)
        print(f"Low-pass filter completed, data shape: {filtered_df.shape}")
        
        return filtered_df.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error in low-pass filter processing: {str(e)}")
        print("Data verification information:")
        print(f"DataFrame type: {type(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data shape: {df.shape}")
        if 'Satellite_Name' in df.columns:
            print(f"Satellite_Name column type: {type(df['Satellite_Name'])}")
            print(f"First 5 Satellite_Name values: {df['Satellite_Name'].head()}")
        raise

def load_and_preprocess_data(data_dir):
    """
    Load and preprocess satellite ephemeris data using multiprocessing.
    Handles data loading, validation, and initial processing steps.
    
    Args:
        data_dir (str): Path to the directory containing ephemeris data files
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame containing validated and processed data
        
    Raises:
        ValueError: If required data files or columns are missing
        Exception: For other processing errors
    """
    print(f"Loading data from {data_dir}...")
    
    # Initialize DataFrame as None
    df = None
    
    # Define checkpoint file paths
    checkpoint_file = os.path.join(data_dir, "ephemeris_data_checkpoint.pkl")
    filtered_checkpoint_file = os.path.join(data_dir, "ephemeris_data_filtered_checkpoint.pkl")
    
    try:
        # Attempt to load filtered checkpoint file
        if os.path.exists(filtered_checkpoint_file):
            print("Found filtered checkpoint file, loading preprocessed and filtered data...")
            try:
                df = pd.read_pickle(filtered_checkpoint_file)
                print("Successfully loaded filtered checkpoint data")
                print(f"Data shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                return df
            except Exception as e:
                print(f"Error loading filtered checkpoint: {str(e)}")
                print("Continue processing original data...")
        
        # Attempt to load original checkpoint file
        if os.path.exists(checkpoint_file):
            print("Found original checkpoint file, loading preprocessed data...")
            try:
                df = pd.read_pickle(checkpoint_file)
                print("Successfully loaded original checkpoint data")
                print(f"Data shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error loading original checkpoint: {str(e)}")
                print("Continue processing original data...")
                df = None
        
        # Process original data if no valid checkpoint exists
        if df is None:
            # Get all CSV files in the directory
            data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            if not data_files:
                raise ValueError(f"No CSV files found in {data_dir}")
            
            print(f"Found {len(data_files)} CSV files to process")
            print("File list:")
            for file in data_files:
                print(f"- {os.path.basename(file)}")
            
            # Configure multiprocessing
            n_cores = max(1, cpu_count() - 1)
            print(f"Using {n_cores} CPU cores for parallel processing")
            
            try:
                # Process files in parallel
                with Pool(processes=n_cores) as pool:
                    dfs = pool.map(process_file, data_files)
                
                # Filter out None results and merge
                dfs = [df for df in dfs if df is not None]
                
                if not dfs:
                    raise ValueError("No valid data files found")
                
                print(f"Successfully processed {len(dfs)} files")
                
                # Merge all data frames
                df = pd.concat(dfs, ignore_index=True)
                print("Successfully merged all data frames")
                print(f"Merged data shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                
            except Exception as e:
                print(f"Error processing file: {str(e)}")
                raise
        
        # Remove duplicate columns
        if df.columns.duplicated().any():
            print("Detected duplicates, removing duplicates...")
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Data validation
        print("\nData verification:")
        print(f"Merged data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Verify required columns
        required_columns = ['Satellite_Name', 'Timestamp', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['Timestamp'])
        
        # Sort by time
        df = df.sort_values('datetime')
        
        # Remove duplicate satellite-time combinations
        df = df.drop_duplicates(subset=['Satellite_Name', 'datetime'])
        
        # Verify satellite data
        print("\nSatellite_Name column check:")
        print(f"Unique satellite count: {df['Satellite_Name'].nunique()}")
        print(f"First 5 satellite names: {df['Satellite_Name'].head().tolist()}")
        
        # Calculate orbital parameters
        print("\nStarting to calculate orbital parameters...")
        df = calculate_orbital_parameters(df)
        
        # Standardize units
        print("\nUnifying unit conversion...")
        # Convert inclination to degrees if needed
        if 'Inclination' in df.columns:
            if df['Inclination'].mean() < np.pi:
                df['Inclination'] = df['Inclination'] * 180 / np.pi
        
        # Convert mean motion to revs/day if needed
        if 'Mean_Motion' in df.columns:
            mean_motion_mean = df['Mean_Motion'].mean()
            if mean_motion_mean > 100:  # Check for incorrect units
                df['Mean_Motion'] = df['Mean_Motion'] * 86400 / (2 * np.pi)
        
        # Normalize RAAN to [0, 2π]
        if 'Right_Ascension_of_Node' in df.columns:
            df['Right_Ascension_of_Node'] = df['Right_Ascension_of_Node'].apply(
                lambda x: x + 2 * np.pi if x < 0 else x
            )
        
        # Save original checkpoint
        try:
            print("Starting to save original checkpoint file...")
            df.to_pickle(checkpoint_file)
            print("Original checkpoint saved successfully")
        except Exception as e:
            print(f"Warning: Unable to save original checkpoint: {str(e)}")
        
        # Apply low-pass filter
        print("\nApplying low-pass filter processing...")
        df = apply_low_pass_filter(df, window_hours=24)  # Use 24 hour window
        
        # Save filtered checkpoint
        try:
            print("Starting to save filtered checkpoint file...")
            df.to_pickle(filtered_checkpoint_file)
            print("Filtered checkpoint saved successfully")
        except Exception as e:
            print(f"Warning: Unable to save filtered checkpoint: {str(e)}")
        
        if df is None:
            raise ValueError("Data processing failed, unable to generate valid DataFrame")
        
        return df
        
    except Exception as e:
        print(f"Error loading and preprocessing data: {str(e)}")
        print("Error details:")
        import traceback
        print(traceback.format_exc())
        raise

def basic_statistics(df, output_dir):
    """
    Calculate and visualize basic statistics of orbital parameters.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save output plots and statistics
    """
    print("\n========== Basic Statistics ==========")
    
    # Calculate satellite count statistics
    unique_satellites = df['Satellite_Name'].nunique()
    print(f"Unique satellite count: {unique_satellites}")
    
    # Calculate time range statistics
    time_range = df['datetime'].max() - df['datetime'].min()
    print(f"Data time range: {time_range.days} days")
    print(f"Start date: {df['datetime'].min().strftime('%Y-%m-%d')}")
    print(f"End date: {df['datetime'].max().strftime('%Y-%m-%d')}")
    
    # Calculate orbital parameter statistics
    for param in ['Inclination', 'Eccentricity', 'Mean_Motion']:
        mean_val = df[param].mean()
        std_val = df[param].std()
        min_val = df[param].min()
        max_val = df[param].max()
        print(f"{param} - Mean: {mean_val:.6f}, Std Dev: {std_val:.6f}, Range: [{min_val:.6f}, {max_val:.6f}]")
    
    # Configure boxplot style
    boxplot_style = {
        'boxprops': dict(facecolor='lightblue', alpha=0.7),
        'medianprops': dict(color='red', linewidth=2),
        'whiskerprops': dict(color='black', linewidth=1),
        'capprops': dict(color='black', linewidth=1),
        'flierprops': dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)
    }
    
    # Generate boxplots for key parameters
    key_params = ['Inclination', 'Eccentricity', 'Mean_Motion', 
                  'RAAN', 'Argument_of_Perigee']
    
    # Prepare data for plotting
    df_plot = df.copy()
    df_plot['RAAN'] = df['Right_Ascension_of_Node']
    
    # Create subplots
    fig, axes = plt.subplots(len(key_params), 1, figsize=(12, 16))
    
    # Generate boxplots for each parameter
    for i, param in enumerate(key_params):
        if param == 'Mean_Motion':
            data = df_plot[param]
            boxplot = axes[i].boxplot(data, vert=False, patch_artist=True, **boxplot_style)
            axes[i].set_xlabel(f'{param} (revs/day)', fontsize=16, weight='bold')
        else:
            boxplot = axes[i].boxplot(df_plot[param], vert=False, patch_artist=True, **boxplot_style)
            axes[i].set_xlabel(param, fontsize=16, weight='bold')
            
        axes[i].set_title(f'{param} Distribution', fontsize=18, pad=10, weight='bold')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Calculate and display statistics
        if param == 'Mean_Motion':
            mean_val = data.mean()
            std_val = data.std()
            min_val = data.min()
            max_val = data.max()
        else:
            mean_val = df_plot[param].mean()
            std_val = df_plot[param].std()
            min_val = df_plot[param].min()
            max_val = df_plot[param].max()
            
        stats_text = f"Mean: {mean_val:.6f}\nStd Dev: {std_val:.6f}\n" \
                     f"Min: {min_val:.6f}\nMax: {max_val:.6f}"
        
        axes[i].text(0.02, 0.7, stats_text, transform=axes[i].transAxes, 
                    bbox=dict(facecolor='white', alpha=0.7), fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "orbital_parameters_boxplots.png"), dpi=300, bbox_inches='tight')
    plt.close()

def position_velocity_analysis(df, output_dir):
    """
    Analyze and visualize position and velocity distributions of satellites.
    
    Args:
        df (pd.DataFrame): DataFrame containing position and velocity data
        output_dir (str): Directory to save output plots
    """
    print("\n========== Position and Velocity Analysis ==========")
    
    # Create subplots for position and velocity analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot XY plane projection
    scatter = axes[0, 0].scatter(df['X'], df['Y'], c=df['datetime'], cmap='viridis', alpha=0.6)
    axes[0, 0].set_title('Satellite Positions (XY Projection)')
    axes[0, 0].set_xlabel('X (km)')
    axes[0, 0].set_ylabel('Y (km)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Time')
    
    # Plot XZ plane projection
    scatter = axes[0, 1].scatter(df['X'], df['Z'], c=df['datetime'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_title('Satellite Positions (XZ Projection)')
    axes[0, 1].set_xlabel('X (km)')
    axes[0, 1].set_ylabel('Z (km)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Time')
    
    # Plot velocity distribution in VX-VY plane
    scatter = axes[1, 0].scatter(df['VX'], df['VY'], c=df['datetime'], cmap='viridis', alpha=0.6)
    axes[1, 0].set_title('Satellite Velocities (VX-VY Projection)')
    axes[1, 0].set_xlabel('VX (km/s)')
    axes[1, 0].set_ylabel('VY (km/s)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Time')
    
    # Plot velocity magnitude distribution
    velocity_magnitude = np.sqrt(df['VX']**2 + df['VY']**2 + df['VZ']**2)
    sns.histplot(velocity_magnitude, ax=axes[1, 1], kde=True)
    axes[1, 1].set_title('Velocity Magnitude Distribution')
    axes[1, 1].set_xlabel('Velocity (km/s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "position_velocity_distribution.png"), dpi=300)
    plt.close()

def orbital_planes_analysis(df, output_dir):
    """
    Analyze and visualize orbital plane distributions using clustering.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save output plots
        
    Returns:
        pd.Series: Cluster assignments for each satellite
    """
    print("\n========== Orbital Plane Analysis ==========")
    
    # Create scatter plot of orbital planes
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Right_Ascension_of_Node'], df['Inclination'], 
                         alpha=0.6, c=df['Mean_Motion'], cmap='viridis')
    
    plt.colorbar(scatter, label='Mean Motion (rad/s)')
    plt.xlabel('Right Ascension of Node (radians)')
    plt.ylabel('Inclination (degrees)')
    plt.title('Satellite Orbital Plane Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "orbital_planes.png"), dpi=300)
    plt.close()
    
    # Perform K-means clustering analysis
    from sklearn.cluster import KMeans
    
    # Prepare features for clustering
    features = df[['Inclination', 'Right_Ascension_of_Node']].copy()
    
    # Calculate inertia for different cluster numbers
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow chart for cluster number selection
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('K-means Clustering Elbow Chart')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "kmeans_elbow.png"), dpi=300)
    plt.close()
    
    # Perform clustering with optimal number of clusters
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
    
    # Calculate and display satellite count by orbital plane
    plane_counts = df.groupby('orbit_cluster')['Satellite_Name'].nunique()
    print("Satellite count by orbital plane:")
    for plane, count in plane_counts.items():
        print(f"Orbital Plane {plane+1}: {count} satellites")
    
    return df['orbit_cluster']

def time_series_analysis(df, output_dir, target_satellites=None):
    """
    Perform time series analysis of orbital parameters for selected satellites.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save output plots
        target_satellites (list, optional): List of specific satellites to analyze.
            If None, analyzes the 5 satellites with most observations.
    """
    print("\n========== Time Series Analysis ==========")
    
    # Select satellites to analyze
    if target_satellites is None:
        obs_counts = df.groupby('Satellite_Name').size()
        target_satellites = obs_counts.nlargest(5).index.tolist()
    
    print(f"Analyzing {len(target_satellites)} specified satellites:")
    for sat in target_satellites:
        print(f"- {sat}")
    
    # Configure plotting style
    plt.style.use('seaborn')
    
    # Generate time series plots for each parameter
    for param in ['Mean_Motion', 'Eccentricity', 'Inclination']:
        plt.figure(figsize=(14, 8))
        
        # Plot time series for each satellite
        for sat in target_satellites:
            sat_data = df[df['Satellite_Name'] == sat].sort_values('datetime')
            if not sat_data.empty:
                plt.plot(sat_data['datetime'], sat_data[param], 
                        marker='o', linewidth=1, markersize=4, label=sat)
        
        plt.xlabel('Time')
        plt.ylabel(param)
        plt.title(f'{param} Changes Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Format y-axis for Eccentricity
        if param == 'Eccentricity':
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"time_series_{param}.png"), dpi=300)
        plt.close()

def analyze_specific_starlink_satellites(df, output_dir):
    """
    Perform detailed analysis of specific Starlink satellites.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save output plots and statistics
    """
    print("\n========== Specific Starlink Satellites Analysis ==========")
    
    # Define target satellites
    target_satellites = [
        'STARLINK-11395',
        'STARLINK-11424',
        'STARLINK-32493',
        'STARLINK-32520'
    ]
    
    print(f"Analyzing specific satellites: {target_satellites}")
    
    # Create output directory for specific satellite analysis
    specific_output_dir = os.path.join(output_dir, "specific_satellites")
    if not os.path.exists(specific_output_dir):
        os.makedirs(specific_output_dir)
    
    # Filter data for target satellites
    target_df = df[df['Satellite_Name'].isin(target_satellites)]
    
    # Generate time series analysis
    time_series_analysis(target_df, specific_output_dir, target_satellites)
    
    # Configure boxplot style
    boxplot_style = {
        'boxprops': dict(facecolor='lightblue', alpha=0.7),
        'medianprops': dict(color='red', linewidth=2),
        'whiskerprops': dict(color='black', linewidth=1),
        'capprops': dict(color='black', linewidth=1),
        'flierprops': dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)
    }
    
    # Generate boxplots for each parameter
    params = ['Inclination', 'Mean_Motion', 'Eccentricity']
    
    for param in params:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Satellite_Name', y=param, data=target_df, **boxplot_style)
        plt.title(f'{param} Distribution by Satellite')
        plt.xlabel('Satellite')
        plt.ylabel(param)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(specific_output_dir, f"{param}_boxplot.png"), dpi=300)
        plt.close()
    
    # Calculate and save statistics
    stats_df = target_df.groupby('Satellite_Name')[params].agg(['mean', 'std', 'min', 'max'])
    stats_df.to_csv(os.path.join(specific_output_dir, "satellite_statistics.csv"))
    
    # Generate position and velocity analysis for each satellite
    for sat in target_satellites:
        sat_data = target_df[target_df['Satellite_Name'] == sat]
        if not sat_data.empty:
            # Plot position in XY plane
            plt.figure(figsize=(12, 8))
            plt.scatter(sat_data['X'], sat_data['Y'], c=sat_data['datetime'], cmap='viridis', alpha=0.6)
            plt.colorbar(label='Time')
            plt.title(f'{sat} Position (XY Projection)')
            plt.xlabel('X (km)')
            plt.ylabel('Y (km)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(specific_output_dir, f"{sat}_position_xy.png"), dpi=300)
            plt.close()
            
            # Plot velocity magnitude over time
            velocity_magnitude = np.sqrt(sat_data['VX']**2 + sat_data['VY']**2 + sat_data['VZ']**2)
            plt.figure(figsize=(12, 6))
            plt.plot(sat_data['datetime'], velocity_magnitude, marker='o', linewidth=1, markersize=4)
            plt.title(f'{sat} Velocity Magnitude Over Time')
            plt.xlabel('Time')
            plt.ylabel('Velocity (km/s)')
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(specific_output_dir, f"{sat}_velocity_magnitude.png"), dpi=300)
            plt.close()
    
    print(f"Specific satellite analysis complete. Results saved to: {specific_output_dir}")

def analyze_dataset(data_dir, output_dir, dataset_name):
    """
    Perform comprehensive analysis of satellite ephemeris dataset.
    
    Args:
        data_dir (str): Directory containing ephemeris data files
        output_dir (str): Directory to save analysis results
        dataset_name (str): Name of the dataset for reporting
    """
    print(f"\n\n{'='*20} ANALYZING {dataset_name.upper()} DATASET {'='*20}\n")
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(data_dir)
        
        # Perform analysis steps
        basic_statistics(df, output_dir)
        position_velocity_analysis(df, output_dir)
        df['orbit_cluster'] = orbital_planes_analysis(df, output_dir)
        
        # Perform specific satellite analysis
        analyze_specific_starlink_satellites(df, output_dir)
        
        print(f"\nAnalysis of {dataset_name} complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        raise

def plot_ephemeris_data(df, satellite_id, output_dir):
    """
    Generate comprehensive time series plots of satellite orbital parameters.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        satellite_id (str): ID of the satellite to analyze
        output_dir (str): Directory to save output plots
    """
    print(f"\nPlotting orbital data for satellite {satellite_id}...")
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Orbital Data Time Series Analysis - Satellite {satellite_id}', 
                 fontsize=28, y=0.95)
    
    # Plot mean motion time series
    sns.lineplot(data=df, x='EPOCH', y='MEAN_MOTION', ax=axes[0,0], 
                color='blue', linewidth=2, label='Mean Motion')
    axes[0,0].set_title('Mean Motion Time Series', fontsize=26)
    axes[0,0].set_xlabel('Time', fontsize=24)
    axes[0,0].set_ylabel('Mean Motion (rev/day)', fontsize=24)
    axes[0,0].tick_params(axis='both', which='major', labelsize=20)
    axes[0,0].legend(fontsize=20)
    
    # Plot eccentricity time series
    sns.lineplot(data=df, x='EPOCH', y='ECCENTRICITY', ax=axes[0,1],
                color='red', linewidth=2, label='Eccentricity')
    axes[0,1].set_title('Eccentricity Time Series', fontsize=26)
    axes[0,1].set_xlabel('Time', fontsize=24)
    axes[0,1].set_ylabel('Eccentricity', fontsize=24)
    axes[0,1].tick_params(axis='both', which='major', labelsize=20)
    axes[0,1].legend(fontsize=20)
    
    # Plot inclination time series
    sns.lineplot(data=df, x='EPOCH', y='INCLINATION', ax=axes[1,0],
                color='green', linewidth=2, label='Inclination')
    axes[1,0].set_title('Inclination Time Series', fontsize=26)
    axes[1,0].set_xlabel('Time', fontsize=24)
    axes[1,0].set_ylabel('Inclination (degrees)', fontsize=24)
    axes[1,0].tick_params(axis='both', which='major', labelsize=20)
    axes[1,0].legend(fontsize=20)
    
    # Plot RAAN time series
    sns.lineplot(data=df, x='EPOCH', y='RA_OF_ASC_NODE', ax=axes[1,1],
                color='purple', linewidth=2, label='RAAN')
    axes[1,1].set_title('Right Ascension of Ascending Node Time Series', fontsize=26)
    axes[1,1].set_xlabel('Time', fontsize=24)
    axes[1,1].set_ylabel('RAAN (degrees)', fontsize=24)
    axes[1,1].tick_params(axis='both', which='major', labelsize=20)
    axes[1,1].legend(fontsize=20)
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'satellite_{satellite_id}_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plots saved to: {output_file}")

def main():
    """
    Main function to handle data analysis and error handling.
    Orchestrates the entire analysis workflow.
    """
    try:
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"Error: Data directory '{data_dir}' does not exist!")
            print("Please create the directory and place your ephemeris data files there.")
            return
        
        # Analyze dataset
        analyze_dataset(data_dir, output_dir, "Ephemeris Data")
        
        print("\nAll analyses complete!")
        
    except Exception as e:
        print("An error occurred during execution:")
        print(f"Error message: {str(e)}")
        print("\nPlease check:")
        print("1. Data directory exists and contains CSV files")
        print("2. CSV files have the required columns (Satellite_Name, Timestamp, X, Y, Z, VX, VY, VZ)")
        print("3. All required Python packages are installed")
        print("\nRequired packages:")
        print("- pandas")
        print("- numpy")
        print("- matplotlib")
        print("- seaborn")
        print("- scipy")
        print("- scikit-learn")

if __name__ == "__main__":
    main()