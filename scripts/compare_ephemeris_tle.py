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
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# Data paths
# Path to ephemeris data directory
# Path to TLE data file
# Output directory

# Create output directory if it does not exist

# List of target satellites

def load_ephemeris_data(data_dir):
    """
    Load and preprocess ephemeris data files from the specified directory.
    
    Args:
        data_dir (str): Directory path containing ephemeris CSV files.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame containing all ephemeris data.
    """
    print("Loading ephemeris data...")
    dfs = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            try:
                file_path = os.path.join(data_dir, file)
                print(f"Processing file: {file}")
                
                df = pd.read_csv(file_path)
                
                # Check for required columns
                required_columns = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Satellite_Name']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"Warning: File {file} is missing required columns: {missing_columns}")
                    continue
                
                # Identify time column
                time_columns = ['Timestamp', 'timestamp', 'time', 'Time', 'DATE']
                time_col = None
                for col in time_columns:
                    if col in df.columns:
                        time_col = col
                        break
                
                if time_col is None:
                    print(f"Warning: File {file} does not contain a recognized time column.")
                    continue
                
                # Rename time column to a unified name
                df = df.rename(columns={time_col: 'datetime'})
                
                # Rename velocity columns if needed
                df = df.rename(columns={
                    'Vx': 'VX',
                    'Vy': 'VY',
                    'Vz': 'VZ'
                })
                
                # Convert time column to datetime and remove timezone info
                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
                
                # Data quality checks
                print(f"\nFile {file} data quality check:")
                print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
                print(f"Number of data points: {len(df)}")
                print(f"Satellites: {df['Satellite_Name'].unique().tolist()}")
                
                # Check position and velocity columns
                for col in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
                    if col in df.columns:
                        print(f"\n{col} value range:")
                        print(f"Min: {df[col].min():.6f}")
                        print(f"Max: {df[col].max():.6f}")
                        print(f"Mean: {df[col].mean():.6f}")
                        print(f"Std: {df[col].std():.6f}")
                        
                        # Outlier detection
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                        if not outliers.empty:
                            print(f"Found {len(outliers)} outliers")
                            print(f"Outlier range: [{outliers[col].min():.6f}, {outliers[col].max():.6f}]")
                
                dfs.append(df)
                
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
    
    if not dfs:
        raise ValueError("No valid ephemeris data files found.")
    
    # Concatenate all data
    df = pd.concat(dfs, ignore_index=True)
    
    # Sort by time
    df = df.sort_values('datetime')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Satellite_Name', 'datetime'])
    
    print("\nEphemeris data loaded:")
    print(f"Total data points: {len(df)}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Satellites: {df['Satellite_Name'].unique().tolist()}")
    
    return df

def load_tle_data(data_file):
    """
    Load and preprocess TLE data from a CSV file.
    
    Args:
        data_file (str): Path to the TLE CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame containing TLE data.
    """
    print("Loading TLE data...")
    try:
        df = pd.read_csv(data_file)
        
        # Check for required columns
        required_columns = ['Satellite_Name', 'Mean_Anomaly', 'Right_Ascension_of_Node',
                          'Argument_of_Perigee', 'Eccentricity', 'Inclination', 'Mean_Motion']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"TLE data is missing required columns: {missing_columns}")
        
        # Standardize column names if needed
        column_renames = {
            '1st_Derivative_of_Mean_Motion': 'First_Derivative_of_Mean_Motion',
            '2nd_Derivative_of_Mean_Motion': 'Second_Derivative_of_Mean_Motion'
        }
        rename_dict = {k: v for k, v in column_renames.items() if k in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        # Convert time column to datetime and remove timezone info
        df['datetime'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
        
        # Data quality checks
        print("\nTLE data quality check:")
        print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Number of data points: {len(df)}")
        print(f"Satellites: {df['Satellite_Name'].unique().tolist()}")
        
        # Check orbital parameters
        for param in ['Mean_Anomaly', 'Right_Ascension_of_Node', 'Argument_of_Perigee',
                     'Eccentricity', 'Inclination', 'Mean_Motion']:
            if param in df.columns:
                print(f"\n{param} value range:")
                print(f"Min: {df[param].min():.6f}")
                print(f"Max: {df[param].max():.6f}")
                print(f"Mean: {df[param].mean():.6f}")
                print(f"Std: {df[param].std():.6f}")
                
                # Outlier detection
                q1 = df[param].quantile(0.25)
                q3 = df[param].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[param] < lower_bound) | (df[param] > upper_bound)]
                if not outliers.empty:
                    print(f"Found {len(outliers)} outliers")
                    print(f"Outlier range: [{outliers[param].min():.6f}, {outliers[param].max():.6f}]")
        
        # Sort by time
        df = df.sort_values('datetime')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Satellite_Name', 'datetime'])
        
        print("\nTLE data loaded:")
        print(f"Total data points: {len(df)}")
        print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Satellites: {df['Satellite_Name'].unique().tolist()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading TLE data: {str(e)}")
        raise

def calculate_orbital_parameters(df):
    """
    Calculate orbital parameters from position and velocity data.
    
    Args:
        df (pd.DataFrame): DataFrame containing position and velocity columns.
    
    Returns:
        pd.DataFrame: DataFrame with additional orbital parameter columns.
    """
    for idx, row in df.iterrows():
        # Position vector
        r = np.array([row['X'], row['Y'], row['Z']])
        # Velocity vector
        v = np.array([row['VX'], row['VY'], row['VZ']])
        
        # Calculate angular momentum vector
        h = np.cross(r, v)
        n = np.cross([0, 0, 1], h)
        
        # Calculate eccentricity vector
        e = np.cross(v, h) / np.linalg.norm(h)**2 - r / np.linalg.norm(r)
        
        # Calculate orbital parameters
        df.at[idx, 'Eccentricity'] = np.linalg.norm(e)
        df.at[idx, 'Inclination'] = np.arccos(h[2] / np.linalg.norm(h)) * 180 / np.pi
        df.at[idx, 'Mean_Motion'] = np.linalg.norm(h) / np.linalg.norm(r)**2
    
    return df

def plot_parameter_comparison(ephemeris_df, tle_df, parameter, output_dir):
    """
    Plot comparison of a given orbital parameter between ephemeris and TLE data for target satellites.
    
    Args:
        ephemeris_df (pd.DataFrame): Ephemeris data.
        tle_df (pd.DataFrame): TLE data.
        parameter (str): Name of the parameter to plot.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(15, 10))
    
    # Set color and marker styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, satellite in enumerate(target_satellites):
        # Plot ephemeris data
        ephem_data = ephemeris_df[ephemeris_df['Satellite_Name'] == satellite]
        if not ephem_data.empty:
            plt.plot(ephem_data['datetime'], ephem_data[parameter],
                    marker=markers[i], markersize=4, linestyle='-',
                    color=colors[i], alpha=0.7,
                    label=f'{satellite} (Ephemeris)')
        
        # Plot TLE data
        tle_data = tle_df[tle_df['Satellite_Name'] == satellite]
        if not tle_data.empty:
            plt.plot(tle_data['datetime'], tle_data[parameter],
                    marker=markers[i], markersize=4, linestyle='--',
                    color=colors[i], alpha=0.7,
                    label=f'{satellite} (TLE)')
    
    plt.title(f'{parameter} Comparison Over Time', fontsize=14, pad=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(parameter, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{parameter}_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Load data
        print("\nStarting data loading...")
        ephemeris_df = load_ephemeris_data(ephemeris_dir)
        tle_df = load_tle_data(tle_file)
        
        # Ensure timestamp types are consistent
        print("\nUnifying timestamp types...")
        ephemeris_df['datetime'] = pd.to_datetime(ephemeris_df['datetime']).dt.tz_localize(None)
        tle_df['datetime'] = pd.to_datetime(tle_df['datetime']).dt.tz_localize(None)
        
        # Ensure time ranges are consistent
        print("\nUnifying time ranges...")
        common_start = max(ephemeris_df['datetime'].min(), tle_df['datetime'].min())
        common_end = min(ephemeris_df['datetime'].max(), tle_df['datetime'].max())
        
        # Filter data within the common time range
        ephemeris_df = ephemeris_df[
            (ephemeris_df['datetime'].astype('datetime64[ns]') >= pd.Timestamp(common_start).to_datetime64()) & 
            (ephemeris_df['datetime'].astype('datetime64[ns]') <= pd.Timestamp(common_end).to_datetime64())
        ].copy()
        
        tle_df = tle_df[
            (tle_df['datetime'].astype('datetime64[ns]') >= pd.Timestamp(common_start).to_datetime64()) & 
            (tle_df['datetime'].astype('datetime64[ns]') <= pd.Timestamp(common_end).to_datetime64())
        ].copy()
        
        print(f"Unified time range: {common_start} to {common_end}")
        print(f"Ephemeris data points: {len(ephemeris_df)}")
        print(f"TLE data points: {len(tle_df)}")
        
        # Calculate orbital parameters for ephemeris data
        print("\nCalculating orbital parameters for ephemeris data...")
        ephemeris_df = calculate_orbital_parameters(ephemeris_df)
        
        # Plot parameter comparisons
        parameters = ['Inclination', 'Mean_Motion', 'Eccentricity', 'Right_Ascension_of_Node']
        for param in parameters:
            print(f"\nGenerating {param} comparison plot...")
            plot_parameter_comparison(ephemeris_df, tle_df, param, output_dir)
        
        print("\nAll plots generated!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 