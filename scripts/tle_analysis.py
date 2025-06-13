#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Satellite Orbital Analysis Module

This module provides comprehensive tools for analyzing satellite orbital parameters,
including TLE (Two-Line Element) data processing, orbital parameter calculations,
and various analysis functions for satellite motion characteristics.

Key Features:
- Unit conversion and standardization
- Orbital parameter calculations
- Statistical analysis and visualization
- Time series analysis
- Orbital plane analysis
- Decay analysis
- Correlation analysis
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
from functools import lru_cache
from matplotlib.lines import Line2D
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates import GCRS
from astropy.time import Time
import json
from sklearn.cluster import KMeans

class UnitConverter:
    """
    Utility class for standardizing orbital parameter units and normalizing angles.
    Provides methods for converting between different units and normalizing angular parameters.
    """
    
    @staticmethod
    def convert_inclination(df):
        """
        Convert inclination from radians to degrees.
        
        Args:
            df (pd.DataFrame): DataFrame containing inclination in radians
            
        Returns:
            pd.DataFrame: DataFrame with inclination converted to degrees
        """
        if 'Inclination' in df.columns:
            # 检查是否已经是度数
            if df['Inclination'].mean() < np.pi:
                df['Inclination'] = df['Inclination'] * 180 / np.pi 
        return df
    
    @staticmethod
    def convert_mean_motion(df):
        """
        Convert mean motion from rad/min to revs/day.
        
        Args:
            df (pd.DataFrame): DataFrame containing mean motion in rad/min
            
        Returns:
            pd.DataFrame: DataFrame with mean motion converted to revs/day
        """
        if 'Mean_Motion' in df.columns:
            # 检查是否是rad/min单位（处理脚本转换后的格式）
            mean_motion_mean = df['Mean_Motion'].mean()
            if 0.01 < mean_motion_mean < 1:  # 如果值在0.01-1之间，说明是rad/min
                # 从rad/min转换为revs/day: rad/min * 60 min/hr * 24 hr/day / (2π rad/rev)
                df['Mean_Motion'] = df['Mean_Motion'] * 60 * 24 / (2 * np.pi)
        return df
    
    @staticmethod
    def convert_mean_motion_to_rad_s(df):
        """将平均运动从rad/min转换为rad/s（如果输入是处理脚本的格式）"""
        if 'Mean_Motion' in df.columns:
            # 检查是否是rad/min单位
            mean_motion_mean = df['Mean_Motion'].mean()
            if 0.01 < mean_motion_mean < 1:  # 如果值在0.01-1之间，说明是rad/min
                # 从rad/min转换为rad/s
                df['Mean_Motion'] = df['Mean_Motion'] / 60
            elif 10 < mean_motion_mean < 20:  # 如果值在10-20之间，说明是revs/day
                # 从revs/day转换为rad/s
                df['Mean_Motion'] = df['Mean_Motion'] * 2 * np.pi / 86400
        return df
    
    @staticmethod
    def normalize_raan(df):
        """
        Normalize Right Ascension of Node to [0, 2π].
        
        Args:
            df (pd.DataFrame): DataFrame containing RAAN values
            
        Returns:
            pd.DataFrame: DataFrame with normalized RAAN values
        """
        if 'Right_Ascension_of_Node' in df.columns:
            df['Right_Ascension_of_Node'] = df['Right_Ascension_of_Node'] % (2 * np.pi)
        return df
    
    @staticmethod
    def normalize_argument_of_perigee(df):
        """
        Normalize Argument of Perigee to [0, 2π].
        
        Args:
            df (pd.DataFrame): DataFrame containing argument of perigee values
            
        Returns:
            pd.DataFrame: DataFrame with normalized argument of perigee values
        """
        if 'Argument_of_Perigee' in df.columns:
            df['Argument_of_Perigee'] = df['Argument_of_Perigee'] % (2 * np.pi)
        return df
    
    @staticmethod
    def normalize_mean_anomaly(df):
        """
        Normalize Mean Anomaly to [0, 2π].
        
        Args:
            df (pd.DataFrame): DataFrame containing mean anomaly values
            
        Returns:
            pd.DataFrame: DataFrame with normalized mean anomaly values
        """
        if 'Mean_Anomaly' in df.columns:
            df['Mean_Anomaly'] = df['Mean_Anomaly'] % (2 * np.pi)
        return df
    
    @staticmethod
    def standardize_units(df):
        """
        Standardize all orbital parameter units in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing orbital parameters
            
        Returns:
            pd.DataFrame: DataFrame with standardized units
        """
        df = UnitConverter.convert_inclination(df)
        df = UnitConverter.convert_mean_motion(df)  # 转换为revs/day
        df = UnitConverter.normalize_raan(df)
        df = UnitConverter.normalize_argument_of_perigee(df)
        df = UnitConverter.normalize_mean_anomaly(df)
        return df

class PlotStyle:
    """
    Utility class for managing plot styles and configurations.
    Provides methods for setting up global plot styles and obtaining specific style configurations.
    """
    
    @staticmethod
    def setup_global_style():
        """
        Set up global plot style configurations.
        Configures matplotlib and seaborn styles for consistent visualization.
        """
        plt.style.use('seaborn')
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
    
    @staticmethod
    def get_boxplot_style():
        """
        Get style configuration for boxplots.
        
        Returns:
            dict: Dictionary containing boxplot style parameters
        """
        return {
            'width': 0.8,
            'fliersize': 5,
            'linewidth': 1.5,
            'palette': 'Set3'
        }
    
    @staticmethod
    def get_satellite_colors():
        """获取卫星颜色映射"""
        return {
            'STARLINK-11395': '#1f77b4',
            'STARLINK-11424': '#ff7f0e',
            'STARLINK-32493': '#2ca02c',
            'STARLINK-32520': '#d62728',
            'STARLINK-6363': '#9467bd'
        }
    
    @staticmethod
    def format_plot(ax, title, xlabel, ylabel, rotate_x=True):
        """统一格式化图表"""
        ax.set_title(title, fontsize=24, pad=15, weight='bold')
        ax.set_xlabel(xlabel, fontsize=24, weight='bold')
        ax.set_ylabel(ylabel, fontsize=24, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        if rotate_x:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

class OrbitalCalculator:
    """
    Utility class for calculating orbital parameters.
    Provides methods for calculating various orbital parameters from position and velocity vectors.
    """
    
    # Earth's gravitational parameter (km³/s²)
    MU = 398600.4418
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def calculate_orbital_parameters_cached(r_vec_tuple, v_vec_tuple):
        """
        Calculate orbital parameters from position and velocity vectors (cached version).
        
        Args:
            r_vec_tuple (tuple): Position vector as a tuple (x, y, z)
            v_vec_tuple (tuple): Velocity vector as a tuple (vx, vy, vz)
            
        Returns:
            dict: Dictionary containing calculated orbital parameters
        """
        r_vec = np.array(r_vec_tuple)
        v_vec = np.array(v_vec_tuple)
        
        # Calculate specific angular momentum
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        
        # Calculate eccentricity vector
        v2 = np.dot(v_vec, v_vec)
        r = np.linalg.norm(r_vec)
        e_vec = ((v2 - OrbitalCalculator.MU/r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / OrbitalCalculator.MU
        e = np.linalg.norm(e_vec)
        
        # Calculate semi-major axis
        a = -OrbitalCalculator.MU / (2 * (v2/2 - OrbitalCalculator.MU/r))
        
        # Calculate inclination
        i = np.arccos(h_vec[2] / h)
        
        # Calculate right ascension of node
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)
        if n != 0:
            raan = np.arccos(n_vec[0] / n)
            if n_vec[1] < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0
        
        # Calculate argument of perigee
        if n != 0:
            argp = np.arccos(np.dot(n_vec, e_vec) / (n * e))
            if e_vec[2] < 0:
                argp = 2 * np.pi - argp
        else:
            argp = 0
        
        # Calculate true anomaly
        ta = np.arccos(np.dot(e_vec, r_vec) / (e * r))
        if np.dot(r_vec, v_vec) < 0:
            ta = 2 * np.pi - ta
        
        # Calculate mean anomaly
        E = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(ta/2))
        M = E - e * np.sin(E)
        
        # Calculate mean motion
        n = np.sqrt(OrbitalCalculator.MU / (a**3))
        
        return {
            'Semi_Major_Axis': a,
            'Eccentricity': e,
            'Inclination': i,
            'Right_Ascension_of_Node': raan,
            'Argument_of_Perigee': argp,
            'True_Anomaly': ta,
            'Mean_Anomaly': M,
            'Mean_Motion': n
        }
    
    @staticmethod
    def calculate_orbital_parameters(r_vec, v_vec):
        """
        Calculate orbital parameters from position and velocity vectors.
        
        Args:
            r_vec (np.ndarray): Position vector
            v_vec (np.ndarray): Velocity vector
            
        Returns:
            dict: Dictionary containing calculated orbital parameters
        """
        return OrbitalCalculator.calculate_orbital_parameters_cached(
            tuple(r_vec), tuple(v_vec)
        )
    
    @staticmethod
    def calculate_position_velocity(orbital_params):
        """
        Calculate position and velocity vectors from orbital parameters.
        
        Args:
            orbital_params (dict): Dictionary containing orbital parameters
            
        Returns:
            tuple: (position_vector, velocity_vector)
        """
        a = orbital_params['Semi_Major_Axis']
        e = orbital_params['Eccentricity']
        i = orbital_params['Inclination']
        raan = orbital_params['Right_Ascension_of_Node']
        argp = orbital_params['Argument_of_Perigee']
        ta = orbital_params['True_Anomaly']
        
        # Calculate radius
        r = a * (1 - e**2) / (1 + e * np.cos(ta))
        
        # Calculate position in orbital plane
        x = r * np.cos(ta)
        y = r * np.sin(ta)
        z = 0
        
        # Calculate velocity in orbital plane
        h = np.sqrt(OrbitalCalculator.MU * a * (1 - e**2))
        vx = -OrbitalCalculator.MU/h * np.sin(ta)
        vy = OrbitalCalculator.MU/h * (e + np.cos(ta))
        vz = 0
        
        # Transform to inertial frame
        r_vec = np.array([x, y, z])
        v_vec = np.array([vx, vy, vz])
        
        # Rotation matrices
        R3_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])
        
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])
        
        R3_argp = np.array([
            [np.cos(argp), -np.sin(argp), 0],
            [np.sin(argp), np.cos(argp), 0],
            [0, 0, 1]
        ])
        
        # Apply rotations
        Q = R3_raan @ R1_i @ R3_argp
        r_vec = Q @ r_vec
        v_vec = Q @ v_vec
        
        return r_vec, v_vec

# 设置全局字体样式
PlotStyle.setup_global_style()

# Data paths
data_dir = "../data/TLE/merged_data"
starlink_file = os.path.join(data_dir, "sl_merged_tle_data.csv")
all_satellites_file = os.path.join(data_dir, "ori_merged_tle_data.csv")
starlink_output_dir = os.path.join(data_dir, "starlink_analysis_results")
all_output_dir = os.path.join(data_dir, "all_satellites_analysis_results")

# Create output directories
for directory in [starlink_output_dir, all_output_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_data(df):
    """
    Preprocess the orbital parameter data by standardizing units and normalizing angles.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with standardized units
    """
    # Standardize units and normalize angles
    df = UnitConverter.standardize_units(df)
    
    # Convert mean motion to radians per second
    df = UnitConverter.convert_mean_motion_to_rad_s(df)
    
    return df

def load_and_preprocess_data(data_file):
    """
    Load and preprocess TLE data from a CSV file.
    
    Args:
        data_file (str): Path to the CSV file containing TLE data
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with standardized units
        
    Raises:
        FileNotFoundError: If the data file does not exist
        ValueError: If the data file is empty or has invalid format
    """
    # Load data from CSV file
    df = pd.read_csv(data_file)
    
    # Verify data integrity
    if df.empty:
        raise ValueError("Data file is empty")
    
    # Check for required columns
    required_columns = ['Satellite', 'Epoch', 'Eccentricity', 'Inclination',
                       'Right_Ascension_of_Node', 'Argument_of_Perigee',
                       'Mean_Anomaly', 'Mean_Motion']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Preprocess data
    df = preprocess_data(df)
    
    return df

class StatisticsCache:
    """
    Cache for storing statistical calculations to improve performance.
    Provides methods for calculating and retrieving statistics for orbital parameters.
    """
    
    def __init__(self):
        """Initialize the statistics cache."""
        self.cache = {}
    
    def get_statistics(self, df, param):
        """
        Calculate or retrieve statistics for a given parameter.
        
        Args:
            df (pd.DataFrame): DataFrame containing the parameter data
            param (str): Name of the parameter to calculate statistics for
            
        Returns:
            dict: Dictionary containing statistical measures
        """
        if param not in self.cache:
            self.cache[param] = {
                'mean': df[param].mean(),
                'std': df[param].std(),
                'min': df[param].min(),
                'max': df[param].max(),
                'median': df[param].median()
            }
        return self.cache[param]
    
    def clear(self):
        """Clear the statistics cache."""
        self.cache.clear()

def basic_statistics(df, output_dir):
    """
    Calculate and visualize basic statistics of orbital parameters.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save the output plots
        
    Returns:
        dict: Dictionary containing statistical measures for each parameter
    """
    # Initialize statistics cache
    stats_cache = StatisticsCache()
    
    # Calculate statistics for each parameter
    parameters = ['Eccentricity', 'Inclination', 'Right_Ascension_of_Node',
                 'Argument_of_Perigee', 'Mean_Anomaly', 'Mean_Motion']
    
    stats = {}
    for param in parameters:
        stats[param] = stats_cache.get_statistics(df, param)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate boxplots for each parameter
    plt.figure(figsize=(15, 10))
    df[parameters].boxplot()
    plt.title('Distribution of Orbital Parameters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'orbital_parameters_boxplot.png'))
    plt.close()
    
    # Generate histograms for each parameter
    for param in parameters:
        plt.figure(figsize=(12, 8))
        sns.histplot(data=df, x=param, bins=50)
        plt.title(f'Distribution of {param}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{param}_histogram.png'))
        plt.close()
    
    return stats

def orbital_parameters_distribution(df, output_dir):
    """
    Analyze and visualize the distribution of orbital parameters.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save the output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate scatter plots for parameter pairs
    parameters = ['Eccentricity', 'Inclination', 'Right_Ascension_of_Node',
                 'Argument_of_Perigee', 'Mean_Anomaly', 'Mean_Motion']
    
    for i, param1 in enumerate(parameters):
        for param2 in parameters[i+1:]:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=df, x=param1, y=param2, alpha=0.5)
            plt.title(f'{param1} vs {param2}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param1}_vs_{param2}_scatter.png'))
            plt.close()
    
    # Generate correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[parameters].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Orbital Parameters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

def orbital_planes_analysis(df, output_dir):
    """
    Analyze and visualize orbital plane distributions using clustering.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save the output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract orbital plane parameters
    X = df[['Right_Ascension_of_Node', 'Inclination']].values
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Generate scatter plot of orbital planes
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Right_Ascension_of_Node', y='Inclination',
                   hue='Cluster', palette='deep')
    plt.title('Orbital Plane Distribution')
    plt.xlabel('Right Ascension of Node (rad)')
    plt.ylabel('Inclination (deg)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'orbital_planes_clustering.png'))
    plt.close()
    
    # Generate cluster statistics
    cluster_stats = df.groupby('Cluster')[['Right_Ascension_of_Node', 'Inclination']].agg(['mean', 'std'])
    cluster_stats.to_csv(os.path.join(output_dir, 'orbital_planes_cluster_stats.csv'))

def time_series_analysis(df, output_dir, target_satellites=None):
    """
    Perform time series analysis of orbital parameters for selected satellites.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save the output plots
        target_satellites (list, optional): List of satellite names to analyze
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for target satellites if specified
    if target_satellites:
        df = df[df['Satellite'].isin(target_satellites)]
    
    # Convert epoch to datetime
    df['Epoch'] = pd.to_datetime(df['Epoch'])
    
    # Generate time series plots for each parameter
    parameters = ['Eccentricity', 'Inclination', 'Right_Ascension_of_Node',
                 'Argument_of_Perigee', 'Mean_Anomaly', 'Mean_Motion']
    
    for param in parameters:
        plt.figure(figsize=(15, 8))
        for satellite in df['Satellite'].unique():
            satellite_data = df[df['Satellite'] == satellite]
            plt.plot(satellite_data['Epoch'], satellite_data[param],
                    label=satellite, alpha=0.7)
        
        plt.title(f'{param} Time Series')
        plt.xlabel('Time')
        plt.ylabel(param)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{param}_time_series.png'))
        plt.close()

def decay_analysis(df, output_dir):
    """
    Analyze orbital decay patterns and trends.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save the output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert epoch to datetime
    df['Epoch'] = pd.to_datetime(df['Epoch'])
    
    # Calculate semi-major axis from mean motion
    df['Semi_Major_Axis'] = (OrbitalCalculator.MU / (df['Mean_Motion']**2))**(1/3)
    
    # Generate decay plots for each satellite
    for satellite in df['Satellite'].unique():
        satellite_data = df[df['Satellite'] == satellite].sort_values('Epoch')
        
        plt.figure(figsize=(15, 8))
        plt.plot(satellite_data['Epoch'], satellite_data['Semi_Major_Axis'],
                label='Semi-Major Axis')
        plt.title(f'Orbital Decay Analysis - {satellite}')
        plt.xlabel('Time')
        plt.ylabel('Semi-Major Axis (km)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{satellite}_decay_analysis.png'))
        plt.close()

def correlation_analysis(df, output_dir):
    """
    Analyze correlations between orbital parameters.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save the output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate correlation matrix
    parameters = ['Eccentricity', 'Inclination', 'Right_Ascension_of_Node',
                 'Argument_of_Perigee', 'Mean_Anomaly', 'Mean_Motion']
    corr_matrix = df[parameters].corr()
    
    # Generate correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Orbital Parameters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # Save correlation matrix to CSV
    corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))

def analyze_specific_starlink_satellites(df, output_dir):
    """
    Perform detailed analysis of specific Starlink satellites.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        output_dir (str): Directory to save the output plots
    """
    # Create output directory for specific satellite analysis
    specific_dir = os.path.join(output_dir, 'specific_satellites')
    os.makedirs(specific_dir, exist_ok=True)
    
    # Define target satellites
    target_satellites = ['STARLINK-11395', 'STARLINK-11424', 'STARLINK-32493',
                        'STARLINK-32520', 'STARLINK-6363']
    
    # Filter data for target satellites
    target_df = df[df['Satellite'].isin(target_satellites)]
    
    # Generate boxplots for each parameter
    parameters = ['Eccentricity', 'Inclination', 'Right_Ascension_of_Node',
                 'Argument_of_Perigee', 'Mean_Anomaly', 'Mean_Motion']
    
    for param in parameters:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=target_df, x='Satellite', y=param)
        plt.title(f'{param} Distribution by Satellite')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(specific_dir, f'{param}_boxplot.png'))
        plt.close()
    
    # Generate time series analysis
    time_series_analysis(target_df, specific_dir, target_satellites)
    
    # Calculate and save statistics
    stats = {}
    for satellite in target_satellites:
        satellite_data = target_df[target_df['Satellite'] == satellite]
        stats[satellite] = {
            param: {
                'mean': satellite_data[param].mean(),
                'std': satellite_data[param].std(),
                'min': satellite_data[param].min(),
                'max': satellite_data[param].max()
            }
            for param in parameters
        }
    
    # Save statistics to CSV
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.to_csv(os.path.join(specific_dir, 'satellite_statistics.csv'))

def convert_tle_to_instantaneous(tle_df):
    """
    Convert TLE data to instantaneous orbital parameters.
    
    Args:
        tle_df (pd.DataFrame): DataFrame containing TLE data
        
    Returns:
        pd.DataFrame: DataFrame with instantaneous orbital parameters
    """
    # Create a copy of the input DataFrame
    df = tle_df.copy()
    
    # Convert mean motion to radians per second
    df['Mean_Motion'] = df['Mean_Motion'] * 2 * np.pi / 86400
    
    # Calculate semi-major axis
    df['Semi_Major_Axis'] = (OrbitalCalculator.MU / (df['Mean_Motion']**2))**(1/3)
    
    # Convert inclination to radians
    df['Inclination'] = df['Inclination'] * np.pi / 180
    
    return df

def calculate_orbital_parameters(df):
    """
    Calculate orbital parameters from position and velocity data.
    
    Args:
        df (pd.DataFrame): DataFrame containing position and velocity data
        
    Returns:
        pd.DataFrame: DataFrame with calculated orbital parameters
    """
    # Initialize lists to store calculated parameters
    eccentricity = []
    inclination = []
    raan = []
    argp = []
    mean_anomaly = []
    mean_motion = []
    
    # Calculate parameters for each row
    for _, row in df.iterrows():
        r_vec = np.array([row['X'], row['Y'], row['Z']])
        v_vec = np.array([row['VX'], row['VY'], row['VZ']])
        
        params = OrbitalCalculator.calculate_orbital_parameters(r_vec, v_vec)
        
        eccentricity.append(params['Eccentricity'])
        inclination.append(params['Inclination'])
        raan.append(params['Right_Ascension_of_Node'])
        argp.append(params['Argument_of_Perigee'])
        mean_anomaly.append(params['Mean_Anomaly'])
        mean_motion.append(params['Mean_Motion'])
    
    # Add calculated parameters to DataFrame
    df['Eccentricity'] = eccentricity
    df['Inclination'] = inclination
    df['Right_Ascension_of_Node'] = raan
    df['Argument_of_Perigee'] = argp
    df['Mean_Anomaly'] = mean_anomaly
    df['Mean_Motion'] = mean_motion
    
    return df

def check_data_quality(df, data_type):
    """
    Check the quality of orbital parameter data.
    
    Args:
        df (pd.DataFrame): DataFrame containing orbital parameters
        data_type (str): Type of data ('TLE' or 'Ephemeris')
        
    Returns:
        dict: Dictionary containing data quality metrics
    """
    # Initialize quality metrics
    quality_metrics = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'parameter_ranges': {}
    }
    
    # Check parameter ranges
    parameters = ['Eccentricity', 'Inclination', 'Right_Ascension_of_Node',
                 'Argument_of_Perigee', 'Mean_Anomaly', 'Mean_Motion']
    
    for param in parameters:
        quality_metrics['parameter_ranges'][param] = {
            'min': df[param].min(),
            'max': df[param].max(),
            'mean': df[param].mean(),
            'std': df[param].std()
        }
    
    # Check for outliers
    for param in parameters:
        z_scores = np.abs((df[param] - df[param].mean()) / df[param].std())
        quality_metrics['parameter_ranges'][param]['outliers'] = (z_scores > 3).sum()
    
    return quality_metrics

def compare_parameters(tle_df, eph_df, param, target_satellites):
    """
    Compare orbital parameters between TLE and ephemeris data.
    
    Args:
        tle_df (pd.DataFrame): DataFrame containing TLE data
        eph_df (pd.DataFrame): DataFrame containing ephemeris data
        param (str): Parameter to compare
        target_satellites (list): List of satellite names to compare
        
    Returns:
        pd.DataFrame: DataFrame containing comparison results
    """
    # Filter data for target satellites
    tle_data = tle_df[tle_df['Satellite'].isin(target_satellites)]
    eph_data = eph_df[eph_df['Satellite'].isin(target_satellites)]
    
    # Merge data on satellite and epoch
    merged_data = pd.merge(tle_data, eph_data, on=['Satellite', 'Epoch'],
                          suffixes=('_tle', '_eph'))
    
    # Calculate differences
    merged_data[f'{param}_diff'] = merged_data[f'{param}_tle'] - merged_data[f'{param}_eph']
    merged_data[f'{param}_diff_pct'] = (merged_data[f'{param}_diff'] / merged_data[f'{param}_eph']) * 100
    
    return merged_data

def plot_parameter_comparison(tle_df, eph_df, param, unit, output_dir, target_satellites):
    """
    Generate comparison plots for orbital parameters.
    
    Args:
        tle_df (pd.DataFrame): DataFrame containing TLE data
        eph_df (pd.DataFrame): DataFrame containing ephemeris data
        param (str): Parameter to compare
        unit (str): Unit of the parameter
        output_dir (str): Directory to save the output plots
        target_satellites (list): List of satellite names to compare
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get comparison data
    comparison_data = compare_parameters(tle_df, eph_df, param, target_satellites)
    
    # Generate time series comparison plot
    plt.figure(figsize=(15, 8))
    for satellite in target_satellites:
        satellite_data = comparison_data[comparison_data['Satellite'] == satellite]
        plt.plot(satellite_data['Epoch'], satellite_data[f'{param}_tle'],
                label=f'{satellite} (TLE)', alpha=0.7)
        plt.plot(satellite_data['Epoch'], satellite_data[f'{param}_eph'],
                label=f'{satellite} (Ephemeris)', alpha=0.7, linestyle='--')
    
    plt.title(f'{param} Comparison')
    plt.xlabel('Time')
    plt.ylabel(f'{param} ({unit})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{param}_comparison.png'))
    plt.close()
    
    # Generate difference plot
    plt.figure(figsize=(15, 8))
    for satellite in target_satellites:
        satellite_data = comparison_data[comparison_data['Satellite'] == satellite]
        plt.plot(satellite_data['Epoch'], satellite_data[f'{param}_diff'],
                label=satellite, alpha=0.7)
    
    plt.title(f'{param} Difference (TLE - Ephemeris)')
    plt.xlabel('Time')
    plt.ylabel(f'Difference ({unit})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{param}_difference.png'))
    plt.close()
    
    # Generate percentage difference plot
    plt.figure(figsize=(15, 8))
    for satellite in target_satellites:
        satellite_data = comparison_data[comparison_data['Satellite'] == satellite]
        plt.plot(satellite_data['Epoch'], satellite_data[f'{param}_diff_pct'],
                label=satellite, alpha=0.7)
    
    plt.title(f'{param} Percentage Difference')
    plt.xlabel('Time')
    plt.ylabel('Percentage Difference (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{param}_percentage_difference.png'))
    plt.close()

def compare_tle_ephemeris_analysis(tle_df, ephemeris_df, output_dir, target_satellites):
    """
    Perform comprehensive comparison analysis between TLE and ephemeris data.
    
    Args:
        tle_df (pd.DataFrame): DataFrame containing TLE data
        ephemeris_df (pd.DataFrame): DataFrame containing ephemeris data
        output_dir (str): Directory to save the output plots
        target_satellites (list): List of satellite names to analyze
    """
    # Create output directory for comparison analysis
    comparison_dir = os.path.join(output_dir, 'tle_ephemeris_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Define parameters to compare
    parameters = {
        'Eccentricity': 'dimensionless',
        'Inclination': 'degrees',
        'Right_Ascension_of_Node': 'degrees',
        'Argument_of_Perigee': 'degrees',
        'Mean_Anomaly': 'degrees',
        'Mean_Motion': 'rad/s'
    }
    
    # Generate comparison plots for each parameter
    for param, unit in parameters.items():
        plot_parameter_comparison(tle_df, ephemeris_df, param, unit,
                                comparison_dir, target_satellites)
    
    # Calculate and save comparison statistics
    stats = {}
    for param in parameters:
        comparison_data = compare_parameters(tle_df, ephemeris_df, param, target_satellites)
        stats[param] = {
            'mean_diff': comparison_data[f'{param}_diff'].mean(),
            'std_diff': comparison_data[f'{param}_diff'].std(),
            'max_diff': comparison_data[f'{param}_diff'].max(),
            'min_diff': comparison_data[f'{param}_diff'].min(),
            'mean_pct_diff': comparison_data[f'{param}_diff_pct'].mean(),
            'std_pct_diff': comparison_data[f'{param}_diff_pct'].std()
        }
    
    # Save statistics to CSV
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.to_csv(os.path.join(comparison_dir, 'comparison_statistics.csv'))

def analyze_dataset(data_file, output_dir, dataset_name):
    """
    Perform comprehensive analysis of satellite orbital dataset.
    
    Args:
        data_file (str): Path to the data file
        output_dir (str): Directory to save the output plots
        dataset_name (str): Name of the dataset for output organization
    """
    # Create output directory for dataset analysis
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data(data_file)
    
    # Check data quality
    quality_metrics = check_data_quality(df, dataset_name)
    
    # Save quality metrics
    with open(os.path.join(dataset_dir, 'data_quality_metrics.json'), 'w') as f:
        json.dump(quality_metrics, f, indent=4)
    
    # Perform basic statistics analysis
    basic_statistics(df, dataset_dir)
    
    # Perform orbital parameters distribution analysis
    orbital_parameters_distribution(df, dataset_dir)
    
    # Perform orbital planes analysis
    orbital_planes_analysis(df, dataset_dir)
    
    # Perform time series analysis
    time_series_analysis(df, dataset_dir)
    
    # Perform decay analysis
    decay_analysis(df, dataset_dir)
    
    # Perform correlation analysis
    correlation_analysis(df, dataset_dir)
    
    # Perform specific satellite analysis
    analyze_specific_starlink_satellites(df, dataset_dir)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Satellite Orbital Analysis Tool')
    parser.add_argument('--data_file', required=True, help='Path to the data file')
    parser.add_argument('--output_dir', required=True, help='Output directory for analysis results')
    parser.add_argument('--dataset_name', required=True, help='Name of the dataset')
    return parser.parse_args()

def main():
    """
    Main function to run the orbital analysis.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up global plot style
    PlotStyle.setup_global_style()
    
    # Perform dataset analysis
    analyze_dataset(args.data_file, args.output_dir, args.dataset_name)

if __name__ == '__main__':
    main()