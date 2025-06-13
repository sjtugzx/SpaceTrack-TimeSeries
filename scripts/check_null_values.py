#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

def check_null_values(data_dir):
    """
    Check for null values and column existence in all CSV files within the specified directory.
    
    Args:
        data_dir (str): Path to the data directory.
    """
    print(f"Checking directory: {data_dir}")
    
    # Get all CSV files
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not data_files:
        print(f"Error: No CSV files found in {data_dir}")
        return
    
    print(f"\nFound {len(data_files)} CSV files")
    
    # Define required columns
    required_columns = ['Satellite_Name', 'Timestamp', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
    
    # Check each file
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        print(f"\nChecking file: {file}")
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns: {missing_columns}")
            
            # Check null values in each column
            null_counts = df.isnull().sum()
            total_rows = len(df)
            
            # Print null value statistics
            print(f"Total rows: {total_rows}")
            print("\nNull value statistics:")
            for column, null_count in null_counts.items():
                if null_count > 0:
                    percentage = (null_count / total_rows) * 100
                    print(f"{column}: {null_count} null values ({percentage:.2f}%)")
            
            # Check for completely empty columns
            empty_columns = df.columns[df.isnull().all()].tolist()
            if empty_columns:
                print(f"\nWarning: The following columns are completely empty: {empty_columns}")
            
            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                print(f"\nWarning: Found {duplicates} duplicate rows")
            
            # Display column names
            print("\nColumns in the file:")
            print(df.columns.tolist())
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            continue

def main():
    """
    Main function to check null values in the specified data directory.
    """
    data_dir = "../data/ephemeris"
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist!")
        return
    
    # Check null values
    check_null_values(data_dir)

if __name__ == "__main__":
    main() 