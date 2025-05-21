"""
Ephemeris Data Processing Pipeline
This script provides a complete pipeline for processing ephemeris data:
1. Parse raw MEME format ephemeris files
2. Merge CSV files for the same satellite
3. Process and standardize the merged files
"""

import os
import pandas as pd
import glob
from datetime import datetime
import pytz
from astropy.time import Time

def parse_datetime(dt_str):
    """
    Parse datetime string to timestamp (seconds)
    
    Args:
        dt_str (str): Datetime string in format "YYYY-MM-DD HH:MM:SS UTC"
    
    Returns:
        float: Timestamp in seconds
    """
    try:
        # Remove any additional information
        dt_str = dt_str.split(' ephemeris_stop:')[0].strip()
        # Parse datetime string
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S UTC')
        # Set to UTC timezone
        dt = pytz.UTC.localize(dt)
        # Convert to timestamp (seconds)
        return dt.timestamp()
    except Exception as e:
        raise ValueError(f"Unable to parse time string '{dt_str}': {str(e)}")

def timestamp_to_iso(timestamp):
    """
    Convert Unix timestamp to ISO format datetime string
    
    Args:
        timestamp (float): Unix timestamp in seconds
    
    Returns:
        str: ISO format datetime string (YYYY-MM-DD HH:MM:SS UTC)
    """
    dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
    return dt.strftime('%Y-%m-%d %H:%M:%S UTC')

def convert_jd_to_iso(jd):
    """
    Convert Julian Date to ISO format datetime string
    
    Args:
        jd (float): Julian Date
    
    Returns:
        str: ISO format datetime string
    """
    try:
        t = Time(jd, format='jd')
        return t.iso
    except:
        return jd

def parse_meme_ephemeris_full(filepath, output_dir='../data/ephemeris/csv'):
    """
    Parse MEME format ephemeris data files and save as CSV format
    
    Args:
        filepath (str): Path to MEME format ephemeris data file
        output_dir (str): Output directory for CSV files
    
    Returns:
        None: Saves parsed results as CSV files
    """
    # Extract filename from path
    filename = os.path.basename(filepath)
    name_parts = filename.split('_')

    # Extract configuration from filename
    # Format: file_type_satellite_catalog_satellite_name_other_info
    file_type = name_parts[0] if len(name_parts) > 0 else ''
    satellite_catalog = name_parts[1] if len(name_parts) > 1 else ''
    satellite_name = name_parts[2] if len(name_parts) > 2 else ''

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build output file path: keep original filename but change extension to .csv
    output_filename = os.path.splitext(filename)[0] + '.csv'
    output_path = os.path.join(output_dir, output_filename)

    # Read file content
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse file header information
    start_time = None
    step_size = None
    for line in lines[:10]:  # Only check first 10 lines
        if 'ephemeris_start:' in line:
            # Extract start time
            start_time_str = line.split('ephemeris_start:')[1].split(' ephemeris_stop:')[0].strip()
            start_time = parse_datetime(start_time_str)
            # Extract time interval
            if 'step_size:' in line:
                step_size = float(line.split('step_size:')[1].strip())
        if start_time is not None and step_size is not None:
            break
    
    if start_time is None or step_size is None:
        raise ValueError(f"Unable to find required start time or time interval in file {filepath}")

    # Store parsed records
    records = []
    i = 0
    data_index = 0  # Index for calculating timestamp
    while i < len(lines):
        line = lines[i].strip()
        tokens = line.split()

        # Check if it's a data row (first token is a number)
        if len(tokens) >= 7 and tokens[0].replace('.', '', 1).isdigit():
            # Calculate current data timestamp
            current_timestamp = start_time + (data_index * step_size)
            
            # Create basic record with timestamp and position/velocity information
            record = {
                'file_type': file_type,
                'satellite_catalog': satellite_catalog,
                'satellite_name': satellite_name,
                'ephemeris_source': 'blend',
                'timestamp': timestamp_to_iso(current_timestamp),
                'x': float(tokens[1]),          # X coordinate
                'y': float(tokens[2]),          # Y coordinate
                'z': float(tokens[3]),          # Z coordinate
                'vx': float(tokens[4]),         # X velocity
                'vy': float(tokens[5]),         # Y velocity
                'vz': float(tokens[6])          # Z velocity
            }

            # Process higher order data (if any)
            # Check subsequent lines for higher order data
            j = 1
            while i + j < len(lines):
                next_line = lines[i + j].strip()
                next_tokens = next_line.split()
                # Stop processing higher order data if new data row is encountered
                if len(next_tokens) >= 7 and next_tokens[0].replace('.', '', 1).isdigit():
                    break
                # Add higher order data to record
                for k, val in enumerate(next_tokens):
                    record[f'high_order_{j}_{k}'] = float(val)
                j += 1

            records.append(record)
            i += j
            data_index += 1
        else:
            i += 1

    # Convert parsed results to DataFrame and save as CSV
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Parsing complete: {len(df)} records saved to {output_path}")

def merge_ephemeris_csv(csv_dir: str):
    """
    Merge ephemeris CSV files for the same satellite, remove duplicates by timestamp,
    keep the latest file data.
    
    Args:
        csv_dir (str): Directory containing CSV files
    """
    # 1. Collect all csv files
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    # 2. Parse filenames and group by satellite name
    sat_groups = {}
    for fname in files:
        parts = fname.split('_')
        if len(parts) < 4:
            continue  # Skip non-standard files
        sat_name = parts[2]
        try:
            file_time = int(parts[3])
        except ValueError:
            continue  # Skip files with unparseable time
        key = sat_name
        sat_groups.setdefault(key, []).append((file_time, fname))

    # 3. Sort files in each group by time
    for sat, file_list in sat_groups.items():
        file_list.sort()  # Sort by file_time ascending

    # 4. Merge files in each group
    for sat, file_list in sat_groups.items():
        merged = None
        for _, fname in file_list:
            fpath = os.path.join(csv_dir, fname)
            df = pd.read_csv(fpath)
            if merged is None:
                merged = df
            else:
                # Remove duplicates by timestamp, keep latest
                merged = pd.concat([merged, df]).drop_duplicates('timestamp', keep='last')
        
        # 5. Output merged file, named as first four fields + _merged.csv
        output_dir = csv_dir.replace('csv', 'merged_csv')
        os.makedirs(output_dir, exist_ok=True)
        out_name = '_'.join(file_list[-1][1].split('_')[:4]) + '_merged.csv'
        out_path = os.path.join(output_dir, out_name)
        merged.to_csv(out_path, index=False)
        print(f"Merged file output: {out_path}")

def process_merged_files(input_dir="../data/ephemeris/merged_csv"):
    """
    Process merged CSV files:
    1. Rename columns to standard format
    2. Convert timestamp to ISO format
    3. Remove _merged suffix from filenames
    
    Args:
        input_dir (str): Directory containing merged CSV files
    """
    # Get all CSV files to process
    csv_files = glob.glob(os.path.join(input_dir, "*_merged.csv"))
    
    # Process each file
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Rename columns
            column_mapping = {
                'file_type': 'Coordinate',
                'satellite_catalog': 'Satellite_Number',
                'satellite_name': 'Satellite_Name',
                'ephemeris_source': 'Ephemeris_Source',
                'timestamp': 'Timestamp',
                'x': 'X',
                'y': 'Y',
                'z': 'Z',
                'vx': 'Vx',
                'vy': 'Vy',
                'vz': 'Vz'
            }
            
            # Rename basic columns
            df = df.rename(columns=column_mapping)
            
            # Rename high_order columns
            high_order_cols = [col for col in df.columns if col.startswith('high_order')]
            for col in high_order_cols:
                new_col = col.replace('high_order', 'High_Order')
                df = df.rename(columns={col: new_col})
            
            # Convert Timestamp column
            df['Timestamp'] = df['Timestamp'].apply(convert_jd_to_iso)
            
            # Generate new filename (remove _merged suffix)
            filename = os.path.basename(csv_file)
            new_filename = filename.replace('_merged.csv', '.csv')
            output_dir = "../data/ephemeris"
            new_filepath = os.path.join(output_dir, new_filename)
            
            # Save processed file
            df.to_csv(new_filepath, index=False)
            
            # Remove original file
            os.remove(csv_file)
            
            print(f"Processing complete: {csv_file} -> {new_filepath}")
            
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")

def process_all_ephemeris_files(input_dir='../data/ephemeris', output_dir='../data/ephemeris/csv'):
    """
    Process all ephemeris data files in the specified directory
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Get all txt files
    txt_files = glob.glob(os.path.join(input_dir, '**/*.txt'), recursive=True)
    
    if not txt_files:
        print(f"Warning: No txt files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} txt files to process")
    
    # Process each file
    for i, filepath in enumerate(txt_files, 1):
        print(f"\nProcessing file {i}/{len(txt_files)}: {filepath}")
        try:
            parse_meme_ephemeris_full(filepath, output_dir)
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")

    print("\nAll files processed!")

def run_pipeline():
    """
    Run the complete ephemeris data processing pipeline:
    1. Parse raw files to CSV
    2. Merge CSV files for same satellite
    3. Process and standardize merged files
    """
    # Step 1: Parse raw files
    process_all_ephemeris_files()
    
    # Step 2: Merge CSV files
    merge_ephemeris_csv('../data/ephemeris/csv')
    
    # Step 3: Process merged files
    process_merged_files()

if __name__ == '__main__':
    run_pipeline() 