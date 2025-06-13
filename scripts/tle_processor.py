"""
TLE Data Processing Pipeline
This script provides a complete pipeline for processing TLE data:
1. Parse raw TLE format files
2. Process and standardize the data
3. Merge files for the same satellite
4. Convert time formats
5. Extract specific satellite data (e.g., Starlink)
"""

import os
import pandas as pd
import glob
import datetime
from multiprocessing import Pool, cpu_count
from astropy.time import Time

# Constants for TLE calculations
DE2RA = 0.0174532925199433  # Degrees to radians conversion
TWOPI = 6.283185307179586   # 2 * PI
XMNPDA = 1440.0             # Minutes per day
TEMP = TWOPI / (XMNPDA * XMNPDA)
AE = 1.0
J2000 = 2451544.5
J1900 = J2000 - 36525.

# Directory paths
DATA_DIR = "../data/TLE"
ORIGINAL_DATA_DIR = os.path.join(DATA_DIR, "original_data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
OVERLAPED_DATA_DIR = os.path.join(DATA_DIR, "overlaped_data")
MERGED_DATA_DIR = os.path.join(DATA_DIR, "merged_data")

def cksum(line: str) -> int:
    """
    Calculate checksum for TLE line
    
    Args:
        line (str): TLE line to check
    
    Returns:
        int: Calculated checksum
    """
    tot = 0
    for i in range(68):
        c = line[i]
        if c.isdigit():
            tot += int(c)
        elif c == '-':
            tot += 1
    return tot % 10

def sci(string: str) -> float:
    """
    Convert scientific notation string to float
    
    Args:
        string (str): Scientific notation string
    
    Returns:
        float: Converted number
    """
    if string[1] == ' ':
        return 0.0
    sign = '-' if string[0] == '-' else ''
    mantissa = '.' + string[1:6]
    exponent = string[6:]
    return float(f"{sign}{mantissa}e{exponent}")

def datetime_to_julian(year, day_of_year):
    """
    Convert year and day of year to Julian date
    
    Args:
        year (int): Year
        day_of_year (float): Day of year
    
    Returns:
        float: Julian date
    """
    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + ((153 * m + 2) // 5) + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + (dt.hour - 12) / 24 + dt.minute / 1440 + dt.second / 86400 + dt.microsecond / 86400000000
    return jd

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

def parse_tle(line1, line2, sat_info):
    """
    Parse TLE data from two lines
    
    Args:
        line1 (str): First line of TLE data
        line2 (str): Second line of TLE data
        sat_info (DataFrame): Satellite information
    
    Returns:
        dict: Parsed TLE data or None if invalid
    """
    if line1[0] != '1' or line2[0] != '2':
        return None
    if cksum(line1) != int(line1[68]) or cksum(line2) != int(line2[68]):
        return None

    year = int(line1[18:20])
    full_year = 2000 + year if year < 57 else 1900 + year
    epoch_day = float(line1[20:32])
    epoch = datetime_to_julian(full_year, epoch_day)

    raw_sat_num = line1[2:7].strip()
    if not raw_sat_num.isdigit():
        return None

    sat_num = raw_sat_num
    if sat_num in sat_info['ID'].values:
        tle = {
            'Satellite_name': sat_info[sat_info["ID"] == sat_num]['Sat_name'].values[0],
            'Satellite_Number': sat_num,
            'International_Designator': line1[9:17].strip(),
            'Class': line1[7],
            'Epoch': epoch,
            'Mean_Anomaly': float(line2[43:52]) * DE2RA,
            'Right_Ascension_of_Node': float(line2[17:25]) * DE2RA,
            'Argument_of_Perigee': float(line2[34:43]) * DE2RA,
            'Eccentricity': float('0.' + line2[26:33].strip()),
            'Inclination': float(line2[8:16]) * DE2RA,
            'Mean_Motion': float(line2[52:63].strip()) * TEMP * XMNPDA,
            'First_Derivative_of_Mean_Motion': float(line1[33:43]) * TEMP,
            'Second_Derivative_of_Mean_Motion': sci(line1[44:52]) * TEMP / XMNPDA,
            'BSTAR_Drag_Term': sci(line1[53:61]) * AE,
            'Revolution_Number_at_Epoch': int(line2[63:69]),
            'Element_Number': int(line1[64:69]),
            'Original_Line1': line1,
            'Original_Line2': line2
        }
        return tle
    else:
        return None

def process_one_file(filename, input_directory, output_directory, sat_info):
    """
    Process a single TLE file
    
    Args:
        filename (str): Name of the file to process
        input_directory (str): Input directory path
        output_directory (str): Output directory path
        sat_info (DataFrame): Satellite information
    """
    if not filename.lower().endswith('.txt'):
        return
    
    input_path = os.path.join(input_directory, filename)
    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f]
    lines = [line for line in lines if line]

    if len(lines) < 2:
        print(f"File has less than 2 lines, skipping: {filename}")
        return

    tle_list = []
    for i in range(0, len(lines) - 1, 2):
        line1 = lines[i]
        line2 = lines[i + 1]
        tle = parse_tle(line1, line2, sat_info)
        if tle:
            tle_list.append(tle)

    if tle_list:
        df = pd.DataFrame(tle_list)
        df = df.sort_values("Epoch").groupby("Satellite_Number").first().reset_index()
        csv_filename = filename.rsplit('.', 1)[0] + '.csv'
        output_path = os.path.join(output_directory, csv_filename)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Saved: {output_path}")
    else:
        print(f"No valid TLE records in {filename}, skipping.")

def process_overlaped_tle_files(input_dir=OVERLAPED_DATA_DIR, output_dir=OVERLAPED_DATA_DIR):
    """
    Process overlaped TLE data files:
    1. Add column names
    2. Convert Epoch from Julian Date to ISO format
    
    Args:
        input_dir (str): Directory containing overlaped TLE CSV files
        output_dir (str): Output directory for processed files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files to process
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} overlaped TLE files to process")
    
    # Define column names
    column_names = [
        'Satellite_Number',
        'Satellite_name',
        'International_Designator',
        'Class',
        'Epoch',
        'Mean_Anomaly',
        'Right_Ascension_of_Node',
        'Argument_of_Perigee',
        'Eccentricity',
        'Inclination',
        'Mean_Motion',
        'First_Derivative_of_Mean_Motion',
        'Second_Derivative_of_Mean_Motion',
        'BSTAR_Drag_Term',
        'Revolution_Number_at_Epoch',
        'Element_Number'
    ]
    
    # Process each file
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print(f"\nProcessing overlaped TLE file {i}/{len(csv_files)}: {csv_file}")
            
            # Read CSV file without header
            df = pd.read_csv(csv_file, header=None)
            
            # Check if the number of columns matches
            if len(df.columns) != len(column_names):
                print(f"Warning: File {csv_file} has {len(df.columns)} columns, expected {len(column_names)}")
                continue
            
            # Assign column names
            df.columns = column_names
            
            # Convert Epoch column from Julian Date to ISO format
            df['Epoch'] = df['Epoch'].apply(convert_jd_to_iso)
            
            # Generate output file path
            filename = os.path.basename(csv_file)
            output_path = os.path.join(output_dir, filename)
            
            # Save processed file
            df.to_csv(output_path, index=False)
            
            print(f"Processing complete: {csv_file} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
    
    print("\nAll overlaped TLE files processed!")

def remove_iso_time_column(input_dir=MERGED_DATA_DIR, output_dir=MERGED_DATA_DIR):
    """
    Remove the last column (iso_time) from merged TLE data files
    
    Args:
        input_dir (str): Directory containing merged TLE CSV files
        output_dir (str): Output directory for processed files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files to process
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} merged TLE files to process")
    
    # Process each file
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print(f"\nProcessing merged TLE file {i}/{len(csv_files)}: {csv_file}")
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Remove the last column (iso_time)
            df = df.iloc[:, :-1]
            
            # Generate output file path
            filename = os.path.basename(csv_file)
            output_path = os.path.join(output_dir, filename)
            
            # Save processed file
            df.to_csv(output_path, index=False)
            
            print(f"Processing complete: {csv_file} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
    
    print("\nAll merged TLE files processed!")

def run_pipeline():
    """
    Run the complete TLE data processing pipeline:
    1. Parse raw TLE files
    2. Process overlaped data
    3. Remove iso_time column from merged data
    """
    # Load satellite information
    sat_info = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, '30_space_track_data.csv'), dtype={'ID': str})
    
    # Process raw TLE files
    files = [f for f in os.listdir(ORIGINAL_DATA_DIR) if f.lower().endswith('.txt')]
    max_workers = min(8, cpu_count() or 4)
    
    # 创建进程池
    with Pool(processes=max_workers) as pool:
        # 使用 starmap 传递多个参数
        args = [(f, ORIGINAL_DATA_DIR, PROCESSED_DATA_DIR, sat_info) for f in files]
        try:
            pool.starmap(process_one_file, args)
        except Exception as e:
            print(f"Error in multiprocessing: {e}")
    
    # Process overlaped TLE files
    process_overlaped_tle_files()
    
    # Remove iso_time column from merged files
    remove_iso_time_column()

if __name__ == '__main__':
    run_pipeline()
