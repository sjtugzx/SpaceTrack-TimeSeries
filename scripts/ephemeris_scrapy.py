import requests
import json
import os
import time

# Login and get session
login_url = "https://www.space-track.org/ajaxauth/login"
login_payload = {
    "identity": "your email",  # Replace with your login email
    "password": "your password"  # Replace with your password
}
session = requests.Session()
response = session.post(login_url, data=login_payload)

if response.status_code != 200:
    print("Login failed, status code:", response.status_code)
    exit()

print("Login successful, Cookie:", session.cookies.get_dict())

# Browse available public file directories
dirs_url = "https://www.space-track.org/publicfiles/query/class/dirs"
response = session.get(dirs_url)

if response.status_code != 200:
    print("Failed to get public file directories, status code:", response.status_code)
    exit()

# Print returned directory list
directories = response.json()
print("Available directory list:", directories)

# Get file details
details_url = "https://www.space-track.org/publicfiles/query/class/loadpublicdata"
response = session.get(details_url)

if response.status_code != 200:
    print("Failed to get file details:", response.status_code)
    exit()

file_details = response.json()
print("File details retrieved successfully")

# Set save path
save_path = "E:\\TLE_data\\Spacetrack_public_file\\Zip"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Download file logic
pause_duration = 120  # Pause duration after each batch (seconds)
max_retries = 3  # Maximum number of retries
retry_delay = 10  # Retry delay after download failure (seconds)

batch_size = 5  # Number of files to process in each batch

# Initialize counter
processed_count = 0

for file_info in file_details:
    if processed_count >= batch_size:
        print(f"Processed {batch_size} files, pausing for {pause_duration} seconds...")
        time.sleep(pause_duration)  # Pause for a while
        processed_count = 0  # Reset counter

    if file_info['type'] == "Ephemeris":  # Filter file type
        file_link = file_info['link']
        file_name = file_info['name'].replace(':', '_')  # Replace illegal characters
        file_url = f"https://www.space-track.org/publicfiles/query/class/download?name={file_link}"

        # Check if file already exists
        file_path = os.path.join(save_path, file_name)
        if os.path.exists(file_path):
            print(f"File already exists, skipping download: {file_name}")
        else:
            print(f"Downloading file: {file_name}")
            for attempt in range(max_retries):
                response = session.get(file_url, stream=True)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                    print(f"File saved: {file_name}")
                    break
                elif response.status_code == 429:
                    print(f"Download failed: {file_name}, status code: {response.status_code}. Retry attempt {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay * (attempt + 1))  # Dynamically increase wait time
                else:
                    print(f"Download failed: {file_name}, status code: {response.status_code}.")
                    break

        # Increment counter for each processed file (existing or newly downloaded)
        processed_count += 1

print("All files processed successfully.")