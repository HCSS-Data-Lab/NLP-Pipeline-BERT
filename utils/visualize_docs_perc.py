"""
Python script to handle the visualisation of how many docs are "securitized" as % of all documents

Because plotting is more convenient in google colab, the bin count dicts that are printed here
are the input for the bar graph.
"""

import re
import os
import pandas as pd
from datetime import datetime, date
import shutil

def get_timestamp_dates(path):
    # Read timestamps
    df = pd.read_csv(path)
    unique_timestamps = set(df["Timestamp"].to_list())

    # Convert strings to date objects and sort them
    sorted_bins = sorted(datetime.strptime(ts, '%Y-%m-%d').date() for ts in unique_timestamps)
    return sorted_bins

def get_bin_counts(folder, timestamps):
    """
    Get bin counts between timestamps for all the files in folder, return as dict
    """
    file_counts = {bin: 0 for bin in timestamps}
    files = os.listdir(folder)

    # Process each file
    for file in files:
        date_str = extract_date(file)
        parts = date_str.split("-")

        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        file_date = date(year, month, day)

        bin = find_bin(file_date, timestamps)
        file_counts[bin] += 1
    return file_counts

def extract_date(name: str):
    """
    Extract date using regex to match the date pattern
    """
    match = re.search(r'\d{4}-\d{2}-\d{2}', name)
    if match:
        return match.group()
    return None

def find_bin(file_date, bins):
    """
    Function to find appropriate bin
    """
    for i in range(len(bins) - 1):
        if bins[i] <= file_date < bins[i + 1]:
            return bins[i]
    return bins[-1]  # If the date is beyond the last bin, categorize in the last bin

def copy_files_years(years, root, out_folder):
    """
    Copy files from root/[year]/text_bodies for all years to out_folder_path
    """
    for year in years:
        folder = os.path.join(root, year, "text_bodies")

        files = os.listdir(folder)
        for file in files:
            if file.endswith('.txt'):
                source_file_path = os.path.join(folder, file)
                destination_file_path = os.path.join(out_folder, file)
                shutil.copy(source_file_path, destination_file_path)

if __name__ == "__main__":

    path = r"C:\Users\ArneEichholtz\Downloads\topics_over_time.csv"
    sorted_timestamps = get_timestamp_dates(path)

    text_bodies_path = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\input\ParlaMint\security_tfidf_vt0.8_2015_2016_2017_2018_2019_2020_2021_2022\text_bodies"
    text_bin_counts = get_bin_counts(text_bodies_path, sorted_timestamps)

    # Make output folder
    years = [str(y) for y in range(2015, 2023)]
    root = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\input\ParlaMint"
    out_folder_name = "_".join(years)
    out_folder_path = os.path.join(root, out_folder_name, "text_bodies")
    os.makedirs(out_folder_path, exist_ok=True)

    # Copy files
    copy_files = False  # Set to True to copy files to output folder
    if copy_files:
        copy_files_years(years, root, out_folder_path)

    all_text_bin_counts = get_bin_counts(out_folder_path, sorted_timestamps)

    print(text_bin_counts)
    print(all_text_bin_counts)

