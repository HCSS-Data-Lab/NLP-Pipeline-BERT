import pandas as pd
import os
from typing import List, Dict
import re
import numpy as np
import json

import config
from src.init_folders import InitFolders
from src.embeddings_pp import EmbeddingsPreProcess
from src.analysis import Analysis
from src.dtm import DynamicTopicModeling

def collect_debates(path) -> List[Dict[str, List[Dict[str, str]]]]:
    """ Collect debates from folder path.
    Each debate is dict (debate id, speeches), each speech is a dict (speech id, speech text) """
    file_names = sorted([name for name in os.listdir(path) if name.endswith(".txt")])
    debates = []
    for debate_file_name in file_names:
        speeches = []
        with open(os.path.join(path, debate_file_name), encoding='utf-8') as f:
            body = f.read()
            speeches_raw = body.split('\n')  # speech id and text
            for s in speeches_raw:
                if len(s) > 0:
                    id = s.split('\t')[0]
                    speech = s.split('\t')[1]
                    clean_speech = clean_speech_body(speech)
                    speeches.append({"speech id": id, "speech text": speech})

        debate_name = os.path.splitext(debate_file_name)[0]
        debates.append({"debate id": debate_name, "speeches": speeches})
    return debates

def clean_speech_body(speech):
    """ Clean text body by removing text between brackets """
    pattern = r'\[\[.*?\]\]'
    return re.sub(pattern, '', speech)

def find_chair_ids(folder, debate_id):
    """ Find ids of chairpersons in debate given by debate_id """
    # Find correspoding meta file
    meta_files = [f for f in os.listdir(folder) if f.endswith('.tsv')]
    meta_file = ''
    for meta in meta_files:
        if debate_id in meta:
            meta_file = meta

    if meta_file:  # If meta file is found
        df = pd.read_csv(os.path.join(folder, meta_file), delimiter='\t')
        df = df[['ID', 'Speaker_role']]
        df_chairs = df[df['Speaker_role'] == 'Chairperson']
        chair_ids = df_chairs['ID'].values.tolist()
        return chair_ids
    else:
        raise ValueError(f"Metadata file for debate {debate_id} not found")

def find_clean_thema_speeches(dataset_folder, year, keyword):
    # Find all debates
    raw_texts_folder = os.path.join(dataset_folder, year, "raw_texts")
    debates = collect_debates(raw_texts_folder)
    print(f"Number of debates for {year}: {len(debates)}")

    # Extract speeches, remove chairperson speeches and keep only speeches with keyword match
    cleaned_speeches = []
    for debate in debates:
        debate_id = debate['debate id']
        chair_ids = find_chair_ids(raw_texts_folder, debate_id)

        # Cleaned speeches with a match on the keyword, add to cleaned debates list
        kw_speeches = [speech for speech in debate['speeches'] if
                       speech['speech id'] not in chair_ids and keyword in speech['speech text']]
        if kw_speeches:
            cleaned_speeches.extend(kw_speeches)

    return cleaned_speeches


if __name__ == '__main__':

    """ Parameters """
    project_root = os.getcwd()
    dataset_name = "ParlaMint paper"

    dataset_folder = os.path.join(project_root, "input", dataset_name)
    year = "2019_2020_2021_2022"
    keyword = "defense"

    """ Cleaning and thematic selection """
    # You can use the leaned speeches I sent and leave this as True
    # The file should be in a folder input/ParlaMint paper/2019_2020_2021_2022/
    # The folder ParlaMint paper I made myself, as well as the folder 2019_2020_2021_2022
    clean_speech_from_file = True

    if clean_speech_from_file:
        print("Reading clean speeches from file...")
        # Read from file
        with open(os.path.join(dataset_folder, year, 'cleaned_speeches'), 'r') as file:
            cleaned_speeches = json.load(file)
    else:
        print("Cleaning speeches...")
        cleaned_speeches = find_clean_thema_speeches(dataset_folder, year, keyword)  # List[Dict[str, str]] - list of dicts with speech id: speech text
        save_cleaned_speeches = True
        if save_cleaned_speeches:
            with open(os.path.join(dataset_folder, year, 'cleaned_speeches'), 'w') as file:
                json.dump(cleaned_speeches, file)

    """ Topic modeling analysis """
    init_folders = InitFolders(project_root=project_root,
                               dataset_name=dataset_name,
                               year_str=year)

    limit = None  # Set this to an integer (say 10) to test the code for 10 speeches; None if all speeches
    texts = []  # List[str]: list with speech texts

    if limit:
        for speech in cleaned_speeches[:limit]:
            texts.append(speech['speech text'])
        cleaned_speeches = cleaned_speeches[:limit]
    else:
        for speech in cleaned_speeches:
            texts.append(speech['speech text'])

    print(f"Texts list: {len(texts)}")
    print(f"Cleaned speeches dict: {len(cleaned_speeches)}")

    # Initialize embeddings and reduced embeddings
    emb_path = init_folders.get_emb_path()
    embeddings_pp = EmbeddingsPreProcess(emb_path)
    embeddings = embeddings_pp.get_embeddings(texts)
    reduced_embeddings = embeddings_pp.get_red_embeddings(embeddings)

    # Initialize topic model
    output_folder = os.path.join(project_root, "output", dataset_name, year)
    analysis = Analysis(out_path=output_folder)
    topic_model = analysis.initialize_topic_model(texts)

    """ Run dynamic topic modeling """
    # Initialize dtm object
    dtm = DynamicTopicModeling(project_root, dataset_name)

    # Find timestamps
    timestamps, _ = dtm.get_time_stamps_speeches(cleaned_speeches)  # Timestamps and chunks

    # Find timestamp bins
    timestamp_bins = dtm.get_timestamp_bins(timestamps=timestamps, frequency='QS')

    print(f'{"Number of timestamps:":<65}{len(timestamp_bins):>10}')
    print(f'{"Number of speeches:":<65}{len(texts):>10}')

    # Run topics over time
    if config.LOAD_TOPICS_OVER_TIME_FROM_FILE:
        print("Loading topics over time from file...")
        topics_over_time = pd.read_csv(os.path.join(output_folder, "models", "topics_over_time.csv"))
    else:
        topics_over_time = dtm.run_dtm(topic_model, texts, timestamp_bins, output_folder)

    # Visualize topics over time
    dtm.visualize_topics(topic_model=topic_model,
                         topics_over_time=topics_over_time,
                         output_folder=output_folder,
                         year_str=year)
