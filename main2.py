import pandas as pd
import os
from typing import List
import re
import json
import shutil
from nltk.tokenize import sent_tokenize

import config
from src.init_folders import InitFolders
from src.embeddings_pp import EmbeddingsPreProcess
from src.tm import TopicModeling
from src.dtm import DynamicTopicModeling
from src.RAG import RAG
from src.plotting import Plotting
from utils.representative_docs_func import _get_representative_docs_

def collect_debates(path):
    """
    Collect debates from folder path.

    Args:
        path (str or PathLike): folder path of debates

    Returns:
        debates (List[Dict[str, List[Dict[str, str]]]]): List of debates,
                where every debate is dict {"debate id": ..., "speeches": ...),
                and each speech is a dict {"speech id": ..., "speech text": ...)
    """
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
                    speeches.append({"speech id": id, "speech text": clean_speech})

        debate_name = os.path.splitext(debate_file_name)[0]
        debates.append({"debate id": debate_name, "speeches": speeches})
    return debates

def clean_speech_body(speech):
    """
    Clean text body by removing text between brackets

    Args:
        speech (str): raw text of speech

    Returns:
        str: cleaned speech, text within brackets removed
    """
    pattern = r'\[\[.*?\]\]'
    return re.sub(pattern, '', speech)

def find_chair_ids(folder, debate_id):
    """
    Find ids of chairpersons in debate given by debate_id

    Args:
        folder (str or PathLike): folder with debates including metadata files
        debate_id (str): debate id

    Returns:
        List[str]: list of speaker IDs of chairperson for given debate
    """
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

def find_clean_thema_speeches(country_folder, year, pattern):
    """
    Find and clean thematic speeches, speech is considered thematic when there is a match with the pattern

    Args:
        country_folder (str or PathLike): folder for specific country
        year (str): year as str
        pattern (str): regex pattern for keyword match

    Returns:
        cleaned_speeches (List[Dict[str, str]]): cleaned speeches
    """
    # Find all debates
    raw_texts_folder = os.path.join(country_folder, year, "raw_texts")
    debates = collect_debates(raw_texts_folder)
    print(f"Number of debates for {year}: {len(debates)}")

    # Extract speeches, remove chairperson speeches and keep only speeches with keyword match
    cleaned_speeches = []
    for debate in debates:
        debate_id = debate['debate id']
        chair_ids = find_chair_ids(raw_texts_folder, debate_id)

        # Cleaned speeches with at least one match on the keywords, add to cleaned debates list
        kw_speeches = [speech for speech in debate['speeches'] if
                       speech['speech id'] not in chair_ids and re.search(pattern, speech['speech text'])]
        if kw_speeches:
            cleaned_speeches.extend(kw_speeches)

    return cleaned_speeches

def copy_files(input_folder, output_folder):
    """
    Copy files from input folder to output folder

    Args:
        input_folder (str or PathLike): input folder
        output_folder (str or PathLike): output folder
    """
    print(f"Files in input folder: {len(os.listdir(input_folder))}")
    for filename in os.listdir(input_folder):
        full_file_name = os.path.join(input_folder, filename)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, output_folder)
    print(f"Total files in output folder: {len(os.listdir(output_folder))}")

def make_regex_pattern(fpbs, fpas, kw_pattern):
    """
    Make regular expression pattern used for thematic speech selection. The correct regex syntax is added
    for phrases before (?<!...) and after (?!...) the keyword.

    Args:
        fpbs (List[str]): list of false positives that come before the keyword (self-defense)
        fpas (List[str]): list of false positives that come after the keyword (defense of our values)
        kw_pattern (str): keyword embedded in a regex pattern, like \bdefense\b

    Returns:
        pattern (str): keyword pattern with false positives excluded
    """
    # False positive before's
    fpb_list = [f"(?<!{fpb})" for fpb in fpbs]  # Combine with the negative lookback assertion
    regex_fpb = "".join(fpb_list)

    # False positive after's
    fpa_list = [f"(?!{fpa})" for fpa in fpas]  # Combine with the negative lookahead assertion
    regex_fpa = "".join(fpa_list)

    # Combine in a regex pattern and return it
    pattern = regex_fpb + kw_pattern + regex_fpa
    return pattern

def get_common_sentences():
    """
    Get common sentences to be removed from all speeches

    Returns:
        common_sentences (List[str]): common sentences to be removed
    """
    common_sentences = []
    pre_phrases = ["Thank you, ", "Yes, ", "No, "]
    phrases = ["Mr. Chairman", "Mr. President", "Madam President"]
    post_phrases = [", thank you", ", thank you very much"]

    common_sentences.extend([f"{pre_p}{p}." for pre_p in pre_phrases for p in phrases])
    common_sentences.extend([f"{p}{post_p}." for post_p in post_phrases for p in phrases])
    common_sentences.extend([f"{p}." for p in phrases])
    return common_sentences

def get_representative_sents(topic_model, documents, num_docs, topics, texts_path, use_custom=False):
    """
    Find and return representative sentences for given topics

    Args:
        topic_model (BERTopic): trained bertopic object
        documents (pd.DataFrame): df with text chunks and topic ID
        num_docs (int): number of representative docs to find for each topic
        topics (List[int]): topic ids to find representative sentences for
        texts_path (str): path where repre sentences will be saved as .txt
        use_custom (bool): use custom representative docs function or not

    Returns:
        sents (List[dict[str, str]]): List of dictionaries with topic_id and sentence
    """
    # Get representative documents for all topics using custom _get_representative_docs_
    print("Reading representative documents...")
    if use_custom:
        repr_docs = _get_representative_docs_(topic_model, documents, num_docs)

    # Read, save, and return representative texts
    sents_lst = []
    with open(os.path.join(texts_path, "repre_sents.txt"), "w+") as file:
        for topic_id in topics:
            # Find representative sentences
            if use_custom:
                sentences = repr_docs[topic_id]
            else:
                if num_docs > 3:
                    raise ValueError(
                        "Number of sentences can be at most 3. The BERTopic attribute representative_docs_ returns at most 3 representative docs. "
                        "If num_docs must be larger than 3, set use_custom to True. "
                    )
                sentences = topic_model.representative_docs_[topic_id][:num_docs]

            sents_lst.append({topic_id: sentences})

            # Write to .txt
            text = "\n\n".join(sentences)
            file.write(f"Topic {topic_id}:\n{text}\n\n")
    return sents_lst

if __name__ == '__main__':

    # Parameters
    project_root = os.getcwd()
    dataset_name = "ParlaMint paper"
    country = "NL"
    task = "tm"
    split_size = config.text_splitting_parameters['split_size']

    country_folder = os.path.join(project_root, "input", dataset_name, country)
    years = ["2019", "2020", "2021", "2022"]
    year_str = "_".join(years)

    print("Country: ", country)
    print("Task: ", task)
    print("Year str: ", year_str)

    # Make regex pattern
    false_positives_before = ["our ", "her ", "his ", "my ", "self-", "line of ", "common "]
    new_false_positives_before = ["your ", "its ", "any ", "government ", "file a ", "lacking in the ", "rights of the ", "fire ", "ecological "]
    false_positives_before += new_false_positives_before
    false_positives_after = [" charges", " of regional languages", " of biodiversity", " amendment", " testify", " of our republican values"]
    kw_pattern = r"\b(defense|defence)\b"
    pattern = make_regex_pattern(false_positives_before, false_positives_after, kw_pattern)

    # ASSERT data folder exists and has files
    texts_folder = os.path.join(country_folder, year_str, "raw_texts")
    os.makedirs(texts_folder, exist_ok=True)
    copy_raw_texts = False
    if copy_raw_texts:
        print("Copying files...")
        for year in years:
            in_folder = os.path.join(country_folder, year, "raw_texts")
            copy_files(in_folder, texts_folder)

    # CLEANING, thematic speech selection, making sentences
    keyword_str = "defense_defence_iter3"
    clean_sentences_from_file = True
    if clean_sentences_from_file:
        print("Reading clean sentences from file...")
        with open(os.path.join(country_folder, year_str, f'cleaned_sentences_{country}_{keyword_str}'), 'r') as file:
            texts_dict = json.load(file)

    else:
        print("Finding clean sentences at runtime...")
        clean_speech_from_file = True
        clean_common_sentences = True
        if clean_speech_from_file:
            print("Reading clean speeches from file...")
            with open(os.path.join(country_folder, year_str, f'cleaned_speeches_{country}_{keyword_str}'), 'r') as file:
                cleaned_speeches = json.load(file)
        else:
            print("Cleaning speeches at runtime...")
            cleaned_speeches = find_clean_thema_speeches(country_folder, year_str, pattern)
            with open(os.path.join(country_folder, year_str, f'cleaned_speeches_{country}_{keyword_str}'), 'w') as file:
                json.dump(cleaned_speeches, file)

        # SENTENCES to remove
        if clean_common_sentences:
            common_sentences = get_common_sentences()
            print("Common sentences to be removed: ", common_sentences)

        # SPLIT speeches into sentences if so specified
        if split_size == "sentence":
            print("Splitting speeches into sentences and removing common sentences...")
            texts_dict = []   # List[Dict[Tuple[str, str], [str, str]]] - list of dicts with tuple: {speech id: ..., speech text: ...}
            for speech in cleaned_speeches:
                speech_id, speech_text = speech["speech id"], speech["speech text"]
                sentences = sent_tokenize(speech_text)  # Collect sentences
                if clean_common_sentences:
                    texts_dict.extend(
                        [{"speech id": speech_id, "speech text": s} for s in sentences if s not in common_sentences])  # Add each sentence as dict with speech id to cleaned_sentences
                else:
                    texts_dict.extend(
                        [{"speech id": speech_id, "speech text": s} for s in sentences])

            # Save texts dict with cleaned sentences
            with open(os.path.join(country_folder, year_str, f'cleaned_sentences_{country}_{keyword_str}'), 'w') as file:
                json.dump(texts_dict, file)
        else:
            texts_dict = cleaned_speeches  # If split_size != sentence, do not split speeches further and run speech-level analysis

    # Extract only texts from id, text dictionary
    limit = None  # Set this to an integer (say 10) to test the code for 10 speeches; None if all speeches
    texts = []  # List[str]: list with speech texts

    if limit:
        for speech in texts_dict[:limit]:
            texts.append(speech['speech text'])
        texts_dict = texts_dict[:limit]
    else:
        for speech in texts_dict:
            texts.append(speech['speech text'])

    print(f"Texts: {len(texts)}")
    print(f"Speech ID and speech text dict: {len(texts_dict)}")

    # Topic modeling analysis
    init_folders = InitFolders(project_root=project_root,
                               dataset_name=dataset_name,
                               year_str=year_str,
                               country=country)

    # Initialize embeddings and reduced embeddings
    emb_path = init_folders.get_emb_path()
    embeddings_pp = EmbeddingsPreProcess(emb_path)
    embeddings = embeddings_pp.get_embeddings(texts)
    reduced_embeddings = embeddings_pp.get_red_embeddings(embeddings)

    # Initialize topic model
    output_folder = init_folders.get_output_folder()
    tm = TopicModeling(out_path=output_folder)
    topic_model = tm.initialize_topic_model(texts)

    save_topic_words = False
    if save_topic_words:
        tm.save_topic_words(topic_model, top_n_topics=20)

    get_repre_sents = False
    if get_repre_sents:
        text_splits_path = init_folders.get_split_texts_path()
        documents = pd.DataFrame({"Document": texts, "Topic": topic_model.topics_})
        num_docs = 6
        topics = [1, 2, 3, 4, 5, 6]
        sents = get_representative_sents(topic_model, documents, num_docs=num_docs, topics=topics, texts_path=text_splits_path, use_custom=False)

    if task == "tm":
        # Plotting
        model_name = tm.get_model_str()
        num_texts = len(texts)

        # Initiate RAG, enhance topic labels based on RAG and summarize docs
        RAG_from_file = False
        summarize_labels = False
        summarize_docs = False
        rag_path = init_folders.get_rag_path()
        rag = RAG(embeddings, texts, RAG_from_file, path=rag_path)

        plotting = Plotting(topic_model=topic_model,
                            model_name=model_name,
                            docs=texts,
                            reduced_embeddings=reduced_embeddings,
                            summarize_labels=summarize_labels,
                            summarize_docs=summarize_docs,
                            rag=rag,
                            folder=os.path.join(output_folder, "figures"),
                            year=year_str)
        plotting.plot()
    else:
        """ Run dynamic topic modeling """
        # Initialize dtm object
        dtm = DynamicTopicModeling(project_root, dataset_name)

        # Find timestamps
        timestamps, _ = dtm.get_time_stamps_speeches(texts_dict)  # Timestamps and chunks

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
        model_str = tm.get_model_str()
        dtm.visualize_topics(topic_model=topic_model,
                             topics_over_time=topics_over_time,
                             output_folder=output_folder,
                             year_str=year_str,
                             model_str=model_str)
