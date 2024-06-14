import os
import pandas as pd

import config
from src.init_folders import InitFolders
from src.text_cleaning import TextCleaning
from src.tm import TopicModeling
from src.plotting import Plotting
from src.evaluation import Evaluation
from src.RAG import RAG
from src.merge import Merge

from src.doc_selection import DocSelection
from src.dtm import DynamicTopicModeling
from src.texts_pp import TextPreProcess
from src.embeddings_pp import EmbeddingsPreProcess
from utils.representative_docs_func import _get_representative_docs_

def get_year_str(task, years, keyword_theme=None):
    """
    Get year str used for tfidf input file selection for dynamic topic modeling, like 'security_vt0.8_2019_2020'

    Args:
        task (str): tm or dtm
        years (List[str]): list of years as str
        keyword_theme (str): keyword theme used for output folder name

    Return:
        str: year str
    """
    if task == "tm":
        # E.g. '2020'
        return years[0]

    elif task == "dtm":
        selection_method = config.doc_selection_parameters["doc_selection_method"]
        if selection_method == "search":
            # E.g. 'security_search_2019_2020'
            return keyword_theme + "_search_" + "_".join(years)

        elif selection_method == "tfidf":
            if config.doc_selection_parameters["tfidf_threshold_type"] == "value":
                threshold_str = "vt"
            elif config.doc_selection_parameters["tfidf_threshold_type"] == "document":
                threshold_str = "dt"

            # E.g. 'security_tfidf_vt0.8_2019_2020'
            return keyword_theme + f"_tfidf_{threshold_str}{config.doc_selection_parameters['tfidf_threshold']}_" + "_".join(
                years)

    else:
        print(f"Task {task} is undefined. ")

def get_representative_sents(topic_model, documents, num_docs, topics, texts_path, use_custom=True):
    """
    Find and return representative sentences for given topics

    Args:
        topic_model (BERTopic object): trained bertopic object
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

def assert_parameters(task, years):
    """
    Assert whether the task is tm or dtm and if task=tm, there is only a single year

    Args:
        task (str): tm or dtm
        years (List[str]): list of years as str
    """
    if task not in ["tm", "dtm"]:
        raise ValueError(
            f"The task: {task} is undefined. Options are tm (topic modeling) and dtm (dynamic topic modeling). ")

    if task == "tm":
        if len(years) > 1:
            raise ValueError(
                "For task: tm (topic modeling) years list must contain only a single value. "
            )

if __name__ == '__main__':

    project_root = os.getcwd()
    dataset_name = "ParlaMint"
    task = "dtm"  # dtm, tm

    years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]

    # Assert parameters
    assert_parameters(task, years)

    # Text cleaning
    if config.clean_parameters["clean_text"]:  # In config set clean_text to False to turn it off
        for year in years:
            text_cleaning = TextCleaning(project_root=project_root, dataset_name=dataset_name, year=year)
            text_cleaning.read_clean_raw_texts()

    # Translate texts
    if config.translate_parameters["translate"]:
        from src.translate import Translate
        for year in years:
            translate_obj = Translate(project_root=project_root, dataset_name=dataset_name, year=year)
            translate_obj.translate_text(from_last_added=True)

    # Document selection based on keywords
    if config.doc_selection_parameters["use_keyword_doc_selection"]:
        # Define keywords (used for search) and keyword theme (used in folder name)
        keyword_theme = 'security'
        keywords = ['safety', 'security', 'threat', 'protest', 'national security', 'demonstration', 'law enforcement',
                    'risk', 'police', 'danger', 'hazard', 'national security', 'law enforcement']

        year_str = get_year_str(task, years, keyword_theme)
        doc_selection = DocSelection(project_root, dataset_name, years)
        if config.doc_selection_parameters["select_documents"]:
            print("Selecting documents...")
            doc_selection.find_copy_docs(keywords=keywords, keyword_year_str=year_str)
        else:
            print("Using selected documents...")
            project_folder = doc_selection.get_project_folder()
            if not os.path.exists(os.path.join(project_folder, year_str)):
                raise ValueError("The folder for selected documents project_folder/year_str does not exist. Set 'select_documents' in doc_selection_parameters config to True. ")
    else:
        keyword_theme, keywords = None, None
        year_str = get_year_str(task, years, keyword_theme)

    init_folders = InitFolders(project_root=project_root,
                               dataset_name=dataset_name,
                               year_str=year_str)

    # Initializing text data
    split_texts_path = init_folders.get_split_texts_path()
    text_bodies_path = init_folders.get_text_bodies_path()
    texts_pp = TextPreProcess(text_bodies_path, split_texts_path)
    texts, text_chunks = texts_pp.get_texts()  # texts has text title and chunk; text_chunks is only chunks

    # Initialize embeddings and reduced embeddings
    emb_path = init_folders.get_emb_path()
    embeddings_pp = EmbeddingsPreProcess(emb_path)
    embeddings = embeddings_pp.get_embeddings(text_chunks)
    reduced_embeddings = embeddings_pp.get_red_embeddings(embeddings)

    # Initialize topic-model
    output_folder = os.path.join(project_root, "output", dataset_name, year_str)
    tm = TopicModeling(out_path=output_folder)
    topic_model = tm.initialize_topic_model(text_chunks)

    save_topic_words = False
    if save_topic_words:
        tm.save_topic_words(topic_model)

    get_repre_sents = False
    if get_repre_sents:
        text_splits_path = init_folders.get_split_texts_path()
        documents = pd.DataFrame({"Document": text_chunks, "Topic": topic_model.topics_})
        num_docs = 6
        topics = [1, 2, 3, 4, 5, 6]
        sents = get_representative_sents(topic_model, documents, num_docs=num_docs, topics=topics,
                                         texts_path=text_splits_path, use_custom=False)

    if task == "tm":
        # Plotting
        model_name = tm.get_model_file_name()
        num_texts = len(text_chunks)

        # Initiate RAG, enhance topic labels based on RAG and summarize docs
        RAG_from_file = False
        summarize_labels = False
        summarize_docs = False
        rag_path = init_folders.get_rag_path()
        rag = RAG(embeddings, text_chunks, RAG_from_file, path=rag_path)

        plotting = Plotting(topic_model=topic_model,
                            reduced_embeddings=reduced_embeddings,
                            model_name=model_name,
                            docs=text_chunks,
                            summarize_labels=summarize_labels,
                            summarize_docs=summarize_docs,
                            rag=rag,
                            folder=os.path.join(output_folder, "figures"),
                            year=year_str)
        plotting.plot()

    elif task == "dtm":
        # Initialize dtm object
        dtm = DynamicTopicModeling(project_root, dataset_name)

        # Find timestamps
        timestamps, _ = dtm.get_time_stamps(texts)  # Timestamps and chunks

        # Find timestamp bins
        timestamp_bins = dtm.get_timestamp_bins(timestamps=timestamps, frequency='QS')

        print(f'{"Number of timestamps:":<65}{len(timestamp_bins):>10}')
        print(f'{"Number of text chunks:":<65}{len(text_chunks):>10}')

        # Run topics over time
        if config.LOAD_TOPICS_OVER_TIME_FROM_FILE:
            print("Loading topics over time from file...")
            topics_over_time = pd.read_csv(os.path.join(output_folder, "models", "topics_over_time.csv"))
        else:
            topics_over_time = dtm.run_dtm(topic_model, text_chunks, timestamp_bins, output_folder)

        # Visualize topics over time
        dtm.visualize_topics(topic_model=topic_model,
                             topics_over_time=topics_over_time,
                             output_folder=output_folder,
                             year_str=year_str)

    #################################################################
    # Merging and Evaluation below (optional)
    #################################################################

    # Merging
    merge_topics = False
    if merge_topics:
        linkage_func = "ward"
        merge_obj = Merge(linkage_func="ward")

        hierar_topics = merge_obj.get_hierarchical_topics(topic_model, texts)
        topics_to_merge = merge_obj.get_topics_to_merge(hierar_topics)
        print(topics_to_merge)

        topic_model.merge_topics(texts, topics_to_merge)

    # Evaluation, calculate coherence
    evaluate_topics = False
    if evaluate_topics:
        evaluation = Evaluation()
        metrics = ["c_v", "c_npmi"]
        coherence_dict = evaluation.calculate_coherence(topic_model, texts, metrics)
        print(coherence_dict)

        # div = evaluation.calculate_diversity(topic_model)
