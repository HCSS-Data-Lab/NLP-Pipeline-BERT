import os
import pandas as pd

import config
from src.init_folders import InitFolders
from src.text_cleaning import TextCleaning
from src.analysis import Analysis
from src.plotting import Plotting
from src.evaluation import Evaluation
from src.RAG import RAG
from src.merge import Merge

from src.doc_selection import DocSelection
from src.dtm import DynamicTopicModeling
from src.texts_pp import TextPreProcess
from src.embeddings_pp import EmbeddingsPreProcess
from src.red_embeddings_pp import RedEmbeddingsPreProcess


def get_year_str(task, years, keyword_theme=None):
    """
    Get year str used for tfidf input file selection for dynamic topic modeling, like 'security_vt0.8_2019_2020'
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
        raise ValueError(
            f"The task: {task} is undefined. Options are tm (topic modeling) and dtm (dynamic topic modeling). ")


if __name__ == '__main__':
    """
    TODO: (Maarten)
    - Maybe use logger instead of print statements (as we can also log configurations) and bring logging to the main instead of in submodules 
    - Do not refer to local folders (find a way to define projectroot agnosticly that also works within devcontainer)
    - Add making automatic folders for project
    - Add option to not install sentence-transformers (and make use of cheap default embedding for development).
    - In general it might also be great to have a dev set that we can use to do test runs (especially because ).  

    TODO:
    - Split up pre-processing part of old repository, make separate preproc repo. Option to add several preproc.
    options, for instance also Grobid
    - Contributor to BERTopic package for Convex Hulls.
    - Think about hierarchy for code, add UI formatting for different tasks; add assertions for parameters; add functionality to read texts with keywords without sampling texts
    - Corpus specific stop words
    - Add .py file with parts of code so subtasks are read; test year_str and other new stuff
    """

    # project_root = os.environ.get(r'C:\Github\NLP-Pipeline-BERT', os.getcwd())
    project_root = os.getcwd()
    dataset_name = "ParlaMint"
    task = "dtm"  # dtm, tm
    subtask = "text splitting"  # text splitting, run topic model, plot tm, run topics over time, plot dtm

    years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
    # years = ["2020", "2021", "2022"]

    # Doc selection parameters
    use_selected_docs = True
    select_documents = False

    selection_method = config.doc_selection_parameters["doc_selection_method"]
    if selection_method:  # If selection method is not None
        keyword_theme = 'security'
        keywords = ['safety', 'security', 'threat', 'protest', 'national security', 'demonstration', 'law enforcement',
                    'risk', 'police', 'danger', 'hazard', 'national security', 'law enforcement']

        # keyword_theme = "defense"
        # keywords = ["military", "defense", "armed forces", "tactics", "army", "navy", "air force", "marines", "weapons",
        #             "warfare", "combat", "operations", "cybersecurity", "deterrence"]
    else:
        keyword_theme, keywords = None, None

    # Get year str, used for folder names
    year_str = get_year_str(task, years, keyword_theme)
    print("Year str: ", year_str)

    # Text cleaning
    if config.clean_parameters["clean_text"]:  # In config set clean_text to False to turn it off
        for year in years:
            text_cleaning = TextCleaning(project_root, dataset_name, year)
            text_cleaning.read_clean_raw_texts()

    # Document selection
    if config.doc_selection_parameters["doc_selection_method"]:  # If doc selection method is not None
        if select_documents:
            print("Selecting documents...")
            doc_selection = DocSelection(project_root, dataset_name, years)
            doc_selection.find_copy_docs(keywords=keywords, keyword_year_str=year_str)
        elif use_selected_docs:
            # In this case, use the docs already selected
            print("Using selected documents...")

    init_folders = InitFolders(project_root=project_root,
                               project=dataset_name,
                               year=year_str)

    # Translate texts
    if config.translate_param["translate"]:
        from src.translate import Translate
        for year in years:
            translate_obj = Translate(project_root=project_root, project=dataset_name, year=year)
            translate_obj.translate_text(from_last_added=True)

    # Initializing text data
    split_texts_path = init_folders.get_split_texts_path()
    text_bodies_path = init_folders.get_text_bodies_path()
    texts_pp = TextPreProcess(text_bodies_path, split_texts_path)
    texts = texts_pp.get_texts()  # texts has text title and chunk

    # Collect text chunks
    text_chunks = [chunk for value in texts.values() for chunk in value]
    print(f'{"Text chunks:":<65}{len(text_chunks):>10}\n')

    # Initialize embeddings and reduced embeddings
    emb_path = init_folders.get_emb_path()
    embeddings_pp = EmbeddingsPreProcess(emb_path)
    embeddings = embeddings_pp.get_embeddings(text_chunks)

    # Initialize reduced embeddings
    red_emb_pp = RedEmbeddingsPreProcess(emb_path)
    reduced_embeddings = red_emb_pp.get_red_embeddings(embeddings)

    # Initialize topic-model
    output_folder = os.path.join(project_root, "output", dataset_name, year_str)
    analysis = Analysis(out_path=output_folder)
    topic_model = analysis.initialize_topic_model(text_chunks)

    save_topic_words = False
    if save_topic_words:
        analysis.save_topic_words(topic_model)

    if task == "tm":
        # Plotting
        model_name = analysis.get_model_file_name()
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
                            summarize_docs=summarize_docs,
                            summarize_labels=summarize_labels,
                            rag=rag,
                            folder=os.path.join(output_folder, "figures"),
                            year=year_str,
                            save_html=False)  # Add in config
        plotting.plot()

    elif task == "dtm":
        # Initialize dtm object
        dtm = DynamicTopicModeling(project_root, dataset_name)

        # Find timestamps
        timestamps_chunks = dtm.get_time_stamps(texts)  # Timestamps and chunks
        timestamps = [item["date"] for item in timestamps_chunks]  # Only Timestamps
        print(f'{"Number of timestamps:":<65}{len(timestamps):>10}')
        print(f'{"Number of text chunks:":<65}{len(text_chunks):>10}')

        # Run topics over time
        if config.LOAD_TOPICS_OVER_TIME_FROM_FILE:
            print("Loading topics over time from file...")
            topics_over_time = pd.read_csv(os.path.join(output_folder, "models", "topics_over_time.csv"))
        else:
            nr_of_bins = len(years) * 4
            topics_over_time = dtm.run_dtm(topic_model, text_chunks, timestamps, nr_of_bins, output_folder)

        # Visualize topics over time
        top_n_topics = 50
        topics_to_show = [1, 4, 5, 6, 7, 9, 11, 13, 14, 15]
        topics_background = [1, 4, 5, 9, 11, 13]
        background_alpha = 0.2
        color_legend_opaque = True
        dtm.visualize_topics(topic_model=topic_model,
                             topics_over_time=topics_over_time,
                             output_folder=output_folder,
                             year_str=year_str,
                             top_n=top_n_topics,
                             topics_to_show=topics_to_show,
                             topics_background=topics_background,
                             background_alpha=background_alpha,
                             legend_opaque=color_legend_opaque)

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
