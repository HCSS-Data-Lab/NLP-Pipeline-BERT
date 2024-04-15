import os
import time

import config
from src.init_folders import InitFolders
from src.text_cleaning import TextCleaning
from src.analysis import Analysis
from src.plotting import Plotting
from src.evaluation import Evaluation
from src.RAG import RAG
from src.merge import Merge

from src.dtm import DynamicTopicModeling
from src.texts_pp import TextPreProcess
from src.embeddings_pp import EmbeddingsPreProcess
from src.red_embeddings_pp import RedEmbeddingsPreProcess

def get_year_str(keyword_theme, years):
    """
    Get year str used for tfidf input file selection for dynamic topic modeling, like 'security_vt0.8_2019_2020'
    """
    if config.dtm_parameters["tfidf_threshold_type"] == "value":
        threshold_str = "vt"
    elif config.dtm_parameters["tfidf_threshold_type"] == "document":
        threshold_str = "dt"
    else:
        threshold_str = ""
    return keyword_theme + f"_tfidf_{threshold_str}{config.dtm_parameters['tfidf_threshold']}_" + "_".join(years)


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
    - Make txt output file with topic output: topics, top n terms, topic size
    - Think about hierarchy for code, add UI formatting for different tasks
    - Corpus specific stop words
    """

    # project_root = os.environ.get(r'C:\Github\NLP-Pipeline-BERT', os.getcwd())
    project_root = os.getcwd()
    data = "ParlaMint"
    task = "dtm"  # dtm, tm
    # years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
    years = ["2019", "2020", "2021", "2022"]

    if config.clean_parameters["clean_text"]:  # In config set clean_text to False to turn it off
        for year in years:
            text_cleaning = TextCleaning(project_root, data, year)
            text_cleaning.read_clean_raw_texts()

    # Do preprocessing for dynamic topic modeling
    if task == "dtm":
        dtm = DynamicTopicModeling(project_root, data, years)
        keyword_theme = "defense"
        keywords = [
            "military",
            "defense",
            "armed forces",
            # "security",
            # "strategy",
            "tactics",
            "army",
            "navy",
            "air force",
            "marines",
            "weapons",
            "warfare",
            "combat",
            # "intelligence",
            "operations",
            # "training",
            # "logistics",
            # "surveillance",
            "cybersecurity",
            "deterrence"]
        # keywords = ["defense", "military"]

        # keyword_theme = "security"
        # keywords = ['safety', 'security', 'threat', 'protest', 'national security', 'demonstration', 'law enforcement',
        #             'risk', 'police', 'danger', 'hazard', 'national security', 'law enforcement']

        if config.dtm_parameters["keyword_find"] == "search":
            year_str = keyword_theme + "_".join(years)
            dtm.find_keyword_docs(year_str, keywords)

        elif config.dtm_parameters["keyword_find"] == "tfidf":
            year_str = get_year_str(keyword_theme, years)
            dtm.find_keyword_docs_tfidf(keywords, year_str)

        else:
            # Adds docs from the years together in a folder, or when keywords_find=None
            # year_str = keyword_theme + "_".join(years)
            year_str = get_year_str(keyword_theme, years)
    else:
        year_str = years[0]

    init_folders = InitFolders(project_root=project_root,
                               project=data,
                               year=year_str)

    # if config.translate_param["translate"]:
    #     from src.translate import Translate
    #     for year in years:
    #         translate_obj = Translate(project_root=project_root, project=data, year=year)
    #         translate_obj.translate_text(from_last_added=True)

    # Initializing text data
    split_texts_path = init_folders.get_split_texts_path()
    text_bodies_path = init_folders.get_text_bodies_path()
    texts_pp = TextPreProcess(text_bodies_path, split_texts_path)
    texts = texts_pp.get_texts()  # Text title and Chunk

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
    output_folder = os.path.join(project_root, "output", data, year_str)
    analysis = Analysis(out_path=output_folder)
    topic_model = analysis.initialize_topic_model(text_chunks)

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
                            save_html=False)
        plotting.plot()

    elif task == "dtm":
        timestamps_chunks = dtm.get_time_stamps(texts)  # Timestamps and chunks
        timestamps = [item["date"] for item in timestamps_chunks]  # Only Timestamps
        print(f"Timestamps: ", len(timestamps))
        print(f"Text chunks: ", len(text_chunks))

        start = time.time()
        topics_over_time = topic_model.topics_over_time(text_chunks, timestamps, datetime_format="%Y-%m-%d", nr_bins=len(years)*2)
        print(f"Time elapsed for topics over time: {time.time() - start:.4f} seconds")
        print("Visualizing topics over time...")
        fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
        fig.write_html(os.path.join(output_folder, "figures", f"topics_over_time_{year_str}.html"))
    else:
        raise ValueError(f"The task {task} is undefined. Options are tm (topic modeling) and dtm (dynamic topic modeling)")

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
