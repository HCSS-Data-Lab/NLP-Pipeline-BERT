import os

import config
from src.init_folders import InitFolders
from src.text_cleaning import TextCleaning
from src.analysis import Analysis
from src.plotting import Plotting
from src.evaluation import Evaluation
from src.RAG import RAG
from src.merge import Merge
from src.translate import Translate

from src.texts_pp import TextPreProcess
from src.embeddings_pp import EmbeddingsPreProcess
from src.red_embeddings_pp import RedEmbeddingsPreProcess

if __name__ == '__main__':
    """
    TODO:
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
    - Simplify regex in config for different projects/data, name by alias 
    
    """
    # project_root = os.environ.get(r'C:\Github\NLP-Pipeline-BERT', os.getcwd()) #Put root project here
    project_root = os.getcwd()
    project = "NOS"
    year = "2022"

    if config.clean_parameters["clean_text"]:  # In config set clean_text to False to turn it off
        text_cleaning = TextCleaning(project_root, project, year)
        text_cleaning.read_clean_raw_texts()

    init_folders = InitFolders(project_root=project_root,
                               project=project,
                               year=year)

    if config.translate_param["translate"]:
        translate_obj = Translate(project_root=project_root, project=project, year=year)
        translate_obj.translate_text(from_last_added=True)

    # Initializing text data
    text_bodies_path = init_folders.get_text_bodies_path()
    split_texts_path = init_folders.get_split_texts_path()
    texts_pp = TextPreProcess(text_bodies_path, split_texts_path)
    texts = texts_pp.get_texts()

    # Initialize embeddings and reduced embeddings
    emb_path = init_folders.get_emb_path()
    embeddings_pp = EmbeddingsPreProcess(emb_path)
    embeddings = embeddings_pp.get_embeddings(texts)

    # Initialize reduced embeddings
    red_emb_pp = RedEmbeddingsPreProcess(emb_path)
    reduced_embeddings = red_emb_pp.get_red_embeddings(embeddings)

    # Initialize topic-model
    output_folder = os.path.join(project_root, "output", project, year)
    analysis = Analysis(out_path=output_folder)
    topic_model = analysis.initialize_topic_model(texts)

    # Plotting
    model_name = analysis.get_model_file_name()
    num_texts = len(texts)

    # Initiate RAG, enhance topic labels based on RAG and summarize docs
    RAG_from_file = False
    summarize_labels = False
    summarize_docs = False
    rag_path = init_folders.get_rag_path()
    rag = RAG(embeddings, texts, RAG_from_file, path=rag_path)

    plotting = Plotting(topic_model=topic_model,
                        reduced_embeddings=reduced_embeddings,
                        model_name=model_name,
                        docs=texts,
                        summarize_docs=summarize_docs,
                        summarize_labels=summarize_labels,
                        rag=rag,
                        folder=os.path.join(output_folder, "figures"),
                        year=year,
                        save_html=True)
    plotting.plot()

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
