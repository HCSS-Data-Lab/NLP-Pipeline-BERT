import os
from src.preprocess import PreProcess
from src.analysis import Analysis
from src.plotting import Plotting
from src.evaluation import Evaluation
from src.RAG import RAG
from src.merge import Merge



if __name__ == '__main__':
    """
    TODO:
    - Maybe use logger instead of print statements (as we can also log configurations) and bring logging to the main instead of in submodules 
    - Do not refer to local folders (find a way to define projectroot agnosticly that also works within devcontainer)
    - Add making automatic folders for project
    - Add option to not install sentence-transformers (and make use of cheap default embedding for development).
    - In general it might also be great to have a dev set that we can use to do test runs (especially because ). 
    - Fix and time initializing RAG (fix subsequent queries)   
    """
    project_root = os.environ.get(r'C:\Github\NLP-Pipeline-BERT', os.getcwd()) #Put root project here
    project = "Politie"
    preprocess = PreProcess(project_root=project_root,
                            project=project)

    # Initializing text data
    splits_from_file = True
    texts = preprocess.initialize_texts(splits_from_file=splits_from_file)

    # Initialize embeddings
    emb_from_file = True
    embeddings = preprocess.initialize_embeddings(emb_from_file=emb_from_file, data=texts)

    red_from_file = True
    reduced_embeddings = preprocess.initialize_red_embeddings(red_from_file=red_from_file, embeddings=embeddings)

    # Initialize topic-model
    mod_from_file = True
    mod_emb_from_file = True
    path = os.path.join(project_root, 'output', project)

    analysis = Analysis(path, mod_from_file, mod_emb_from_file)
    topic_model = analysis.initialize_topic_model(texts)

    # Plotting
    model_name = analysis.get_model_file_name()
    num_texts = len(texts)
    folder = os.path.join(project_root, 'output', "figures")
    
    # Initiate RAG, enhance topic labels based on RAG and summarize docs
    RAG_from_file=True
    summarize_labels=True 
    summarize_docs=True
    rag = RAG(embeddings, texts, RAG_from_file, path=os.path.join(path, 'RAG'))
    
    plotting = Plotting(topic_model=topic_model,
                        reduced_embeddings=reduced_embeddings,
                        model_name=model_name,
                        docs=texts,
                        summarize_docs=summarize_docs,
                        summarize_labels=summarize_labels,
                        folder=folder,
                        rag=rag)
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











