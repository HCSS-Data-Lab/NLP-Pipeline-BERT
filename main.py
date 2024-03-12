import os

from src.preprocess import PreProcess
from src.analysis import Analysis
from src.plotting import Plotting
from src.evaluation import Evaluation
from src.merge import Merge

if __name__ == '__main__':

    """
    TODO:
    - Split up pre-processing part of old repository, make separate preproc repo. Option to add several preproc.
    options, for instance also Grobid
    - Contributor to BERTopic package for Convex Hulls.
    """

    # project_root = os.environ.get(r'C:\Github\NLP-Pipeline-BERT', os.getcwd())  # Put root project here
    # project_root = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT"

    project_root = os.getcwd()  # This should work when running it from the NLP-Pipeline-BERT folder
    project = "Politie"
    preprocess = PreProcess(project_root=project_root,
                            project=project)

    # Initializing text data
    texts = preprocess.initialize_texts()

    # Initialize embeddings and reduced embeddings
    embeddings = preprocess.initialize_embeddings(data=texts)
    reduced_embeddings = preprocess.initialize_red_embeddings(embeddings=embeddings)

    # Initialize analysis obj and topic-model
    output_folder = os.path.join(project_root, "output", project)
    analysis = Analysis(output_folder)
    topic_model = analysis.initialize_topic_model(texts)

    # Plotting
    model_name = analysis.get_model_file_name()
    num_texts = len(texts)
    folder = os.path.join(output_folder, "figures")

    summarize_labels = False
    summarize_docs = False

    plotting = Plotting(topic_model=topic_model,
                        reduced_embeddings=reduced_embeddings,
                        model_name=model_name,
                        docs=texts,
                        summarize_labels=summarize_labels,
                        summarize_docs=summarize_docs,
                        folder=folder)
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











