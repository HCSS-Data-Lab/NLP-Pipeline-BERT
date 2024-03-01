import os

from src.preprocess import PreProcess
from src.analysis import Analysis
from src.plotting import Plotting
from src.evaluation import Evaluation
from src.merge import Merge

if __name__ == '__main__':

    """
    TODO:
    - Add making automatic folders for project
    - Clean up plotting class
    
    
    """


    input_folder = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\input"
    output_folder = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\output"
    project = "ParlaMint"

    preprocess = PreProcess(in_folder=input_folder,
                            out_folder=output_folder,
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
    path = os.path.join(output_folder, project)

    analysis = Analysis(path, mod_from_file, mod_emb_from_file)
    topic_model = analysis.initialize_topic_model(texts)

    # Plotting
    model_name = analysis.get_model_file_name()
    num_texts = len(texts)
    folder = os.path.join(output_folder, "figures")
    plotting = Plotting(topic_model=topic_model,
                        reduced_embeddings=reduced_embeddings,
                        model_name=model_name,
                        docs=texts,
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











