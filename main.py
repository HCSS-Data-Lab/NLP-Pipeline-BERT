import os

from src.preprocess import PreProcess
from src.analysis import Analysis
from src.plotting import Plotting
from src.evaluation import Evaluation

if __name__ == '__main__':

    """
    TODO:
    - Inheritance of classes for pre-processing, texts_pp and embeddings_pp; maybe a neat OO solution is possible here
    - Add default pp param from config to function definitions.
    - Add/improve documentation
    """

    input_folder = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\input"
    output_folder = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\output"
    project = "Politie"

    preprocess = PreProcess(in_folder=input_folder,
                            out_folder=output_folder,
                            project=project)

    # Initializing text data
    splits_from_file = False
    texts = preprocess.initialize_texts(splits_from_file=splits_from_file)

    # Initialize embeddings
    emb_from_file = False
    embeddings = preprocess.initialize_embeddings(emb_from_file=emb_from_file, data=texts)

    red_from_file = False
    reduced_embeddings = preprocess.initialize_red_embeddings(red_from_file=red_from_file, embeddings=embeddings)

    # Initialize topic-model
    mod_emb_from_file = True
    mod_from_file = False
    path = os.path.join(output_folder, project)

    analysis = Analysis(path, mod_from_file, mod_emb_from_file)
    topic_model = analysis.initialize_topic_model(texts)

    # Plotting
    model_name = analysis.get_model_file_name()
    num_texts = len(texts)
    plotting = Plotting(topic_model, reduced_embeddings, model_name, num_texts)
    plotting.plot()

    # Evaluation
    # evaluation = Evaluation()
    # coherence_dict = evaluation.calculate_coherence(topic_model, texts)
    # print(coherence_dict)

    # div = evaluation.calculate_diversity(topic_model)











