
from src.preprocess import PreProcess
from src.analysis import Analysis
import config


if __name__ == '__main__':

    """
    TODO:
    - inheritance of classes for pre-processing, texts_pp and embeddings_pp; maybe a neat OO solution is possible here
    - fix passing of split_size and clean_method for texts_pp. Default only in one function? Ask chatgpt
    - data is given as param to get_embeddings(), and also returned. This can be made neater
    - split embeddings and texts from file 
    
    """

    input_folder = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\input"
    output_folder = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\output"
    project = "Politie"

    preprocess = PreProcess(in_folder=input_folder, out_folder=output_folder, project=project)

    # something for emb_from_file, because if True, it is not necessary to read input texts

    # Initializing text data
    text_clean = "ft"  # ft (filter-texts), vect (vectorization) or def (default)
    split_size = "chunk"
    data = preprocess.initialize_texts(text_clean_method=text_clean, text_split_size=split_size)

    # Initialize embeddings
    emb_from_file = False
    embeddings = preprocess.initialize_embeddings(data=data, emb_from_file=emb_from_file)


    # # Initializing topic-model
    # model_from_file = False
    # model_name = f"bertopic_model_{text_clean}"
    # analysis = Analysis(model_from_file=model_from_file, text_clean=text_clean)
    #
    # analysis.initialize_topic_model()























