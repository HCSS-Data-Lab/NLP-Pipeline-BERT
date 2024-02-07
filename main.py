from src.preprocess import PreProcess

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

    splits_from_file = False
    emb_from_file = False

    preprocess = PreProcess(in_folder=input_folder,
                            out_folder=output_folder,
                            project=project)

    # Parameters
    text_clean = "def"  # ft (filter-texts), vect (vectorization) or def (default)
    split_size = "chunk"

    # Initializing text data
    texts = preprocess.initialize_texts(splits_from_file=splits_from_file, text_clean_method=text_clean, text_split_size=split_size)

    # Initialize embeddings
    embeddings = preprocess.initialize_embeddings(emb_from_file=emb_from_file, data=texts, text_clean_method=text_clean, text_split_size=split_size)






