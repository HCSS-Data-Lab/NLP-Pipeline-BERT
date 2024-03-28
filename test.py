import re
import os
import glob


"""
    TODO:
    - Add create RAG folder to init_folders
    - Move TextCleaning forward in data pipeline, so that it comes before InitFolder, in main under swim lanes.
    - Make txt output file with topic output: topics, top n terms, topic size

    TODO long term:
    - Simplify regex in config for different projects/data, name by alias
"""

if __name__ == "__main__":

    def newest_file(folder):
        """
        Finds latest added file in folder and returns its name
        """
        files = glob.glob(os.path.join(folder, "*"))
        latest_file_path = max(files, key=os.path.getctime)
        return os.path.basename(latest_file_path)


    # path = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\input\NOS\text_bodies\2022_en"
    # text_names = os.listdir(path)
    # print(os.listdir(path))
    #
    # print(newest_file(path))

    test = "en_1 op 3 verpleegkundigen had vorig jaar last van seksueel ongewenst gedrag.txt"
    print(test.split("_")[1:])













































































