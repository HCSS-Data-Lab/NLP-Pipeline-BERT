import re

import numpy as np
import math
import os
import pandas as pd

# from src.visualize_documents_func import visualize_documents_

from scipy.spatial import ConvexHull
import plotly.graph_objects as go

import matplotlib.pyplot as plt



if __name__ == "__main__":


    pattern = "\[\[|\]\]|ParlaMint.+?\s"
    test = ["ParlaMint-NL_2014-04-16-tweedekamer-2.u1	There are now enough"]

    for t in test:
        new = re.sub(pattern, "", t)
        print(new)






























































