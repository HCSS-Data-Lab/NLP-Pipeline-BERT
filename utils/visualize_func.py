import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.preprocessing import normalize
import config

from umap import UMAP
from typing import List, Union

import config

def add_convex_hulls(fig, topic_per_doc, embeddings_2d, topic_range):
    for id in topic_range:
        if id != -1:
            inds = np.where(topic_per_doc == id)[0]
            hull_points = embeddings_2d[inds]
            hull = ConvexHull(hull_points)

            for simplex in hull.simplices:
                fig.add_trace(go.Scatter(x=hull_points[simplex, 0], y=hull_points[simplex, 1], mode='lines', line_color='darkblue'))

            hull_vertices = hull_points[hull.vertices]
            fig.add_trace(go.Scatter(x=hull_vertices[:, 0], y=hull_vertices[:, 1], fill='toself', fillcolor='darkblue',
                                    opacity=0.1, line=dict(color='darkblue')))
    return fig

def get_sample_indices(topic_model, sample=1.0):
    np.random.seed(0)
    num_topics_in_fig = config.plotting_parameters["n_total"]
    topic_per_doc = topic_model.topics_
    sample_indices = []
    for topic in list(set(topic_per_doc))[:num_topics_in_fig]:
        indices_for_topic = np.where(np.array(topic_per_doc) == topic)[0]  # Indices for topic of for-loop
        size = int(len(indices_for_topic) * sample)  # Size (number of) samples for this topic
        sample_indices.extend(np.random.choice(indices_for_topic, size=size, replace=False))
    indices = np.array(sample_indices)
    return indices

def visualize_documents_(topic_model,
                         docs: List[str],
                         indices: List[int] = None,  # Added parameter indices
                         topics: List[int] = None,
                         embeddings: np.ndarray = None,
                         reduced_embeddings: np.ndarray = None,
                         sample: float = None,
                         hide_annotations: bool = False,
                         hide_document_hover: bool = False,
                         custom_labels: Union[bool, str] = False,
                         title: str = "<b>Documents and Topics</b>",
                         width: int = 1200,
                         height: int = 750):
    """
    THIS IS THE MODIFIED FUNCTION from the BERTopic module in Python,
    such that the sampled indices are returned.

    0Visualize documents and their topics in 2D

    Arguments:
        topic_model: A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`.
        embeddings: The embeddings of all documents in `docs`.
        reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
        sample: The percentage of documents in each topic that you would like to keep.
                Value can be between 0 and 1. Setting this value to, for example,
                0.1 (10% of documents in each topic) makes it easier to visualize
                millions of documents as a subset is chosen.
        hide_annotations: Hide the names of the traces on top of each cluster.
        hide_document_hover: Hide the content of the documents when hovering over
                             specific points. Helps to speed up generation of visualization.
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:

    To visualize the topics simply run:

    ```python
    topic_model.visualize_documents(docs)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and prefered pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Prepare embeddings
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic
    topic_model = BERTopic().fit(docs, embeddings)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_documents(docs, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
    fig.write_html("path/to/file.html")
    ```

    <iframe src="../../getting_started/visualization/documents.html"
    style="width:1000px; height: 800px; border: 0px;""></iframe>
    """
    topic_per_doc = topic_model.topics_

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    # OLD CODE
    # indices = []
    # for topic in set(topic_per_doc):
    #     s = np.where(np.array(topic_per_doc) == topic)[0]
    #     size = len(s) if len(s) < 100 else int(len(s) * sample)
    #     indices.extend(np.random.choice(s, size=size, replace=False))
    # indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit(embeddings_to_reduce)
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    unique_topics = set(topic_per_doc)
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    else:
        names = [f"{topic}_" + "_".join([word for word, value in topic_model.get_topic(topic)][:3]) for topic in unique_topics]

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = set(unique_topics).difference(topics)
    if len(non_selected_topics) == 0:
        non_selected_topics = [-1]

    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode='markers+text',
            name="other",
            showlegend=False,
            marker=dict(color='#CFD8DC', size=5, opacity=0.5)
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5)
                )
            )

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # # Convex hulls
    # topic_per_doc_samples = [topic_per_doc[index] for index in indices]  # Topic id for each of the sampled documents
    # tpds_arr = np.array(topic_per_doc_samples)
    # fig = add_convex_hulls(fig, tpds_arr, embeddings_2d, topics)  # Add convex hulls to figure

    return fig

def visualize_topics_over_time_(topic_model,
                                topics_over_time: pd.DataFrame,
                                top_n_topics: int = None,
                                topics: List[int] = None,
                                normalize_frequency: bool = False,
                                custom_labels: Union[bool, str] = False,
                                title: str = "<b>Topics over Time</b>",
                                width: int = 1250,
                                height: int = 450,
                                topics_background: List[int] = [],
                                background_alpha: float = 1.0,
                                color_legend_opaque: bool = True) -> go.Figure:  # Added topics_background parameter
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if isinstance(custom_labels, str):
        topic_names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        topic_names = ["_".join([label[0] for label in labels[:4]]) for labels in topic_names]
        topic_names = [label if len(label) < 30 else label[:27] + "..." for label in topic_names]
        topic_names = {key: topic_names[index] for index, key in enumerate(topic_model.topic_labels_.keys())}
    elif topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
    else:
        topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                       for key, value in topic_model.topic_labels_.items()}
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        marker_color = colors[index % 7]
        alpha = background_alpha if topic in topics_background else 1  # Conditional opacity based on whether the topic is in topics_background

        if color_legend_opaque:
            fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                                     mode='lines',
                                     marker_color=marker_color,
                                     opacity=alpha,  # Set opacity here
                                     hoverinfo="text",
                                     name=topic_name,
                                     hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))
        else:
            # Actual plot line
            fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                                     mode='lines',
                                     marker_color=marker_color,
                                     opacity=alpha,
                                     hoverinfo="text",
                                     hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words],
                                     showlegend=False))  # Disable legend for actual lines

            # Invisible line for the legend (always full opacity)
            fig.add_trace(go.Scatter(x=[None], y=[None],  # No actual data
                                     mode='lines',
                                     marker_color=marker_color,
                                     name=topic_name,
                                     opacity=1))  # Full opacity

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Frequentie (norm.)" if normalize_frequency else "Frequency",
        xaxis_title="Jaren",
        title={
            'text': f"{title}<br><sub>Nederlandse parlementaire debatten</sub>",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title="<b>Onderwerpen",
        )
    )
    return fig