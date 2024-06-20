import os

from utils.visualize_func import visualize_topics_over_time_
import config

class DtmPlotting:

    def __init__(self):
        self.use_custom_labels = config.dtm_plotting_parameters["custom_labels"]
        print("dtm plotting object created. ")

    def visualize_topics(self, topic_model, topics_over_time, output_folder, year_str, model_str, custom_vis_func=False):
        print("Visualizing topics over time...")

        if self.use_custom_labels:
            # Make custom labels, ie topic id and first word
            info_df = topic_model.get_topic_info()
            names = info_df['Name'].tolist()
            custom_labels = [f"{i}: {name.split('_')[1]}" for i, name in enumerate(names)]
            topic_model.set_topic_labels(custom_labels)

        if custom_vis_func:
            fig, norm_freqs = visualize_topics_over_time_(topic_model,
                                                          topics_over_time,
                                                          **config.dtm_plotting_parameters)
            norm_freqs_out_path = os.path.join(output_folder, "models", f"norm_freq.csv")
            norm_freqs.to_csv(norm_freqs_out_path, index=False)
        else:
            fig = topic_model.visualize_topics_over_time(topics_over_time,
                                                         **config.dtm_plotting_parameters)

        out_path = os.path.join(output_folder, "figures", f"topics_over_time_{model_str}_{year_str}.html")
        if os.path.exists(out_path):
            fig.write_html(os.path.join(output_folder, "figures", f"test_topics_over_time_{model_str}_{year_str}.html"))
        else:
            fig.write_html(out_path)


