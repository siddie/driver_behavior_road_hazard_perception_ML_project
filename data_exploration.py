import h5py
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_preparation import Preparation


class Data_exploration:
    data_dir = Preparation.data_dir  # gets the directory of "data" folder
    head_facial_features_ID = Preparation.head_facial_features_ID  # Order is important
    cols_2_drop_hazardPressesQualtrics_file = Preparation.cols_2_drop_hazardPressesQualtrics_file
    head_face_features_by_order = Preparation.head_face_features_by_order
    convet_sec_2_milis = 1000
    features = ['Pitch', 'Yaw', 'Roll', 'Anger',
       'Sadness', 'Disgust', 'Joy', 'Surprise', 'Fear', 'Contempt',
       'Brow Furrow', 'Brow Raise', 'Lip Corner Depressor', 'InnerBrowRaise',
       'EyeClosure', 'NoseWrinkle', 'UpperLipRaise', 'LipSuck', 'LipPress',
       'MouthOpen', 'ChinRaise', 'Smirk', 'LipPucker', 'Cheek Raise',
       'Dimpler', 'Eye Widen', 'Lid Tighten', 'Lip Stretch', 'Jaw Drop'] # no section label MediaTime Participant

    def __init__(self, reaction_time=1.7, fps=30):
        self.reaction_time = reaction_time  # The facial expression we want to analyze
        self.hd_file_db= os.path.join(Data_exploration.data_dir,"reaction_time_{}_splitted_data_DB.hd5".format(reaction_time))
        if not os.path.isfile(self.hd_file_db):
            print("can't find hdf5 file for reaction time {}, please try again".format(reaction_time))
        self.demographicData = os.path.join(Data_exploration.data_dir, "demographicData.xlsx")
        self.hazardPressesQualtrics = os.path.join(Data_exploration.data_dir, "hazardPressesQualtrics.xlsx")
        self.fps = fps # to remove
        self.set_meta_data()

    def set_meta_data(self):
        hdf5_file = h5py.File(self.hd_file_db, "r")
        hdf5_db = hdf5_file["no_NaN_splitted_data"]
        self.features_names = hdf5_db.attrs["features_names"]
        self.section_length = hdf5_db.attrs["section_length"]
        #self.fps = hdf5_db.attrs["fps"]
        self.sections_counter = hdf5_db.attrs["sections_counter"]
        self.data = pd.DataFrame(data=hdf5_db["splitted_data_2_use"][()], columns=self.features_names)
        self.target = self.data.groupby('section')['label'].sum()
        self.target[self.target != 0] = "Hazard"
        self.target[self.target == 0] = "No hazard"
        hdf5_file.close()

    def plot_label_distribution(self):
        plot = sns.countplot(x=self.target, palette='rainbow')
        total = self.target.size
        for patch in plot.patches:
            percentage = '{:.1f}%'.format(100 * patch.get_height() / total)
            x = patch.get_x() + patch.get_width() / 2 - 0.05
            y = patch.get_y() + patch.get_height()
            plot.annotate(percentage, (x, y), size=12)
        title = "Target Distribution"
        plt.title(title)
        plt.savefig(title + ".png")

    def correlation_matrix(self):
        data_features_lable = self.data.drop(["section", "MediaTime", "Participant", "label"], axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        sns.heatmap(data_features_lable.corr(), ax=ax, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.05)
        title = "Correlation Matrix"
        plt.title(title)
        plt.savefig(title + "_all_features.png")

    def correlation_matrix_by_threshold(self, threshold=0.3):
        data_features_lable = self.data.drop(["section", "MediaTime", "Participant", "label"], axis=1)
        components = list()
        visited = set()
        corr_matrix = data_features_lable.corr()
        for col in data_features_lable.columns:
            if col in visited:
                continue
            component = set([col, ])
            just_visited = [col, ]
            visited.add(col)
            while just_visited:
                visited_col = just_visited.pop(0)
                for idx, val in corr_matrix[visited_col].items():
                    if abs(val) > threshold and idx not in visited:
                        just_visited.append(idx)
                        visited.add(idx)
                        component.add(idx)
            components.append(component)
        counter = 0
        for component in components:
            if len(component) == 1:
                continue
            plt.figure(figsize=(12, 8))
            counter += 1
            sns.heatmap(corr_matrix.loc[component, component], cmap='coolwarm',annot=True,fmt='.2f',linewidths=0.05)
            title = "Sub Correlation Matrix"
            plt.title(title)
            plt.savefig(title + "_" + str(counter) + ".png")

    def boxplot_target_other_feature(self, feature_name):
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='label', y=feature_name, data=self.data, palette='rainbow')
        title = "{} by target".format(feature_name)
        plt.title(title)
        plt.savefig(title + ".png")

    def plots_data_exploration(self, threshold=0.3):
        #d = Data_exploration()
        #d.plots_data_exploration(threshold=0.3)
        self.plot_label_distribution()
        self.correlation_matrix()
        self.correlation_matrix_by_threshold(threshold=threshold)
        for feature in Data_exploration.features:
            self.boxplot_target_other_feature(feature)

    def get_hazard_descriptive_statistics(self):
        hazard_descriptive_statistics = pd.DataFrame([], columns=["Movie", "Total differents hazards", "Total presses",
                                                                  "Total participants have been this road",
                                                                  "mean presses per participant in movie"])
        for m in Preparation(reaction_time=self.reaction_time).mIDs:
             #m=Preparation(reaction_time=d.reaction_time).mIDs[0]
            hazard_m_data = pd.read_excel(self.hazardPressesQualtrics, sheet_name=m)
            hazard_m_data = hazard_m_data.iloc[:-3, :]
            hazard_m_data = hazard_m_data.drop(["Movie", "Participant"], axis=1)
            total_differents_hazards = hazard_m_data.shape[1] - 2
            total_presses = np.sum(hazard_m_data["Total number of presses"])
            total_participants_have_been_road = np.sum(
                hazard_m_data["The number of times they have been this way"].notnull())
            mean_presses = np.mean(hazard_m_data["Total number of presses"])
            new_row = pd.DataFrame(
                [(m, total_differents_hazards, total_presses, total_participants_have_been_road, mean_presses)],
                columns=["Movie", "Total differents hazards", "Total presses", "Total participants have been this road",
                         "mean presses per participant in movie"])
            hazard_descriptive_statistics = hazard_descriptive_statistics.append(new_row)
        hazard_descriptive_statistics_per_col = {}
        hazard_descriptive_statistics_no_movie_col = hazard_descriptive_statistics.iloc[:, 1:].apply(pd.to_numeric)
        for col in hazard_descriptive_statistics_no_movie_col.columns:
            hazard_descriptive_statistics_per_col[col] = pd.DataFrame(
                hazard_descriptive_statistics_no_movie_col[col].describe())
        return hazard_descriptive_statistics, hazard_descriptive_statistics_per_col
