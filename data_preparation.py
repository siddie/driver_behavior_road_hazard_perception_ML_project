import h5py
import os
import numpy as np
import pandas as pd
from create_rawData import RawDataCreation


class Preparation:
    """
    TO DO
    After data pre-processing
    """
    data_dir = RawDataCreation.datadir  # gets the directory of "data" folder
    head_facial_features_ID = ["head", "facial"]  # Order is important
    attrs_4_validation_hd_file_head_face = {'m1': 40, 'm10': 40, 'm11': 39, 'm12': 41, 'm13': 42, 'm14': 40, 'm15': 36,
                                            'm16': 40, 'm17': 42, 'm18': 38, 'm19': 39, 'm3': 40, 'm4': 40, 'm5': 42,
                                            'm6': 41, 'm7': 39}
    cols_2_drop_hazardPressesQualtrics_file = ["Movie", "Participant", "The number of times they have been this way",
                                               "Total number of presses"]
    head_face_features_by_order = ['MediaTime', 'Pitch', 'Yaw', 'Roll', 'Anger', 'Sadness', 'Disgust', 'Joy',
                                   'Surprise', 'Fear', 'Contempt', 'Brow Furrow', 'Brow Raise', 'Lip Corner Depressor',
                                   'InnerBrowRaise', 'EyeClosure', 'NoseWrinkle', 'UpperLipRaise', 'LipSuck',
                                   'LipPress', 'MouthOpen', 'ChinRaise', 'Smirk', 'LipPucker', 'Cheek Raise', 'Dimpler',
                                   'Eye Widen', 'Lid Tighten', 'Lip Stretch', 'Jaw Drop']
    convert_sec_2_millis = 1000
    reaction_time_list = np.arange(0.6, 1.7, 0.1)

    def __init__(self, reaction_time=1):
        self.reaction_time = reaction_time  # The facial expression we want to analyze
        self.hd_file_head = os.path.join(Preparation.data_dir,
                                         "{}_DB.hd5".format(Preparation.head_facial_features_ID[0]))
        self.hd_file_face = os.path.join(Preparation.data_dir,
                                         "{}_DB.hd5".format(Preparation.head_facial_features_ID[1]))
        self.demographicData = os.path.join(Preparation.datdata_diradir, "demographicData.xlsx")
        self.set_meta_data()
        self.hazardPressesQualtrics = os.path.join(Preparation.data_dir, "hazardPressesQualtrics.xlsx")
        self.hd_file_head_face = os.path.join(Preparation.data_dir, "head_face_DB.hd5")
        self.section_length = int(self.reaction_time * self.fps)

    def set_subjects_ID(self, hdf5_face_DB, hdf5_head_DB):
        subjectList = []
        for m in self.mIDs:
            # m=p.mIDs[0]
            m_db_face = hdf5_face_DB[m]["processed_data_with_NaN"]
            m_db_head = hdf5_head_DB[m]["processed_data_with_NaN"]
            subjectList = subjectList + list(m_db_face.keys())
            subjectList = subjectList + list(m_db_head.keys())
        self.subjectIDs = list(set(subjectList))
        self.subjectIDs.sort(key=int)

    def set_general_attributes(self, hdf5_face_DB, hdf5_head_DB):
        movies_length = {}
        for m in self.mIDs:
            # m=p.mIDs[0]
            m_db_face = hdf5_face_DB[m]
            m_db_head = hdf5_head_DB[m]
            if m == self.mIDs[0]:
                v = m_db_head.attrs["features"].tolist()
                v2 = m_db_face.attrs["features"].tolist()
                v.remove("MediaTime")
                v2.remove("MediaTime")
                self.head_features = m_db_head.attrs["features"].tolist()
                self.face_features = m_db_face.attrs["features"].tolist()
                self.fps = m_db_head.attrs["fps"]
            movies_length[m] = m_db_head.attrs["length"]
        self.movies_length = movies_length

    def set_meta_data(self):
        hdf5_face_DB = h5py.File(self.hd_file_face, "r")
        hdf5_head_DB = h5py.File(self.hd_file_head, "r")
        self.mIDs = list(hdf5_head_DB.keys())
        self.set_subjects_ID(hdf5_face_DB, hdf5_head_DB)
        self.set_general_attributes(hdf5_face_DB, hdf5_head_DB)
        hdf5_face_DB.close()
        hdf5_head_DB.close()

    def check_subjects_not_in_both_DBs(self, subject, movie, m_db_head, m_db_face, subjects_not_in_both_DBs):
        is_subject_not_in_both_DBs = False
        if not subject in m_db_head:
            new_row = pd.DataFrame([(movie, subject, self.head_facial_features_ID[0])],
                                   columns=["Movie", "Participant", "DB name"])
            subjects_not_in_both_DBs = subjects_not_in_both_DBs.append(new_row)
            is_subject_not_in_both_DBs = True
        if not subject in m_db_face:
            new_row = pd.DataFrame([(movie, subject, self.head_facial_features_ID[1])],
                                   columns=["Movie", "Participant", "DB name"])
            subjects_not_in_both_DBs = subjects_not_in_both_DBs.append(new_row)
            is_subject_not_in_both_DBs = True
        return is_subject_not_in_both_DBs, subjects_not_in_both_DBs

    def reset_media_time_col(self, data):
        data_diff = data.diff()
        mean_data_diff = np.round(np.mean(data_diff[1:]))
        data_diff[0] = mean_data_diff
        newdata = data_diff.cumsum(axis=0)
        return newdata

    def combine_face_head_DB(self):
        hdf5_face_DB = h5py.File(self.hd_file_face, "a")
        hdf5_head_DB = h5py.File(self.hd_file_head, "a")
        hd5dir_face_head = os.path.join(Preparation.data_dir, "head_face_DB.hd5")
        hd = h5py.File(hd5dir_face_head, "a")
        subjects_not_in_both_DBs = pd.DataFrame([], columns=["Movie", "Participant", "DB name"])
        num_subject_per_movie = {}
        for m in self.mIDs:
            # m=p.mIDs[0]
            m_db_face = hdf5_face_DB[m]["processed_data_with_NaN"]
            m_db_head = hdf5_head_DB[m]["processed_data_with_NaN"]
            if m in hd:
                if "processed_data" in hd[m]:
                    del hd[m]["processed_data"]
                processed_data = hd[m].create_group("processed_data")
            else:
                processed_data = hd.create_group(m + "/processed_data")
            for s in self.subjectIDs:
                # s = p.subjectIDs[0]
                is_subject_not_in_both_DBs, subjects_not_in_both_DBs = self.check_subjects_not_in_both_DBs(subject=s,
                                                                                                           movie=m,
                                                                                                           m_db_head=m_db_head,
                                                                                                           m_db_face=m_db_face,
                                                                                                           subjects_not_in_both_DBs=subjects_not_in_both_DBs)
                if is_subject_not_in_both_DBs:
                    continue
                s_head_data = pd.DataFrame(m_db_head[s][()], columns=self.head_features)
                s_face_data = pd.DataFrame(m_db_face[s][()], columns=self.face_features)
                if s_head_data["MediaTime"].equals(s_face_data["MediaTime"]):
                    s_face_data = s_face_data.drop("MediaTime", axis=1)
                    s_combined_data = pd.concat([s_head_data, s_face_data], axis=1)
                    media_time_col = s_combined_data["MediaTime"]
                    new_media_time_col = self.reset_media_time_col(media_time_col.copy())
                    s_combined_data["MediaTime"] = new_media_time_col
                    processed_data.create_dataset(s, data=s_combined_data)
                else:
                    print("At movie {:s} and participant {:s} the face data and head data are not equals".format(m, s))
            num_subject_per_movie[m] = len(processed_data)
        for key, value in num_subject_per_movie.items():
            hd.attrs.create(key, value)
        self.subjects_not_in_both_DBs = subjects_not_in_both_DBs.drop_duplicates(['Movie', 'Participant'], keep='first')
        self.subjects_not_in_both_DBs = self.subjects_not_in_both_DBs.loc[:, ["Movie", "Participant"]]
        self.hd_file_head_face = os.path.join(hd5dir_face_head)
        hd.close()
        return subjects_not_in_both_DBs

    def get_hd_file_head_face(self):
        if not os.path.isfile(self.hd_file_head_face):
            _ = self.combine_face_head_DB()
            hdf5_head_face_DB = h5py.File(self.hd_file_head_face, "a")
        else:
            hdf5_head_face_DB = h5py.File(self.hd_file_head_face, "a")
            dict_hdf5_head_face_DB = dict(hdf5_head_face_DB.attrs)
            if dict_hdf5_head_face_DB != self.attrs_4_validation_hd_file_head_face:
                _ = self.combine_face_head_DB()
                hdf5_head_face_DB = h5py.File(self.hd_file_head_face, "a")
        return hdf5_head_face_DB

    def add_new_row_2_splitted_df(self, data, start_index, end_index, label, counter, df, subject, movie=None,
                                  is_not_used_sections=True):
        new_row = pd.DataFrame(data[start_index: end_index], columns=Preparation.head_face_features_by_order)
        new_row["label"] = label
        new_row["section"] = counter
        new_row["Participant"] = subject
        if is_not_used_sections:
            new_row["Movie"] = movie
        df = df.append(new_row, sort=False, ignore_index=True)
        counter += 1
        return counter, df

    def split_sub_df(self, start_index, end_index, df, movie, subject, not_used_sections_counter, sections_counter,
                     not_used_sections_df, all_data_splitted_df):
        # df = s_data_head_face
        start_indexs_2_split_not_hazard = np.arange(start_index, end_index, self.section_length)
        for i, start_index in enumerate(start_indexs_2_split_not_hazard):
            if i == start_indexs_2_split_not_hazard.size - 1:
                if end_index - start_index < self.section_length:
                    not_used_sections_counter, not_used_sections_df = self.add_new_row_2_splitted_df(data=df,
                                                                                                     start_index=start_index,
                                                                                                     end_index=end_index,
                                                                                                     label=0,
                                                                                                     counter=not_used_sections_counter,
                                                                                                     df=not_used_sections_df,
                                                                                                     movie=movie,
                                                                                                     subject=subject,
                                                                                                     is_not_used_sections=True)
                else:
                    sections_counter, all_data_splitted_df = self.add_new_row_2_splitted_df(data=df,
                                                                                            start_index=start_index,
                                                                                            end_index=end_index,
                                                                                            label=0,
                                                                                            counter=sections_counter,
                                                                                            df=all_data_splitted_df,
                                                                                            subject=subject,
                                                                                            is_not_used_sections=False)
                continue
            sections_counter, all_data_splitted_df = self.add_new_row_2_splitted_df(data=df,
                                                                                    start_index=start_index,
                                                                                    end_index=
                                                                                    start_indexs_2_split_not_hazard[
                                                                                        i + 1], label=0,
                                                                                    counter=sections_counter,
                                                                                    df=all_data_splitted_df,
                                                                                    subject=subject,
                                                                                    is_not_used_sections=False)
        return not_used_sections_counter, sections_counter, not_used_sections_df, all_data_splitted_df

    def calculate_start_end_index_hazard_sections(self, hazard_time_milis, data):
        # data = s_data_head_face
        hazard_start_end_dict = {}  # key = num_hazard, value = [start index, end index]
        for num_hazard in range(0, hazard_time_milis.size):
            # num_hazard = range(0, hazard_time_milis.size)[0]
            # num_hazard = 0
            closet_time_2_hazard_time_index = np.abs(data["MediaTime"] - int(hazard_time_milis[0][num_hazard])).idxmin()
            hazard_section_start = closet_time_2_hazard_time_index - self.section_length + 1
            hazard_section_end = closet_time_2_hazard_time_index + 1
            hazard_start_end_dict[num_hazard] = [hazard_section_start, hazard_section_end]
        return hazard_start_end_dict

    def split_non_hazards_section(self, start_index_section, end_index_section, data, movie, subject,
                                  not_used_sections_counter, sections_counter, not_used_sections_df,
                                  all_data_splitted_df):
        prevew_section_size = end_index_section - start_index_section
        if prevew_section_size > 0:  # true = there is a sections 2 split before hazard section
            if prevew_section_size < self.section_length:
                not_used_sections_counter, not_used_sections_df = self.add_new_row_2_splitted_df(
                    data=data, start_index=start_index_section, end_index=end_index_section,
                    label=0,
                    counter=not_used_sections_counter, df=not_used_sections_df, movie=movie,
                    subject=subject,
                    is_not_used_sections=True)
            else:
                not_used_sections_counter, sections_counter, not_used_sections_df, all_data_splitted_df = self.split_sub_df(
                    start_index=start_index_section, end_index=end_index_section, df=data,
                    movie=movie, subject=subject,
                    not_used_sections_counter=not_used_sections_counter,
                    sections_counter=sections_counter, not_used_sections_df=not_used_sections_df,
                    all_data_splitted_df=all_data_splitted_df)
        return not_used_sections_counter, sections_counter, not_used_sections_df, all_data_splitted_df

    def split_df_by_hazard(self, hazard_time_milis, data, movie, subject, not_used_sections_counter, sections_counter,
                           not_used_sections_df, all_data_splitted_df):
        # data = s_data_head_face
        # movie = m
        # subject = s
        hazard_start_end_dict = self.calculate_start_end_index_hazard_sections(hazard_time_milis, data)
        for num_hazard in range(0, hazard_time_milis.size):
            # num_hazard = range(0, hazard_time_milis.size)[0]
            # num_hazard=0
            start_index_hazard_section = hazard_start_end_dict[num_hazard][0]
            end_index_hazard_section = hazard_start_end_dict[num_hazard][1]
            if num_hazard == hazard_time_milis.size - 1:
                start_index_next_hazard_section = data.shape[0]
            else:
                start_index_next_hazard_section = hazard_start_end_dict[num_hazard + 1][0]
            if num_hazard == 0:
                end_index_prevew_hazard_section = 0
            else:
                end_index_prevew_hazard_section = hazard_start_end_dict[num_hazard - 1][1]
            if start_index_hazard_section < 0:  # hazard_section < section_length + no sections 2 split before current hazard section
                not_used_sections_counter, not_used_sections_df = self.add_new_row_2_splitted_df(data=data,
                                                                                                 start_index=0,
                                                                                                 end_index=end_index_hazard_section,
                                                                                                 label=1,
                                                                                                 counter=not_used_sections_counter,
                                                                                                 df=not_used_sections_df,
                                                                                                 movie=movie,
                                                                                                 subject=subject,
                                                                                                 is_not_used_sections=True)
            else:
                sections_counter, all_data_splitted_df = self.add_new_row_2_splitted_df(data=data,
                                                                                        start_index=start_index_hazard_section,
                                                                                        end_index=end_index_hazard_section,
                                                                                        label=1,
                                                                                        counter=sections_counter,
                                                                                        df=all_data_splitted_df,
                                                                                        subject=subject,
                                                                                        is_not_used_sections=False)
                if num_hazard == 0:
                    # handle sections before hazard section
                    start_index_prevew_section = end_index_prevew_hazard_section
                    end_index_prevew_section = start_index_hazard_section
                    not_used_sections_counter, sections_counter, not_used_sections_df, all_data_splitted_df = self.split_non_hazards_section(
                        start_index_section=start_index_prevew_section, end_index_section=end_index_prevew_section,
                        data=data, movie=movie, subject=subject, not_used_sections_counter=not_used_sections_counter,
                        sections_counter=sections_counter, not_used_sections_df=not_used_sections_df,
                        all_data_splitted_df=all_data_splitted_df)

                # handle sections after hazard section
                start_index_prevew_section = end_index_hazard_section
                end_index_prevew_section = start_index_next_hazard_section
                not_used_sections_counter, sections_counter, not_used_sections_df, all_data_splitted_df = self.split_non_hazards_section(
                    start_index_section=start_index_prevew_section, end_index_section=end_index_prevew_section,
                    data=data, movie=movie, subject=subject, not_used_sections_counter=not_used_sections_counter,
                    sections_counter=sections_counter, not_used_sections_df=not_used_sections_df,
                    all_data_splitted_df=all_data_splitted_df)
        return not_used_sections_counter, sections_counter, not_used_sections_df, all_data_splitted_df

    def split_data_by_reaction_time(self):
        # p=Preparation(reaction_time=1.7)
        hdf5_head_face_DB = self.get_hd_file_head_face()
        sections_counter = 0
        not_used_sections_counter = 0
        all_data_splitted_df = pd.DataFrame([], columns=["section", "label"] + Preparation.head_face_features_by_order)
        not_used_sections_df = pd.DataFrame([], columns=["Movie", "Participant", "label",
                                                         "section"] + Preparation.head_face_features_by_order)
        features_2_save = ['label', 'section'] + Preparation.head_face_features_by_order
        splitted_data_DB = os.path.join(Preparation.data_dir, "reaction_time_{:s}_splitted_data_DB.hd5".format(str(self.reaction_time)))
        hd = h5py.File(splitted_data_DB, "a")
        for m in self.mIDs:
            # m=p.mIDs[0]
            print(m)
            subjects = list(hdf5_head_face_DB[m]["processed_data"].keys())
            if m in hd:
                if "not_used_splitted_data" in hd[m]:
                    del hd[m]["not_used_splitted_data"]
                not_used_splitted_data = hd[m].create_group("not_used_splitted_data")
            else:
                not_used_splitted_data = hd.create_group(m + "/not_used_splitted_data")
            for s in subjects:
                # s = subjects[0]
                print(s)
                hazard_m_data = pd.read_excel(self.hazardPressesQualtrics, sheet_name=m)
                s_hazards = hazard_m_data[hazard_m_data["Participant"] == int(s)]
                s_hazards = s_hazards.drop(Preparation.cols_2_drop_hazardPressesQualtrics_file, axis=1)
                s_hazards.dropna(inplace=True, axis=1)
                s_hazards = pd.DataFrame([np.unique(s_hazards.values)])
                if s_hazards.shape[1] == 0:
                    continue
                elif s_hazards.shape[1] > 1:
                    s_hazards_diff = s_hazards.diff(axis=1)
                    s_hazards = s_hazards.iloc[:, [0] + np.where(s_hazards_diff >= self.reaction_time)[1].tolist()]
                    s_hazards = s_hazards.values
                else:
                    s_hazards = s_hazards.values
                hazard_time_milis = np.sort(np.round(s_hazards * Preparation.convert_sec_2_millis, 0))
                s_data_head_face = pd.DataFrame(hdf5_head_face_DB[m]["processed_data"][s][()],
                                                columns=Preparation.head_face_features_by_order)
                not_used_sections_counter, sections_counter, not_used_sections_df, all_data_splitted_df = self.split_df_by_hazard(
                    hazard_time_milis=hazard_time_milis, data=s_data_head_face, movie=m, subject=s,
                    not_used_sections_counter=not_used_sections_counter, sections_counter=sections_counter,
                    not_used_sections_df=not_used_sections_df, all_data_splitted_df=all_data_splitted_df)
                not_used_sections_df[['label', 'section']] = not_used_sections_df[['label', 'section']].apply(
                    pd.to_numeric)
                not_used_splitted_data.create_dataset(s, data=not_used_sections_df[features_2_save])

        if "splitted_data" in hd:
            del hd["splitted_data"]
            splitted_data = hd.create_group("splitted_data")
        else:
            splitted_data = hd.create_group("splitted_data")
        all_data_splitted_df = all_data_splitted_df.apply(pd.to_numeric)
        splitted_data.create_dataset("splitted_data_2_use", data=all_data_splitted_df)
        splitted_data.attrs.create("reaction_time", self.reaction_time)
        splitted_data.attrs.create("features_names", list(all_data_splitted_df.columns))
        splitted_data.attrs.create("section_length", self.section_length)
        splitted_data.attrs.create("fps", self.fps)
        splitted_data.attrs.create("sections_counter", sections_counter)
        hd.close()
        hdf5_head_face_DB.close()
        return not_used_sections_counter, sections_counter, not_used_sections_df, all_data_splitted_df

def save_only_sections_without_NaN():
    string_2_look_for = "reaction_time"
    hdf5_files_names_list = sorted(list(set([hdf5_dir for hdf5_dir in os.listdir(Preparation.data_dir) if (string_2_look_for in hdf5_dir) & (hdf5_dir.endswith(".hd5"))])))
    for hdf5_dir in hdf5_files_names_list:
        hd_dir = os.path.join(Preparation.data_dir, hdf5_dir)
        print(hdf5_dir)
        #hd_dir = "C:\\Users\Coral\\facial head behavior and road hazard perceptions\\data\\reaction_time_1.65_splitted_data_DB.hd5"
        hd = h5py.File(hd_dir, "a")
        hd_DB = hd["splitted_data"]
        features_name = hd_DB.attrs["features_names"]
        data = pd.DataFrame(data = hd_DB["splitted_data_2_use"][()], columns=features_name)
        no_NaN_data = pd.DataFrame([], columns=features_name)
        section_length = hd_DB.attrs["section_length"]
        data_without_nan = data.dropna()
        data_group_by_sections_label_count = data_without_nan.groupby(by='section')["label"].count()
        sections_id_list = data_group_by_sections_label_count.index.tolist() # index represent section ID
        for section_id in sections_id_list:
            #section_id = sections_id_list[0]
            if data_group_by_sections_label_count[section_id] != section_length:
                continue
            no_NaN_data = no_NaN_data.append(data[data['section']==section_id], ignore_index=True, sort=False)
        if "no_NaN_splitted_data" in hd:
            del hd["no_NaN_splitted_data"]
            no_NaN_splitted_data = hd.create_group("no_NaN_splitted_data")
        else:
            no_NaN_splitted_data = hd.create_group("no_NaN_splitted_data")
        no_NaN_data = no_NaN_data.apply(pd.to_numeric)
        no_NaN_splitted_data.create_dataset("splitted_data_2_use", data=no_NaN_data)
        no_NaN_splitted_data.attrs.create("reaction_time", hd_DB.attrs["reaction_time"])
        no_NaN_splitted_data.attrs.create("features_names", hd_DB.attrs["features_names"])
        no_NaN_splitted_data.attrs.create("section_length", section_length)
        no_NaN_splitted_data.attrs.create("fps", hd_DB.attrs["fps"])
        no_NaN_splitted_data.attrs.create("sections_counter", hd_DB.attrs["sections_counter"])
        hd.close()
        return no_NaN_data

