import os, glob, re
import pandas as pd

def mfile2mid(fname):
    if re.match(r'm_\d+\.[\w\d]{2,4}$',fname):
        return re.sub("_0?([0-9]+)\..*",r'\1',fname)
    return None

class RawDataCreation:
    """
    This class split each participant's absolute raw data (of iMotion) into movie segments (txt format).
    The raw data is assumed to contain the ID's subject on disk in txt format files.
    The absolute raw data is under <root>/data/raw_data/allmovies_1file/**<subject_id>**.txt
    The absolute raw data after dropping double rows is under <root>/data/raw_data/1file_removedDoubleRows/**<subject_id>**.txt
    """
    root = re.sub("[\/\\\][^\/\\\]+$", "", os.path.abspath('')) # gets the directory of ISC folder
    datadir = os.path.join(root, "data") # gets the directory of "data" folder
    rawDatadir = os.path.join(datadir, "raw_data") # gets the directory of "raw_data" folder in "data" folder
    fpat = re.compile("[\\\/](\d+)\.txt")  # a regular expression pattern for identifying subjects' text files

    def dropDoubleRows(self):
        """
            TO DO
            Its not really dropping - I need to decide how to deal with ittntkv
            :return:
            """
        allmovies_1filedir = os.path.join(RawDataCreation.rawDatadir,"allmovies_1file")  # getting the wd of "allmovies_1file" (all the origin output of Affectiva) folder in "data" folder
        subjids = sorted(list(set([fileName for fileName in os.listdir(allmovies_1filedir) if fileName.endswith(".txt")])))
        rows2DeleteMap = pd.DataFrame(columns=['ID', 'rowIndex2Delete'])
        for s in subjids:
            #s = '001_243.txt'
            print(s)
            s_dir = os.path.join(allmovies_1filedir, s)
            # relevantCols = ['Name', 'FrameIndex', 'LiveMarker', "Pitch", "Yaw", "Roll"]
            s_data = pd.read_csv(s_dir, sep="\t", skiprows=5, header=0)
            # identify double index rows (because of pressing the button)
            doubleRowsIndexs = s_data[s_data["LiveMarker"] == "Identify"].index - 1
            rows2Delete = pd.DataFrame()
            if not doubleRowsIndexs.empty:
                rows2Delete['rowIndex2Delete'] = doubleRowsIndexs + 2
                rows2Delete["ID"] = s
                rows2DeleteMap = rows2DeleteMap.append(rows2Delete, sort=False)
                # deleting double rows of hazard identification
                s_data.drop(doubleRowsIndexs, inplace=True)
            writeMovie_dir = os.path.join(RawDataCreation.rawDatadir,
                                          "1file_removedDoubleRows")  # the dir of the specific raw data in specific movie
            if not os.path.isdir(writeMovie_dir):
                try:
                    os.mkdir(writeMovie_dir)
                except OSError:
                    print("Creation of the directory %s failed" % writeMovie_dir)
                else:
                    print("Successfully created the directory %s " % writeMovie_dir)
            writeFile = os.path.join(writeMovie_dir, s)
            s_data.to_csv(writeFile, sep="\t", index=False)


        rows2DeleteMap.to_csv(os.path.join(RawDataCreation.datadir, "rows2DeleteMap_1file.csv"), index=False)
        return rows2DeleteMap

    def saveDataByMovies(self, data, movie, participant, Destinationfolder):
        """

        :param Destinationfolder: before pre processing missing values "data2PreProcess"
                                  after pre processing missing values and to split participant's data to exact movies sections"datadir"
                                  after pre processing missing values and before split participant's data to exact movies sections: "rawDatadir"
        :return:
        """
        if Destinationfolder == "data2PreProcess":
            Destinationfolder_dir = os.path.join(RawDataCreation.rawDatadir, Destinationfolder)  # getting the wd of "raw_data" (all the origin output of Affectiva) folder in "data" folder
        elif Destinationfolder == "rawDatadir":
            Destinationfolder_dir = RawDataCreation.rawDatadir
        else:
            Destinationfolder_dir = RawDataCreation.datadir
        if not os.path.isdir(Destinationfolder_dir):
            try:
                os.mkdir(Destinationfolder_dir)
            except OSError:
                print("Creation of the directory %s failed" % Destinationfolder_dir)
            else:
                print("Successfully created the directory %s " % Destinationfolder_dir)
        writeMovie_dir = os.path.join(Destinationfolder_dir, movie)  # the dir of the specific raw data in specific movie
        if not os.path.isdir(writeMovie_dir):
            try:
                os.mkdir(writeMovie_dir)
            except OSError:
                print("Creation of the directory %s failed" % writeMovie_dir)
            else:
                print("Successfully created the directory %s " % writeMovie_dir)
        writeFile = os.path.join(writeMovie_dir, participant + ".txt")
        data.to_csv(writeFile, sep="\t", index=False)

    def createRawData(self, addStart=1000, addEnd=1000, Destinationfolder="data2PreProcess"):
        """
        data2PreProcess datadir rawDatadir
        :param addStart: default value for data before pre processing of missing values
                        for data after pre processing missing values I used 0
        :param addEnd: default value for data before pre processing of missing values
                        for data after pre processing missing values I used 0
        :param Destinationfolder: default value for before after pre processing of missing values
                                for data after pre processing missing values I used "datadir"
        :return:
        """
        if Destinationfolder == "data2PreProcess":
            origin_dir = os.path.join(RawDataCreation.rawDatadir, "1file_removedDoubleRows")  # getting the directory of "1file_removedDoubleRows"
        elif Destinationfolder == "rawDatadir":
            origin_dir = os.path.join(RawDataCreation.rawDatadir,"data2PreProcess")  # getting the directory of "data2PreProcess"
        else: # Destinationfolder == "datadir"
            origin_dir = RawDataCreation.rawDatadir # getting the directory of "raw_data" - data after pre processing missing values before the exact movie split

        # first we need to load the excel file "timing2split" for knowing the positions to cut the raw data of Affectiva for each participant
        timing2split_data = pd.read_excel("C:\\Users\\Coral\\Desktop\\timing2split.xlsx",sheet_name="screenRecored_affectivaTime")
        nrow = timing2split_data.shape[0]
        ncol = timing2split_data.shape[1]
        for col in range(2, ncol):
            #col=5
            movie = timing2split_data.columns[col]
            print(movie)
            if (movie == "m2") | (movie == "m8"):
                continue
            #data = timing2split_data.iloc[:, col]

            if (Destinationfolder == "datadir"):
                mIDdir = os.path.join(origin_dir, movie)
                fpat = RawDataCreation.fpat
                currentSubjID = sorted(list(set([re.findall(fpat, f)[0] for f in glob.glob(os.path.join(mIDdir, "*.txt"))])))

            for subj in range(0, nrow, 2):
                #subj = 0
                participant = str(timing2split_data.iloc[subj, 0])  # get ID participant
                if (Destinationfolder == "datadir"):
                    if participant not in currentSubjID:
                        continue
                print(timing2split_data.iloc[subj, 0])
                if Destinationfolder == "data2PreProcess":
                    origin_file_name = str(glob.glob(os.path.join(origin_dir, "*_" + participant + ".txt"))[0])  # get origin raw data of Affectiva for specific participant
                else:
                    origin_file_name = os.path.join(origin_dir, movie, participant + ".txt")  # get origin raw data of Affectiva for specific participant
                #print(origin_file_name)
                movie_times = timing2split_data.loc[:, ["participant", "part", movie]]
                start = movie_times[(movie_times["participant"] == int(participant)) & (movie_times["part"] == "affectiva_start")] # the start point for this movie of this participant
                end = movie_times[(movie_times["participant"] == int(participant)) & (movie_times["part"] == "affectiva_end")] # the end point for this movie of this participant
                start = int(start[movie]) - addStart
                end = int(end[movie]) + addEnd

                subj_rawData = pd.read_csv(origin_file_name, sep="\t", skiprows=5)
                cuttedData = subj_rawData[(subj_rawData['MediaTime'] > start) & (subj_rawData['MediaTime'] < end)]  # cutting the data
                #subj_rawData = subj_rawData[(subj_rawData['MediaTime'] > start) & (subj_rawData['MediaTime'] < end)] # cutting the data
                self.saveDataByMovies(data=cuttedData, movie=movie, participant=participant, Destinationfolder=Destinationfolder)


"""
    def checkNrowsSubMovie(self):
        # first we need to load the excel file "timing2split" for knowing the positions to cut the raw data of Affectiva for each participant
        timing2split_data = pd.read_excel("C:\\Users\\Coral\\Desktop\\timing2split.xlsx",
                                          sheet_name="screenRecored_affectivaTime")
        missingValuesData_dir = os.path.join(RawDataCreation.datadir, "missingValuesData")
        movieID = timing2split_data.columns
        nrowMovie = pd.DataFrame()
        for m in movieID:
            # m = 'm2'
            if (m == "participant") | (m == "part") | (m == "m2") | (m == "m8"):
                continue
            mIDdir = os.path.join(RawDataCreation.rawDatadir, m)
            subjID = sorted(list(set([re.findall(RawDataCreation.fpat, f)[0] for f in glob.glob(os.path.join(mIDdir, "*.txt"))])))
            temp_nrow_list = []
            print("*****_", m, "_*****")
            for sub in subjID:
                # sub = 201
                rawfile = os.path.join(RawDataCreation.rawDatadir, m, str(sub) + ".txt")  # the dir of the specific raw data in specific movie
                data = pd.read_csv(rawfile, sep="\t", header=0)
                temp_nrow_list.append(data.shape[0])
                #print(data.shape[0])
                # print("sub ", sub, " nrow ", data.shape[0])
                data = data[(data['Pitch'].isnull()) & (data['Yaw'].isnull())]  # cutting the data
                if not data.empty:
                    writeMovie_dir = os.path.join(missingValuesData_dir,m)  # the dir of the specific raw data in specific movie
                    if not os.path.isdir(writeMovie_dir):
                        try:
                            os.mkdir(writeMovie_dir)
                        except OSError:
                            print("Creation of the directory %s failed" % writeMovie_dir)
                        else:
                            print("Successfully created the directory %s " % writeMovie_dir)
                    writeFile = os.path.join(writeMovie_dir, str(sub) + ".txt")
                    data.to_csv(writeFile, sep="\t", index=True)
            nrowMovie[m] = pd.Series(temp_nrow_list)
        return nrowMovie
"""
