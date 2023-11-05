import os
import zipfile
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class MLModel:
    def __init__(self):
       self.__data = None
       self.df = None
       self.X_train = None 
       self.X_test = None
       self.y_train = None
       self.y_test = None
    
    def setup(self,target_directory):
        os.system('kaggle datasets download -d wanghaohan/confused-eeg')
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        zip_file_path = 'confused-eeg.zip'
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_directory)
        print(f"Successfully extracted '{zip_file_path}' to '{target_directory}'.")
    
    def setup_dataframe(self,include_demographics=False,print_stats=False):
        data = pd.read_csv("./data/EEG_data.csv")
        if include_demographics:
            demo_df = pd.read_csv('./data/demographic_info.csv')
            demo_df = demo_df.rename(columns = {'subject ID': 'SubjectID'})
            data = data.merge(demo_df,how = 'inner',on = 'SubjectID')
            data = pd.get_dummies(data)
        self.__data = data
        if print_stats:
            self.__data.info()
        self.df=pd.DataFrame(self.__data)
        return self.df

    def split(self,ratio):
        X_int=df.drop('user-definedlabeln',axis=1).values
        Y_int=df['user-definedlabeln'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_int,Y_int, test_size=ratio,random_state=42)

    def random_forest_model(self,show_plot = False):
        # Data collection
        accuracies = []
        for i in range(25, 50):
            clf = RandomForestClassifier(n_estimators=i, max_depth=2, random_state=13)
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            score = clf.score(self.X_test, self.y_test)
            accuracies.append(score)

        if show_plot:
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(25, 50), accuracies, marker='o', linestyle='-', color='b')
            plt.title('Random Forest Classifier Accuracy vs. n_estimators')
            plt.xlabel('n_estimators')
            plt.ylabel('Accuracy')
            plt.grid(True)
            # Show the plot
            plt.show()
        
        return max(accuracies)

    def return_data(self):
        return self.__data

def sorted_correlation(data,output_feature):
    output_variable = data[output_feature]
    correlations = []
    for column in data.columns:
        if column != output_feature:
            feature = data[column]
            correlation = feature.corr(output_variable)
            correlations.append((column, correlation))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    correlations_array = np.array(correlations)
    print(correlations_array)
    feature_names_array = np.array([item[0] for item in correlations_array])
    return feature_names_array

if __name__ == "__main__":
    m = MLModel()
    m.setup('data')
    df=m.setup_dataframe(include_demographics=True,print_stats=False)
    data=m.return_data()
    print(df.columns)
    print(data['user-definedlabeln'].unique())
    plt.figure(figsize = (15,15))
    
    # Plotting the Correlation Matrix
    corr_matrix = df.corr()
    seaborn.heatmap(corr_matrix,vmin = -1.0, square=True, annot = True)
    
    # Random Forest Model, considering all the features
    accuracies = []
    ratio_arr = []
    ratio_i = 0.05
    while ratio_i < 1:
        ratio_arr.append(ratio_i)
        m.split(ratio=ratio_i)
        accuracies.append(m.random_forest_model())
        ratio_i=ratio_i+0.025
    # Plotting the Random Forest Model
    plt.figure(figsize=(10, 6))
    plt.plot(ratio_arr, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracies vs. Ratios')
    plt.xlabel('Ratios')
    plt.ylabel('Accuracies')
    plt.grid(True)
    plt.show()
    print(ratio_arr)
    print(accuracies)

    print(sorted_correlation(data=data,output_feature='user-definedlabeln'))