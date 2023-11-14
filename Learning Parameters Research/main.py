import os
import zipfile
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import svm
import xgboost as xgb

class MLModel:
    def __init__(self):
        self.__data = None
        self.df = None
        self.X_train = None 
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clf = None
        self.map_dictionary = {}
    
    def setup(self, target_directory):
        os.system('kaggle datasets download -d wanghaohan/confused-eeg')
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        zip_file_path = 'confused-eeg.zip'
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_directory)
        print(f"Successfully extracted '{zip_file_path}' to '{target_directory}'.")
    
    def setup_dataframe(self, include_demographics=False, print_stats=False, remove_features=[]):
        data = pd.read_csv("./data/EEG_data.csv")
        if include_demographics:
            demo_df = pd.read_csv('./data/demographic_info.csv')
            demo_df = demo_df.rename(columns={'subject ID': 'SubjectID'})
            data = data.merge(demo_df, how='inner', on='SubjectID')
            data = pd.get_dummies(data)
        self.__data = data
        if print_stats:
            self.__data.info()
        self.df = pd.DataFrame(self.__data)
        self.remove_features(remove_features)

        return self.df

    def split(self, ratio):
        X_int = self.df.drop('user-definedlabeln', axis=1).values
        Y_int = self.df['user-definedlabeln'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_int, Y_int, test_size=ratio, random_state=42)
        for index, value in enumerate(self.X_train):
            self.map_dictionary[tuple(self.X_train[index])]=self.y_train[index]
        for index, value in enumerate(self.X_test):
            self.map_dictionary[tuple(self.X_test[index])]=self.y_test[index]

    def random_forest_model(self, show_plot=False):
        accuracies = []
        estimators = []
        for i in range(40, 45):
            clf = RandomForestClassifier(n_estimators=i, max_depth=2, random_state=13)
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            score = clf.score(self.X_test, self.y_test)
            accuracies.append(score)
            estimators.append(i)

        if show_plot:
            self.plot_accuracies(accuracies, range(25, 50), 'Random Forest Classifier Accuracy vs. n_estimators')

        maxi_acc = max(accuracies)
        i = 0
        optimal_estimator = estimators[i]
        for acc in accuracies:
            if acc==maxi_acc:
                optimal_estimator = estimators[i]
            i = i+1
        
        self.clf = RandomForestClassifier(n_estimators=optimal_estimator,max_depth=2,random_state=13)
        return max(accuracies)

    def svm_model(self):
        clf = svm.SVC()
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        self.clf=clf
        return clf.score(self.X_test, self.y_test)

    def boosted_trees(self, show_plot=False):
        accuracies = []
        estimators = []
        for i in range(1, 5):
            xg = xgb.XGBClassifier(objective='binary:logistic', n_estimators=i, seed=1)
            xg.fit(self.X_train, self.y_train)
            predict = xg.predict(self.X_test)
            accuracies.append(xg.score(self.X_test, self.y_test))
            estimators.append(i)

        if show_plot:
            self.plot_accuracies(accuracies, range(1, 5), 'Boosted Accuracy vs. n_estimators')

        maxi_acc = max(accuracies)
        i = 0
        optimal_estimator = estimators[i]
        for acc in accuracies:
            if acc==maxi_acc:
                optimal_estimator = estimators[i]
            i = i+1
        
        self.clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=optimal_estimator, seed=1)

        return max(accuracies)

    def return_data(self):
        return self.__data

    def plot_correlation_matrix(self):
        plt.figure(figsize=(15, 15))
        corr_matrix = self.df.corr()
        seaborn.heatmap(corr_matrix, vmin=-1.0, square=True, annot=True)

    def plot_accuracies(self, accuracies, x_values, title):
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, accuracies, marker='o', linestyle='-', color='b')
        plt.title(title)
        plt.xlabel('Ratios' if 'Ratios' in title else 'Features Removed')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()
    
    def remove_features(self, feature_array):
        self.df.drop(feature_array,axis = 1,inplace=True)
    
    def remove_and_boost(self,features):
        self.remove_features(features)
        accuracies = []
        ratio_arr = []
        ratio_i = 0.1
        while ratio_i < 0.9:
            ratio_arr.append(ratio_i)
            m.split(ratio=ratio_i)
            accuracies.append(m.boosted_trees())
            ratio_i += 0.1
        return max(accuracies)
    
    def remove_and_svm(self,features):
        self.remove_features(features)
        accuracies = []
        ratio_arr = []
        ratio_i = 0.1
        while ratio_i < 0.9:
            ratio_arr.append(ratio_i)
            m.split(ratio=ratio_i)
            accuracies.append(m.svm_model())
            ratio_i += 0.1
        return max(accuracies)
    
    def remove_and_forest(self,features):
        self.remove_features(features)
        accuracies = []
        ratio_arr = []
        ratio_i = 0.1
        while ratio_i < 0.9:
            ratio_arr.append(ratio_i)
            m.split(ratio=ratio_i)
            accuracies.append(m.random_forest_model())
            ratio_i += 0.025
        return max(accuracies)
    
    def return_model(self):
        return self.clf
    
    def get_top_n_samples(self, n):
        self.clf.fit(m.X_train, m.y_train)  # Train your model
        print(self.clf)
        # Get the predicted probabilities for class 0 and class 1
        probabilities = self.clf.predict_proba(self.X_test)
        # Calculate the absolute differences between class 0 and class 1 probabilities
        abs_differences = abs(probabilities[:, 0] - probabilities[:, 1])
        # Sort the absolute differences and get the indices of the sorted elements
        sorted_indices = abs_differences.argsort()
        # Retrieve the top n entries from self.X_test based on the sorted indices
        top_n_samples = self.X_test[sorted_indices[:n]]

        y_samples = []
        for i in top_n_samples:
            y_samples.append(self.map_dictionary[tuple(i)])
        return top_n_samples,y_samples

def sorted_correlation(data, output_feature):
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

def reverse(lst):
    new_lst = lst[::-1]
    return new_lst

if __name__ == "__main__":
    m = MLModel()
    m.setup('data')
    df = m.setup_dataframe(include_demographics=False, print_stats=False)
    data = m.return_data()
    print(df.columns)
    print(data['user-definedlabeln'].unique())
    print(len(data))
    # m.plot_correlation_matrix()
    # print("Random Forest Model, considering all the features")
    # # Random Forest Model, considering all the features
    # accuracies = []
    # ratio_arr = []
    # ratio_i = 0.05
    # while ratio_i < 1:
    #     ratio_arr.append(ratio_i)
    #     m.split(ratio=ratio_i)
    #     accuracies.append(m.random_forest_model())
    #     ratio_i += 0.025
    # m.plot_accuracies(accuracies, ratio_arr, 'Random Forest Model: Accuracies vs. Ratios')
    # print(ratio_arr)
    # print(accuracies)

    print(sorted_correlation(data=data, output_feature='user-definedlabeln'))
    corr_arr = sorted_correlation(data=data, output_feature='user-definedlabeln')
    corr_arr = reverse(corr_arr)

    # # SVM Machine, considering all the features
    # accuracies = []
    # ratio_arr = []
    # ratio_i = 0.1
    # while ratio_i < 0.9:
    #     ratio_arr.append(ratio_i)
    #     m.split(ratio=ratio_i)
    #     accuracies.append(m.svm_model())
    #     ratio_i += 0.1
    # m.plot_accuracies(accuracies, ratio_arr, 'SVM Model: Accuracies vs. Ratios')
    # print(ratio_arr)
    # print(accuracies)

    # # Boosted Trees, considering all the features
    # accuracies = []
    # ratio_arr = []
    # ratio_i = 0.1
    # while ratio_i < 0.9:
    #     ratio_arr.append(ratio_i)
    #     m.split(ratio=ratio_i)
    #     accuracies.append(m.boosted_trees())
    #     ratio_i += 0.1
    # m.plot_accuracies(accuracies, ratio_arr, 'Boosted Trees (XGBoost) Model: Accuracies vs. Ratios')
    # print(ratio_arr)
    # print(accuracies)

    # df_temp = df.copy()
    # accuracies = []
    # number_removed = []
    # i=0
    # while i<len(corr_arr)-1:
    #     accuracies.append(m.remove_and_boost(corr_arr[i:i+1]))
    #     number_removed.append(i+1)
    #     i = i+1

    # df=df_temp.copy()
    # m.plot_accuracies(accuracies, number_removed, 'Boosted Trees (XGBoost) Model: Accuracies vs. Features Removed')
    # print(number_removed)
    # print(accuracies)

    # m.setup_dataframe(include_demographics=False,print_stats=False)
    
    # accuracies = []
    # removed_tracker = []
    # number_removed = []
    # i=0
    # while i<len(corr_arr)-1:
    #     removed_tracker.append(corr_arr[i:i+1])
    #     print("Working with the Subset Removed: {}".format(removed_tracker))
    #     accuracies.append(m.remove_and_svm(corr_arr[i:i+1]))
    #     number_removed.append(i+1)
    #     i = i+1
    # df=df_temp.copy()
    # m.plot_accuracies(accuracies, number_removed, 'SVM Model: Accuracies vs. Features Removed')
    # print(number_removed)
    # print(accuracies)

    # m.setup_dataframe(include_demographics=False,print_stats=False)

    # accuracies = []
    # removed_tracker = []
    # number_removed = []
    # i=0
    # while i<len(corr_arr)-1:
    #     removed_tracker.append(corr_arr[i:i+1])
    #     print("Working with the Subset Removed: {}".format(removed_tracker))
    #     accuracies.append(m.remove_and_forest(corr_arr[i:i+1]))
    #     number_removed.append(i+1)
    #     i = i+1
    # df=df_temp.copy()
    # m.plot_accuracies(accuracies, number_removed, 'Forest Model: Accuracies vs. Features Removed')
    # print(number_removed)
    # print(accuracies)

    m.split(ratio=0.67)
    print("Accuracy is {}".format(m.random_forest_model()))

    m.setup_dataframe(include_demographics=False,print_stats=False)

    X_train = m.X_train
    y_train = m.y_train
    print(len(X_train))
    print("Ratio of train/test is {}".format(len(m.X_train)/len(data)))

    clf = m.return_model()
    clf.fit(X_train, y_train)  # Train your model
    print(clf)

    print(clf.score(m.X_test, m.y_test))
    # uncertainty_scores = clf.predict_proba(m.X_test)
    # print(uncertainty_scores)

    top_samples, y_samples = m.get_top_n_samples(10)
    print(clf.predict_proba(top_samples))
    print(top_samples)
    print(y_samples)

    my_2d_list = top_samples
    # Calculate the number of rows and columns
    num_rows = len(my_2d_list)
    num_columns = len(my_2d_list[0])

    # Print the shape
    print(f"Shape of the top samples: ({num_rows}, {num_columns})")
    print(f"Shape of the inital X_train: ({len(X_train)}, {len(X_train[0])})")

    print(type(X_train))
    print(X_train)
    print(type(top_samples))

    print("Size before appending is {}".format(len(X_train)))
    # working with 10 samples first
    #for i in range(1, len(top_samples)):
    X_train = np.concatenate((X_train, top_samples), axis=0)
        #X_train = np.concatenate(X_train,i)
        # X_train = X_train.concatenate(np.ndarray(i))
    print("Size after appending is {}".format(len(X_train)))
    print("Y Size before appending is {}".format(len(y_train)))

    y_train = np.concatenate((y_train,y_samples), axis =0)
    print("Y Size after appending is {}".format(len(y_train)))

    clf = None
    clf = m.return_model()
    clf.fit(X_train, y_train)  # Train your model
    print(clf)

    # uncertainty_scores = clf.predict_proba(m.X_test)
    # print(uncertainty_scores)

    print(clf.score(m.X_test, m.y_test))