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
        self.X_train = None 
        self.X_test = None
        self.y_train = None
        self.y_test = None
        X_int = self.df.drop('user-definedlabeln', axis=1).values
        Y_int = self.df['user-definedlabeln'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_int, Y_int, test_size=ratio)
        for index, value in enumerate(self.X_train):
            self.map_dictionary[tuple(self.X_train[index])]=self.y_train[index]
        for index, value in enumerate(self.X_test):
            self.map_dictionary[tuple(self.X_test[index])]=self.y_test[index]

    # Function:- Choose the best n_estimators parameter to train the RandomForestClassifier and plot the accuracies if required.
    def random_forest_model(self, show_plot=False):
        accuracies = []
        estimators = []
        for i in range(40, 45):
            clf = RandomForestClassifier(n_estimators=i, max_depth=2)
            clf.fit(self.X_train, self.y_train)
            clf.predict(self.X_test)
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
        
        self.clf = RandomForestClassifier(n_estimators=optimal_estimator,max_depth=2)
        return max(accuracies)

    # Function:- SVM_Model application on test and train set.
    def svm_model(self):
        clf = svm.SVC()
        clf.fit(self.X_train, self.y_train)
        clf.predict(self.X_test)
        self.clf=clf
        return clf.score(self.X_test, self.y_test)

    # Function:- Choose the best n_estimators parameter to train the Boosted Trees (XGBoost) and plot the accuracies if required.
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
    #print(correlations_array)
    feature_names_array = np.array([item[0] for item in correlations_array])
    return feature_names_array

def reverse(lst):
    new_lst = lst[::-1]
    return new_lst

if __name__ == "__main__":
    print_info = True
    random_forest_model_info = True
    svm_model_info = True
    boosted_trees_info = True
    m= MLModel()
    m.setup('data')
    df = m.setup_dataframe(include_demographics=False, print_stats=False)
    data = m.return_data()
    
    if print_info:
        print(df.columns)
        print(data['user-definedlabeln'].unique())
        print(len(data))
        m.plot_correlation_matrix()
    
    if random_forest_model_info:
        print("Random Forest Model, considering all the features")
        # Random Forest Model, considering all the features
        accuracies = []
        ratio_arr = []
        ratio_i = 0.1
        while ratio_i <= 0.9:
            ratio_arr.append(ratio_i)
            m.split(ratio=ratio_i)
            accuracies.append(m.random_forest_model())
            ratio_i += 0.05
        m.plot_accuracies(accuracies, ratio_arr, 'Random Forest Model: Accuracies vs. Test/Total Ratios')
        print(ratio_arr)
        print(accuracies)
        print(len(m.return_data()))
        print(len(m.X_train))
        print("Maximum Accuracy for Random Forest Model is {}".format(max(accuracies)))

    #print(sorted_correlation(data=data, output_feature='user-definedlabeln'))
    corr_arr = sorted_correlation(data=data, output_feature='user-definedlabeln')
    corr_arr = reverse(corr_arr)

    if svm_model_info:
        # SVM Machine, considering all the features
        accuracies = []
        ratio_arr = []
        ratio_i = 0.1
        while ratio_i < 0.9:
            ratio_arr.append(ratio_i)
            m.split(ratio=ratio_i)
            accuracies.append(m.svm_model())
            ratio_i += 0.1
        m.plot_accuracies(accuracies, ratio_arr, 'SVM Model: Accuracies vs. Test/Total Ratios')
        print(ratio_arr)
        print(accuracies)
        print("Maximum Accuracy for SVM Model is {}".format(max(accuracies)))

    if boosted_trees_info:
        # Boosted Trees, considering all the features
        accuracies = []
        ratio_arr = []
        ratio_i = 0.1
        while ratio_i < 0.9:
            ratio_arr.append(ratio_i)
            m.split(ratio=ratio_i)
            accuracies.append(m.boosted_trees())
            ratio_i += 0.1
        m.plot_accuracies(accuracies, ratio_arr, 'Boosted Trees (XGBoost) Model: Accuracies vs. Test/Total Ratios')
        print(ratio_arr)
        print(accuracies)
        print("Maximum Accuracy for Boosted Model is {}".format(max(accuracies)))

    remove_and_boost_info = True
    remove_and_forest_info = True
    remove_and_svm_info = True
    df_temp = df.copy()
    m.setup_dataframe(include_demographics=False,print_stats=False)

    if remove_and_boost_info:
        accuracies = []
        number_removed = []
        i=0
        while i<len(corr_arr)-1:
            accuracies.append(m.remove_and_boost(corr_arr[i:i+1]))
            number_removed.append(i+1)
            i = i+1

        df=df_temp.copy()
        m.plot_accuracies(accuracies, number_removed, 'Boosted Trees (XGBoost) Model: Accuracies vs. Features Removed')
        print(number_removed)
        print(accuracies)

    if remove_and_svm_info:
        m.setup_dataframe(include_demographics=False,print_stats=False)
        accuracies = []
        removed_tracker = []
        number_removed = []
        i=0
        while i<len(corr_arr)-1:
            removed_tracker.append(corr_arr[i:i+1])
            print("Working with the Subset Removed: {}".format(removed_tracker))
            accuracies.append(m.remove_and_svm(corr_arr[i:i+1]))
            number_removed.append(i+1)
            i = i+1
        df=df_temp.copy()
        m.plot_accuracies(accuracies, number_removed, 'SVM Model: Accuracies vs. Features Removed')
        print(number_removed)
        print(accuracies)

    if remove_and_forest_info:
        m.setup_dataframe(include_demographics=False,print_stats=False)

        accuracies = []
        removed_tracker = []
        number_removed = []
        i=0
        while i<len(corr_arr)-1:
            removed_tracker.append(corr_arr[i:i+1])
            print("Working with the Subset Removed: {}".format(removed_tracker))
            accuracies.append(m.remove_and_forest(corr_arr[i:i+1]))
            number_removed.append(i+1)
            i = i+1
        df=df_temp.copy()
        m.plot_accuracies(accuracies, number_removed, 'Forest Model: Accuracies vs. Features Removed')
        print(number_removed)
        print(accuracies)

def func():
    ratio = 0.1
    y_axis = []
    x_axis = []
    while ratio < 0.9:
        x_axis.append(1-ratio)
        uncertain_sample_num = 500
        m.split(ratio=ratio)
        m.random_forest_model()
        m.setup_dataframe(include_demographics=False,print_stats=False)

        X_train = m.X_train
        y_train = m.y_train

        print("\nTotal in Training Set is {} and Ratio of Train/Total is {}".format(len(X_train),len(X_train)/len(data)))

        print("\nBefore Sampling\n")
        clf = m.return_model()
        clf.fit(X_train, y_train)  
        print("Accuracy:- {}".format(clf.score(m.X_test, m.y_test)*100))

        printing_stats = False
        top_samples, y_samples = m.get_top_n_samples(uncertain_sample_num)
        if printing_stats:
            print(clf)
            print(clf.predict_proba(top_samples))
            print(top_samples)
            print(y_samples)
            print(f"Shape of the top samples: ({len(top_samples)}, {len(top_samples[0])})")
            print(f"Shape of the inital X_train: ({len(X_train)}, {len(X_train[0])})")
            print(type(X_train))
            print(X_train)
            print(type(top_samples))

        print("\nX Size before appending is {}".format(len(X_train)))
        print("Y Size before appending is {}".format(len(y_train)))
        X_train = np.concatenate((X_train, top_samples), axis=0)
        y_train = np.concatenate((y_train,y_samples), axis =0)
        print("X Size after appending is {}".format(len(X_train)))
        print("Y Size after appending is {}".format(len(y_train)))

        print("\nAfter Uncertainty Sampling:-")
        clf = None
        clf = m.return_model()
        clf.fit(X_train, y_train)  # Train your model
        print("Accuracy for training data size {}:- {}".format(len(X_train),clf.score(m.X_test, m.y_test)*100))
        uncertain_prob = clf.score(m.X_test, m.y_test)*100

        new_split_ratio = len(X_train)/len(data)
        print("\nNew Split Ratio is {}".format(new_split_ratio))
        m.split(ratio=1-new_split_ratio)
        m.random_forest_model()
        m.setup_dataframe(include_demographics=False,print_stats=False)

        print("\nRandom Sampling with the new ratio:-")
        clf = None
        clf = m.return_model()
        clf.fit(m.X_train, m.y_train)  # Train your model
        print("Accuracy for training data size {}:- {}".format(len(m.X_train),clf.score(m.X_test, m.y_test)*100))
        random_prob = clf.score(m.X_test, m.y_test)*100
        ratio = ratio + 0.1
        y_axis.append(uncertain_prob - random_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, y_axis, marker='o', linestyle='-', color='b')
    plt.title("Difference in Sampling Approach vs Train/Test Ratio")
    plt.xlabel('Train/Test Ratio')
    plt.ylabel('Difference between Uncertain Sampling vs Random Sampling')
    plt.grid(True)
    plt.show()

    print(y_axis)
    print(x_axis)