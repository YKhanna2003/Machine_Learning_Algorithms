# Machine Learning Algorithms

## Code 1 - Linear Regression and Predicting Canada PCI

The first part of the code is focused on linear regression and predicting Canada's Per Capita Income (PCI) based on the year. It uses the scikit-learn library for linear regression. The code reads a CSV file named "canada_pci.csv" and predicts PCI for the year 2021. The predicted value is printed to the console.

## Code 2 - Multilayer Perceptron (MLP) for MNIST Classification

This repository contains Python code for training and evaluating a Multilayer Perceptron (MLP) model on the MNIST dataset. The code is divided into two parts: one for loading and preprocessing the data, and another for defining and training the MLP model.  
The second part of the code is dedicated to building and training a Multilayer Perceptron (MLP) model for image classification on the MNIST dataset.  

The code is structured as follows:

- It optimizes CPU and GPU settings to ensure efficient execution.  
- It loads and preprocesses the MNIST training and testing data from CSV files.  
- It defines a general MLP model with configurable parameters, such as the number of layers, units, and activation functions.  
- The model is trained on the MNIST training data.  
- It evaluates the model's performance on the MNIST test data, reporting accuracy and loss.  

The code includes the evaluation of multiple MLP models with different configurations, such as the number of layers, units, and activation functions. The results for each model are printed to the console.  

## Instructions for Running the Code

- Ensure you have the required Python libraries installed. You can install them using pip if they are not already installed:  

        pip install pandas numpy matplotlib scikit-learn tensorflow

- Download the MNIST dataset and save it as "mnist_train.csv" and "mnist_test.csv" in the "./Dataset/archive/" directory.  

- Run the main script by executing the following command:  
    
        python mnist_main.py

- Review the results printed to the console for the various MLP model configurations.  

## Configuration

You can modify the model configurations, such as the number of layers, units, and activation functions, in the models list within the code. Experiment with different configurations to observe how they affect the model's performance on the MNIST dataset.

## Results

Multiple Layers, same number but greater number of neurons.  
Time taken by greater neurons to train is higher as compared to the optimized one.  

### Model Set 1

    models = [  
        ([25, 15, 10], ['relu', 'relu', 'linear'], "ReLu, ReLu, Linear layers"),  
        ([250, 150, 10], ['relu', 'relu', 'linear'], "ReLu, ReLu, Linear layers"),  
    ]

### Output

#### Model 2 - ReLu, ReLu, Linear layers:  
    # Test Accuracy: 0.9642  
    # Test Loss: 1.011023759841919  

#### Model 1 - ReLu, ReLu, Linear layers:  
    # Test Accuracy: 0.9352  
    # Test Loss: 0.290063738822937  

### Model Set 2  

    models = [  
        ([10], ['linear'], "Linear layer"),  
        ([15, 10], ['relu', 'linear'], "ReLu, Linear layers"),  
        ([25, 15, 10], ['relu', 'relu', 'linear'], "ReLu, ReLu, Linear layers"),  
        ([35, 25, 15, 10], ['relu', 'relu', 'relu', 'linear'], "Relu, ReLu, ReLu, Linear layers"),  
        ([55, 45, 35, 25, 15, 10], ['relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, ReLu, ReLu, ReLu, ReLu, Linear layers"),  
        ([65, 55, 45, 35, 25, 15, 10], ['relu','relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers"),  
        ([75, 65, 55, 45, 35, 25, 15, 10], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers"),  
        ([85, 75, 65, 55, 45, 35, 25, 15, 10], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, ReLu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers"),  
        ([95, 85, 75, 65, 55, 45, 35, 25, 15, 10], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, Relu, ReLu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers")  
    ]

### Output

#### Model 9 - Relu, Relu, ReLu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:  
    # Test Accuracy: 0.9788  
    # Test Loss: 0.14306193590164185  

#### Model 8 - Relu, ReLu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:  
    # Test Accuracy: 0.9747  
    # Test Loss: 0.16111640632152557  

#### Model 7 - Relu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:  
    # Test Accuracy: 0.9771  
    # Test Loss: 0.21444159746170044  

#### Model 6 - Relu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:  
    # Test Accuracy: 0.9729  
    # Test Loss: 0.30189087986946106  

#### Model 5 - Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:  
    # Test Accuracy: 0.9755  
    # Test Loss: 0.3504304885864258  

#### Model 4 - Relu, ReLu, ReLu, Linear layers:  
    # Test Accuracy: 0.9603  
    # Test Loss: 0.4428563117980957  

#### Model 3 - ReLu, ReLu, Linear layers:  
    # Test Accuracy: 0.9437  
    # Test Loss: 0.4288692772388458  

#### Model 2 - ReLu, Linear layers:  
    # Test Accuracy: 0.9011  
    # Test Loss: 0.4153865575790405  

#### Model 1 - Linear layer:  
    # Test Accuracy: 0.8913  
    # Test Loss: 7.499961853027344  

Note: Ensure you have a reasonable amount of computational resources available, especially if you choose complex model configurations with a large number of layers and units.
