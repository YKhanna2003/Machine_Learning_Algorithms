import csv
import tensorflow as tf
from tensorflow import keras
import numpy as np
from multiprocessing import Process, Queue

def train_and_evaluate_model(train_data, train_labels, test_data, test_labels, units_arr, activation_arr, model_idx, results_queue):
    optimize_max_cpu()
    model = general_model(train_data, train_labels, units_arr, activation_arr)
    test_accuracy, test_loss = test_model(test_data, test_labels, model)
    results_queue.put((model_idx, test_accuracy, test_loss))

def optimize_max_cpu():
    # Set the number of threads for parallel execution of operations within a single op (intra-op parallelism).
    tf.config.threading.set_intra_op_parallelism_threads(0)  # 0 means use all available cores.
    # Set the number of threads for parallel execution of independent operations (inter-op parallelism).
    tf.config.threading.set_inter_op_parallelism_threads(0)  # 0 means use all available cores.

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def load_csv_data(file_path):
    # Initialize a 2D array to store the data
    csv_data = []
    labels = []

    # Open the CSV file for reading
    with open(file_path, 'r', newline='') as input_file:
        csv_reader = csv.reader(input_file)

        # Skip the first row (header)
        next(csv_reader)

        # Read and process the remaining rows
        for row in csv_reader:
            labels.append(int(row[0]))  # Assuming the labels are in the first column
            csv_data.append(row[1:])

    int_2d_array = [[int(cell) for cell in row] for row in csv_data]

    return int_2d_array, labels

def general_model(train_data,train_labels,units_arr,activation_arr):

    if len(units_arr) != len(activation_arr):
        print("Incorrect Parameters")
        return
    
    # Create and compile the model
    layers_arr = []
    i=0
    for unit in units_arr:
        layers_arr.append(tf.keras.layers.Dense(units=unit,activation=activation_arr[i]))
        i=i+1

    model = keras.Sequential(layers_arr)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Train the model
    model.fit(train_data, train_labels, epochs=100)
    return model

def test_model(test_data,test_labels,model):
    test_loss = model.evaluate(test_data, test_labels)
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_labels == test_labels)
    test_accuracy = correct_predictions / len(test_labels)
    return test_accuracy,test_loss

def main():
    optimize_max_cpu()

    # Load the training data
    train_data, train_labels = load_csv_data('./Dataset/archive/mnist_train.csv')
    # Load the testing data
    test_data, test_labels = load_csv_data('./Dataset/archive/mnist_test.csv')
    # Evaluate the model on the test data
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    # Multiple Layers, same number but greater number of neurons.
    # Time taken by greater neurons to train is higher as compared to the optimized one.
    models = [
        ([25, 15, 10], ['relu', 'relu', 'linear'], "ReLu, ReLu, Linear layers"),
        ([250, 150, 10], ['relu', 'relu', 'linear'], "ReLu, ReLu, Linear layers"),
    ]

    # Output
    # Model 2 - ReLu, ReLu, Linear layers:
    # Test Accuracy: 0.9642
    # Test Loss: 1.011023759841919

    # Model 1 - ReLu, ReLu, Linear layers:
    # Test Accuracy: 0.9352
    # Test Loss: 0.290063738822937

    # Multiple Layer Case, the accuracy seems to be flatlining
    # models = [
    #     ([10], ['linear'], "Linear layer"),
    #     ([15, 10], ['relu', 'linear'], "ReLu, Linear layers"),
    #     ([25, 15, 10], ['relu', 'relu', 'linear'], "ReLu, ReLu, Linear layers"),
    #     ([35, 25, 15, 10], ['relu', 'relu', 'relu', 'linear'], "Relu, ReLu, ReLu, Linear layers"),
    #     ([55, 45, 35, 25, 15, 10], ['relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, ReLu, ReLu, ReLu, ReLu, Linear layers"),
    #     ([65, 55, 45, 35, 25, 15, 10], ['relu','relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers"),
    #     ([75, 65, 55, 45, 35, 25, 15, 10], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers"),
    #     ([85, 75, 65, 55, 45, 35, 25, 15, 10], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, ReLu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers"),
    #     ([95, 85, 75, 65, 55, 45, 35, 25, 15, 10], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'], "Relu, Relu, ReLu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers")
    # ]

    # Output
    # Model 9 - Relu, Relu, ReLu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:
    # Test Accuracy: 0.9788
    # Test Loss: 0.14306193590164185

    # Model 8 - Relu, ReLu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:
    # Test Accuracy: 0.9747
    # Test Loss: 0.16111640632152557

    # Model 7 - Relu, ReLu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:
    # Test Accuracy: 0.9771
    # Test Loss: 0.21444159746170044

    # Model 6 - Relu, Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:
    # Test Accuracy: 0.9729
    # Test Loss: 0.30189087986946106

    # Model 5 - Relu, ReLu, ReLu, ReLu, ReLu, Linear layers:
    # Test Accuracy: 0.9755
    # Test Loss: 0.3504304885864258

    # Model 4 - Relu, ReLu, ReLu, Linear layers:
    # Test Accuracy: 0.9603
    # Test Loss: 0.4428563117980957

    # Model 3 - ReLu, ReLu, Linear layers:
    # Test Accuracy: 0.9437
    # Test Loss: 0.4288692772388458

    # Model 2 - ReLu, Linear layers:
    # Test Accuracy: 0.9011
    # Test Loss: 0.4153865575790405

    # Model 1 - Linear layer:
    # Test Accuracy: 0.8913
    # Test Loss: 7.499961853027344

    results_queue = Queue()
    processes = []

    for i, (units_arr, activation_arr, model_name) in enumerate(models):
        process = Process(target=train_and_evaluate_model, args=(train_data, train_labels, test_data, test_labels, units_arr, activation_arr, i, results_queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Sort and print results
    results.sort(key=lambda x: x[1], reverse=True)
    for model_idx, test_accuracy, test_loss in results:
        print(f"Model {model_idx + 1} - {models[model_idx][2]}:")
        print("Test Accuracy:", test_accuracy)
        print("Test Loss:", test_loss)
        print("\n")

if __name__ == "__main__":
    main()
