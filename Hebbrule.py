import numpy as np

def hebbian_learning(X, Y, learning_rate=1):
   
    
    num_samples, num_features = X.shape
    
    #  Initialize Weights and Bias
    w = np.zeros(num_features)
    b = 0.0
    
    print(f"Initial weights: {w}, Initial bias: {b}\n")
    
    for i in range(num_samples):
        x = X[i]
        y = Y[i]
        
        delta_w = learning_rate * x * y
        delta_b = learning_rate * y
        
        # Update weights and bias
        w += delta_w
        b += delta_b
        
        print(f"Sample {i+1}: new_w: {w}, new_b: {b}")

    print("\n--- Training Complete ---")
    print(f"Final Weights (w): {w}")
    print(f"Final Bias (b): {b}")
    return w, b

def bipolar_step_activation(net_input):

    return 1 if net_input >= 0 else -1

if __name__ == "__main__":

    X_train = np.array([
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]
    ])
    
    # Bipolar targets for OR
    Y_train = np.array([1, 1, 1, -1])
    
    print("--- Starting Hebbian Learning for OR Gate ---")

    w, b = hebbian_learning(X_train, Y_train, learning_rate=1)
    
    print("\n--- Testing the Trained Neuron ---")
    
    for i in range(X_train.shape[0]):
        x_test = X_train[i]
        y_target = Y_train[i]
        
        net_input = np.dot(x_test, w) + b
        
        # predicted output
        y_predicted = bipolar_step_activation(net_input)
        
        print(f"Input: {x_test}, Target: {y_target}, Predicted: {y_predicted}")