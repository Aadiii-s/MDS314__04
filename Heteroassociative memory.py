import numpy as np

def to_bipolar(arr):
    return np.where(np.array(arr) <= 0, -1, 1)

def train_hebbian(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    n_patterns = X.shape[0]
    n_inputs  = X.shape[1]
    n_outputs = Y.shape[1]

    W = np.zeros((n_outputs, n_inputs))

    for p in range(n_patterns):
        x = X[p].reshape(n_inputs, 1)
        y = Y[p].reshape(n_outputs, 1)
        W += y @ x.T

    return W / n_patterns

def train_pseudoinverse(X, Y):
    X = np.array(X).T 
    Y = np.array(Y).T   
    X_pinv = np.linalg.pinv(X)
    return Y @ X_pinv  

def recall(W, x):
    x = np.array(x).reshape(-1)
    out = W @ x
    return np.where(out >= 0, 1, -1)


P = 4
n_inputs  = 6
n_outputs = 4


X = np.random.randint(0, 2, (P, n_inputs))
Y = np.random.randint(0, 2, (P, n_outputs))


Xb = to_bipolar(X)
Yb = to_bipolar(Y)


W_hebb = train_hebbian(Xb, Yb)
W_pin  = train_pseudoinverse(Xb, Yb)

print("Hebbian Weight Matrix:\n", W_hebb)
print("\nPseudoinverse Weight Matrix:\n", W_pin)

print("\n---- Testing Recall ----")
for i in range(P):
    print(f"\nPattern {i}:")
    print("Input      :", Xb[i])
    print("Target Out :", Yb[i])
    print("Hebbian Out:", recall(W_hebb, Xb[i]))
    print("PseudoInv  :", recall(W_pin,  Xb[i]))
