from EDF import *   
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


CLASS_SIZE = 100
LEARNING_RATE = 0.01
EPOCHS = 100
TEST_SIZE = 0.25


np.random.seed(0)


class0_a = multivariate_normal.rvs([0, 0], [[0.2, 0], [0, 0.2]], CLASS_SIZE // 2)
class0_b = multivariate_normal.rvs([2, 2], [[0.2, 0], [0, 0.2]], CLASS_SIZE // 2)
class0 = np.vstack((class0_a, class0_b))
y0 = np.zeros(CLASS_SIZE)

# Class 1 
class1_a = multivariate_normal.rvs([0, 2], [[0.2, 0], [0, 0.2]], CLASS_SIZE // 2)
class1_b = multivariate_normal.rvs([2, 0], [[0.2, 0], [0, 0.2]], CLASS_SIZE // 2)
class1 = np.vstack((class1_a, class1_b))
y1 = np.ones(CLASS_SIZE)

# Combine and shuffle
X = np.vstack((class0, class1))
y = np.hstack((y0, y1))

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]


test_size = int(len(X) * TEST_SIZE)
X_train, X_test = X[test_size:], X[:test_size]
y_train, y_test = y[test_size:], y[:test_size]


x1_node = Input()
x2_node = Input()
y_node = Input()

W0 = np.zeros(1)
W1 = np.random.randn(1) * 0.1
W2 = np.random.randn(1) * 0.1

w0_node = Parameter(W0)
w1_node = Parameter(W1)
w2_node = Parameter(W2)


u1_node = Multiply(x1_node, w1_node)
u2_node = Multiply(x2_node, w2_node)
u12_node = Addition(u1_node, u2_node)
u_node = Addition(u12_node, w0_node)
sigmoid = Sigmoid(u_node)
loss = BCE(y_node, sigmoid)

graph = [x1_node, x2_node, w0_node, w1_node, w2_node, u1_node, u2_node, u12_node, u_node, sigmoid, loss]
trainable = [w0_node, w1_node, w2_node]

def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t][0]


for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(X_train.shape[0]):
        x1_node.value = X_train[i][0].reshape(1, -1)
        x2_node.value = X_train[i][1].reshape(1, -1)
        y_node.value = y_train[i].reshape(1, -1)

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)

        loss_value += loss.value

    print(f"Epoch {epoch+1}, Loss: {loss_value / X_train.shape[0]:.4f}")


correct_predictions = 0
for i in range(X_test.shape[0]):
    x1_node.value = X_test[i][0].reshape(1, -1)
    x2_node.value = X_test[i][1].reshape(1, -1)
    forward_pass(graph)

    prediction = 1 if sigmoid.value >= 0.5 else 0
    if prediction == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"\nAccuracy: {accuracy * 100:.2f}%")


x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    x1_node.value = np.array([i]).reshape(1, -1)
    x2_node.value = np.array([j]).reshape(1, -1)
    forward_pass(graph)
    Z.append(sigmoid.value)
Z = np.array(Z).reshape(xx.shape)

#plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title("XOR Dataset and Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
