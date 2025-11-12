import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from EDF import *
from scipy.stats import multivariate_normal


EPOCHS = 50
LEARNING_RATE = 0.1
TEST_RATIO = 0.4
BATCH_SIZE = 16
HIDDEN_UNITS = 64

# -----------------------------
# تحميل بيانات MNIST البسيطة (Digits)
# -----------------------------
digits = datasets.load_digits()
X = digits.data
y = digits.target.astype(int)

# تحويل التصنيفات إلى One-hot encoding
y_encoded = np.zeros((len(y), 10))
y_encoded[np.arange(len(y)), y] = 1

# -----------------------------
# تقسيم البيانات إلى تدريب واختبار
# -----------------------------
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

split_point = int(len(X) * (1 - TEST_RATIO))
train_idx, test_idx = indices[:split_point], indices[split_point:]

X_train, y_train = X[train_idx], y_encoded[train_idx]
X_test, y_test = X[test_idx], y_encoded[test_idx]

# -----------------------------
# إنشاء العقد (Nodes)
# -----------------------------
x_node = Input()
y_node = Input()

# الشبكة العصبية: طبقة خفية + طبقة خرج
# determine feature/output sizes
n_features = X_train.shape[1]
n_output = 10

# hidden layer parameters (A: [out, in], b: [out])
A1 = Parameter(np.random.randn(HIDDEN_UNITS, n_features))
b1 = Parameter(np.zeros(HIDDEN_UNITS))
hidden_layer = Linear(A1, x_node, b1)
activated_hidden = Sigmoid(hidden_layer)

# output layer parameters
A2 = Parameter(np.random.randn(n_output, HIDDEN_UNITS))
b2 = Parameter(np.zeros(n_output))
output_layer = Linear(A2, activated_hidden, b2)
activated_output = Softmax(output_layer)

# دالة الخسارة
loss_node = CrossEntropy(y_node, activated_output)

# -----------------------------
# بناء الرسم البياني (Graph)
# -----------------------------
graph, trainable = [], []

def build_graph(node):
    """Recursive function to build computation graph"""
    if node not in graph:
        for input_node in node.inputs:
            build_graph(input_node)
        graph.append(node)
        if isinstance(node, Parameter):
            trainable.append(node)

build_graph(loss_node)

# -----------------------------
# تعريف دوال التدريب
# -----------------------------
def forward(graph):
    for n in graph:
        n.forward()

def backward(graph):
    for n in reversed(graph):
        n.backward()

def update_params(trainable, lr):
    for param in trainable:
        param.value -= lr * param.gradients[param]

# -----------------------------
# التدريب
# -----------------------------
for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = len(X_train) // BATCH_SIZE

    for b in range(num_batches):
        start = b * BATCH_SIZE
        end = start + BATCH_SIZE

        x_node.value = X_train[start:end]
        y_node.value = y_train[start:end]

        forward(graph)
        backward(graph)
        update_params(trainable, LEARNING_RATE)

        total_loss += loss_node.value

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss = {avg_loss:.4f}")

# -----------------------------
# التقييم
# -----------------------------
n_classes = 10
conf_matrix = np.zeros((n_classes, n_classes))
y_true = np.argmax(y_test, axis=1)
correct = 0

for i in range(X_test.shape[0]):
    x_node.value = X_test[i:i+1]
    forward(graph)
    pred_label = np.argmax(activated_output.value)
    true_label = y_true[i]

    conf_matrix[true_label, pred_label] += 1
    if pred_label == true_label:
        correct += 1

accuracy = correct / len(X_test)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# رسم مصفوفة الارتباك (Confusion Matrix)
# -----------------------------
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)

thresh = conf_matrix.max() / 2.0
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, int(conf_matrix[i, j]),
                 ha="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
