from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



N_SAMPLES_PER_QUADRANT = 100  
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.01
EPOCHS = 999
TEST_SIZE = 0.25
BATCH_SIZE = 8  


MEAN_C0_1 = np.array([2, 2])
MEAN_C0_2 = np.array([-2, -2])
MEAN_C1_1 = np.array([-2, 2])
MEAN_C1_2 = np.array([2, -2])
COV = np.array([[0.5, 0], [0, 0.5]])  



# نقاط الفئة 0
X_c0_1 = multivariate_normal.rvs(MEAN_C0_1, COV, N_SAMPLES_PER_QUADRANT)
X_c0_2 = multivariate_normal.rvs(MEAN_C0_2, COV, N_SAMPLES_PER_QUADRANT)


# نقاط الفئة 1
X_c1_1 = multivariate_normal.rvs(MEAN_C1_1, COV, N_SAMPLES_PER_QUADRANT)
X_c1_2 = multivariate_normal.rvs(MEAN_C1_2, COV, N_SAMPLES_PER_QUADRANT)


# دمج النقاط وتوليد العناوين (Labels)
# X shape: (400, 2), y shape: (400,)
X = np.vstack((X_c0_1, X_c0_2, X_c1_1, X_c1_2))
y = np.hstack((np.zeros(2 * N_SAMPLES_PER_QUADRANT), np.ones(2 * N_SAMPLES_PER_QUADRANT)))



# رسم البيانات التي تم إنشاؤها
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated XOR Data')
plt.show()

# تقسيم البيانات
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# متغيرات النموذج
n_features = X_train.shape[1]
n_output = 1

# تهيئة الأوزان والانحياز
a = np.random.randn(n_output, n_features) * 0.1
b = np.random.randn(1) * 0.1

# إنشاء العقد
x_node = Input()
y_node = Input()
A_node = Parameter(a)
b_node = Parameter(b)

# بناء الرسم البياني للحوسبة
z_node = Linear(A_node, x_node, b_node)
sigmoid = Sigmoid(z_node)
loss = BCE(y_node, sigmoid)

# إنشاء الرسم البياني خارج حلقة التدريب
graph = [x_node, A_node, b_node, z_node, y_node, sigmoid, loss]
trainable = [A_node, b_node]

# دوال المرور الأمامي والخلفي
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

# دالة تحديث SGD
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t][0]

# حلقة التدريب
for epoch in range(EPOCHS):
    loss_value = 0
    num_batches = X_train.shape[0] // BATCH_SIZE
    for i in range(num_batches):
        k = i * BATCH_SIZE
        x_batch = X_train[k:k + BATCH_SIZE]
        y_batch = y_train[k:k + BATCH_SIZE].reshape(-1, 1)

        if x_batch.shape[0] == 0:
            continue

        x_node.value = x_batch
        y_node.value = y_batch

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)

        loss_value += loss.value
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]:.4f}")

# تقييم النموذج
correct_predictions = 0
for i in range(X_test.shape[0]):
    x_node.value = X_test[i].reshape(1, -1) # التأكد من أن المدخلات ثنائية الأبعاد
    forward_pass(graph)

    prediction = 1 if sigmoid.value > 0.5 else 0
    if prediction == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

# رسم حدود القرار
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z_list = []
for point in np.c_[xx.ravel(), yy.ravel()]:
    x_node.value = point.reshape(1, -1)
    forward_pass(graph)
    Z_list.append(sigmoid.value[0,0])

Z = np.array(Z_list).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Logistic Regression Decision Boundary (Accuracy: {accuracy*100:.2f}%)')
plt.show()