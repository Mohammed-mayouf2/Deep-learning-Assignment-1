import numpy as np
from EDF import *
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import multivariate_normal
from keras import datasets
from matplotlib import pyplot
 
 # Hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 20
TEST_SIZE = 0.4
BATCH_SIZE=10
WIDTH=250


(train_X, train_y), (test_X, test_y) =  datasets.mnist.load_data()


print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()


n_output = 10
columnsNO=train_X.shape[1]*train_X.shape[2]
train_X,test_X=train_X.reshape((train_X.shape[0],columnsNO)),test_X.reshape((test_X.shape[0],columnsNO))
n_features = train_X.shape[1]

onehot_y,onehot_ytest=np.zeros((train_y.shape[0],n_output)),np.zeros((test_y.shape[0],n_output))
onehot_y[np.arange(onehot_y.shape[0]),train_y]=1
onehot_ytest[np.arange(onehot_ytest.shape[0]),test_y]=1


x_node = Input()
y_node = Input()



# Build computation graph


A1 = Parameter(np.random.randn(WIDTH, n_features) * 0.01)
b1 = Parameter(np.zeros((WIDTH,)))

h1node = Linear(A1, x_node, b1)
activatedH1 = Sigmoid(h1node)

A2 = Parameter(np.random.randn(WIDTH, WIDTH) * 0.01)
b2 = Parameter(np.zeros((WIDTH,)))

h2node = Linear(A2, activatedH1, b2)
activatedH2 = Sigmoid(h2node)

A3 = Parameter(np.random.randn(n_output, WIDTH) * 0.01)
b3 = Parameter(np.zeros((n_output,)))

h0node = Linear(A3, activatedH2, b3)
activatedOutput = Softmax(h0node)
loss = CrossEntropy(y_node, activatedOutput)

# Create graph outside the training loop
graph = []
trainable = []

# Topological Sort
def topologicalSort(node,graph,trainable, visited=None):
    if visited is None:
        visited = set()
    if id(node) in visited:
        return
    visited.add(id(node))
    for n in getattr(node, 'inputs', []):
        topologicalSort(n,graph,trainable, visited)
    graph.append(node)
    if isinstance(node, Parameter):
            trainable.append(node)

# build graph and trainable list
topologicalSort(loss,graph,trainable)
    
# Forward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()
# Backward Pass
def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

# SGD Update
def sgd_update(trainables, LEARNING_RATE):
    for t in trainables:
        t.value -= LEARNING_RATE * t.gradients[t]


# Training 
learning_rate = 0.1
for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(int(train_X.shape[0]/BATCH_SIZE)):
        k=i*BATCH_SIZE

        x_node.value=train_X[k:k+BATCH_SIZE]
        y_node.value = onehot_y[k:k+BATCH_SIZE]
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)

        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss_value / train_X.shape[0]}")

# Evaluate the model
correct_predictions = 0
confusionMatrix=np.zeros((n_output,n_output))

for i in range(test_X.shape[0]):
    x_node.value = test_X[i:i+1]
    y_node.value = test_y[i:i+1]
    forward_pass(graph)
    pred_class = np.argmax(activatedOutput.value, axis=1)[0]
    if  onehot_ytest[i,pred_class]==1:
        correct_predictions += 1
    confusionMatrix[np.where( onehot_ytest[i]==1)[0],pred_class]+=1
    
    


plt.figure(figsize=(8,6))
plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

tick_marks = np.arange(n_output)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
thresh = confusionMatrix.max() / 2.

# Add color bar
plt.tight_layout()
plt.show()
accuracy = correct_predictions / test_X.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusionMatrix)


# Add text annotations
for i in range(n_output):
    for j in range(n_output):
        plt.text(j, i, int(confusionMatrix[i, j]),
                 horizontalalignment="center",
                 color="white" if confusionMatrix[i, j] > thresh else "black")
