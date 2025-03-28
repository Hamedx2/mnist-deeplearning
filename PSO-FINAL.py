import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class PSO:
    def __init__(self, c1, c2, w, num_particles, dimensions, epochs, x, y, all_layers_sizes):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.epochs = epochs
        self.x = x
        self.y = y
        self.all_layers_sizes = all_layers_sizes

        # Xavier initialization for weights
        self.particles = np.zeros((num_particles, dimensions))
        ptr = 0
        for i in range(len(all_layers_sizes) - 1):
            n_in = all_layers_sizes[i]
            n_out = all_layers_sizes[i + 1]
            limit = np.sqrt(2 / (n_in + n_out))
            size = n_in * n_out + n_out # weights and biases
            self.particles[:, ptr:ptr+size] = np.random.uniform(-limit, limit, (num_particles, size))
            ptr += size

        self.velocities = np.random.uniform(-0.1, 0.1, (num_particles, dimensions))
        self.pbest = self.particles.copy()
        self.pbest_scores = np.full(num_particles, float('inf'))
        self.gbest = self.pbest[0].copy()
        self.gbest_score = float('inf')
        
    def optimize_function(self):


        for epoch in range(self.epochs):
            
            # Evaluate each particle
            for i in range(self.num_particles):
                all_w, all_b = decode_solution(self.particles[i], self.all_layers_sizes)
                output = feedforward(all_w, all_b, self.x)
                loss = compute_loss(output, self.y)
                
                # Update personal best
                if loss < self.pbest_scores[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_scores[i] = loss
                
                # Update global best
                if loss < self.gbest_score:
                    self.gbest = self.particles[i].copy()
                    self.gbest_score = loss
                    print(f'Epoch {epoch}: Loss = {loss:.4f}')
            
            # Update velocities and positions
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dimensions)
                r2 = np.random.rand(self.dimensions)
                # update velocities
                # m,n = self.velocities[i].size
                self.velocities[i] = (
                    np.random.rand(self.dimensions) * self.velocities[i] +
                    self.c1 * r1 * (self.pbest[i] - self.particles[i]) +
                    self.c2 * r2 * (self.gbest - self.particles[i])
                )
                # mahdodo cardane velocity
                self.velocities[i] = np.clip(self.velocities[i], -1, 1)
                # Update position
                self.particles[i] += self.velocities[i]
            # Early stopping condition
            if self.gbest_score < 0.1:
                break
        
        all_w, all_b = decode_solution(self.gbest, self.all_layers_sizes)
        return all_w, all_b


def compute_loss(predictions, Y):
    if Y.shape[0] != predictions.shape[0]:
        Y = one_hot(Y)
    m = Y.shape[1]
    predictions = np.clip(predictions, 1e-8, 1.0)  # Avoid log(0)
    loss = -np.sum(Y * np.log(predictions)) / m
    return loss

def read_data():
    data = pd.read_csv('./train.csv')
    data = np.array(data)
    _, n = data.shape
    np.random.shuffle(data)
    data_test = data[:1000].T
    Y_test = data_test[0]
    X_test = data_test[1:n] / 255.
    data_train = data[1000:].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.
    return X_test, Y_test, X_train, Y_train

def relu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    shifted_Z = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(shifted_Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def get_predictions(output):
    return np.argmax(output, 0)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def feedforward(all_w, all_b, x):
    all_z = []
    all_a = []
    z_1 = all_w[0].dot(x) + all_b[0]
    a_1 = relu(z_1)
    all_z.append(z_1)
    all_a.append(a_1)
    for i in range(1, len(all_w)):
        z_i = all_w[i].dot(all_a[i-1]) + all_b[i]
        if i != len(all_w)-1:
            a_i = relu(z_i)
        else:
            a_i = softmax(z_i)
        all_z.append(z_i)
        all_a.append(a_i)
    return all_a[-1]

def decode_solution(particle, all_layers_size):
    index = 0
    all_w, all_b = [], []
    for i in range(len(all_layers_size) - 1):
        w_size = all_layers_size[i] * all_layers_size[i+1]
        b_size = all_layers_size[i+1]
        w = particle[index:index+w_size].reshape(all_layers_size[i+1], all_layers_size[i])
        index += w_size
        b = particle[index:index+b_size].reshape(all_layers_size[i+1], 1)
        index += b_size
        all_w.append(w)
        all_b.append(b)
    return all_w, all_b

def make_predictions(X, all_w, all_b):
    output = feedforward(all_w, all_b, X)
    predictions = get_predictions(output)
    return predictions

def test_prediction(index, all_w, all_b, X, Y):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], all_w, all_b)
    label = Y[index]
    print("Prediction:", prediction)
    print("Label:", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def init_param():
    X_test, Y_test, X_train, Y_train = read_data()

    all_layers_sizes = [784, 16, 10]
    dimensions = 0
    for i in range(len(all_layers_sizes) - 1):
        n_in = all_layers_sizes[i]
        n_out = all_layers_sizes[i+1]
        dimensions += n_in * n_out + n_out
    # PSO hyperparameters
    epochs = 100
    c1 = 1.494
    c2 = 1.494
    w = 0.729
    num_particles = 100
    # Optimize network parameters using PSO
    all_w, all_b = PSO(c1, c2, w, num_particles, dimensions, epochs, X_train, Y_train, all_layers_sizes).optimize_function()
    test_predictions_train = make_predictions(X_train, all_w, all_b)
    accuracy_train = get_accuracy(test_predictions_train, Y_train)
    print(f'Train accuracy: {accuracy_train*100:.1f}%')
    test_predictions = make_predictions(X_test, all_w, all_b)
    accuracy = get_accuracy(test_predictions, Y_test)
    print(f'Test accuracy: {accuracy*100:.1f}%')
    while True:
        try:
            test_index = int(input('Enter an index (0-999): '))
            test_prediction(test_index, all_w, all_b, X_test, Y_test)
        except:
            break

init_param()
