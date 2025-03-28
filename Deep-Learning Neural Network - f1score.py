import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def read_data():
    # uses mnist data set
    data = pd.read_csv('./train.csv')
    data = np.array(data)
    _,n = data.shape
    np.random.shuffle(data)
    dataset = data.T
    Y_data = dataset[0]
    X_data = dataset[1:] / 255.
    
    return X_data, Y_data

def cross_over(X_data,Y_data):
    fold_x_1 = X_data[:,:10500]
    fold_y_1 = Y_data[:10500]
    
    fold_x_2 = X_data[:,10500:21000]
    fold_y_2 = Y_data[10500:21000]
    
    fold_x_3 = X_data[:,21000:31500]
    fold_y_3 = Y_data[21000:31500]
    
    fold_x_4 = X_data[:,31500:]
    fold_y_4 = Y_data[31500:]
    
    X_folds = [fold_x_1, fold_x_2, fold_x_3, fold_x_4]
    Y_folds = [fold_y_1, fold_y_2, fold_y_3, fold_y_4]
    
    return X_folds, Y_folds
    
    
def confusion_matrix(Y_test, predictions):
    matrix = np.zeros((10,10))
    for i, j in zip(Y_test, predictions):
        matrix[int(i)][int(j)] += 1
        
    return matrix

def f1_score(matrix):
    scores = []
    for i in range(len(matrix)):
        Ti = matrix[i][i]
        Fj = np.sum(matrix[i]) - matrix[i][i]
        Fi = np.sum(matrix[:][i]) - matrix[i][i]
        
        f1_i = Ti / (Ti + (1/2 * (Fj + Fi)))
        scores.append(f1_i)
        
    final_score = 1 / len(matrix) * np.sum(scores)
    return final_score
        
        


def create_values(input_layer_nodes, hidden_layer_nodes, output_layer_nodes):
    all_w = []
    all_b = []
    # first layer w and b
    w_1 = np.random.randn(hidden_layer_nodes[0], input_layer_nodes) * np.sqrt(2 / input_layer_nodes)
    b_1 = np.zeros((hidden_layer_nodes[0], 1))
    all_w.append(w_1)
    all_b.append(b_1)
    # hidden w and b generation
    for i in range(1, len(hidden_layer_nodes)):
        w_i = np.random.randn(hidden_layer_nodes[i], hidden_layer_nodes[i-1]) * np.sqrt(2 / hidden_layer_nodes[i-1])
        b_i = np.zeros((hidden_layer_nodes[i], 1))
        all_w.append(w_i)
        all_b.append(b_i)
    
    # last layer w and b
    w_Last = np.random.randn(output_layer_nodes, hidden_layer_nodes[-1]) * np.sqrt(2 / hidden_layer_nodes[-1])
    b_Last = np.zeros((output_layer_nodes, 1))
    all_w.append(w_Last)
    all_b.append(b_Last)
    
    return all_w, all_b

def relu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    # Subtract the maximum value for each column for numerical stability.
    shifted_Z = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(shifted_Z)
    # Sum over rows (axis=0) for each column
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A

def deriv_relu(Z):
    return Z > 0
    
def feedforward(all_w, all_b, x):
    all_z = []
    all_a = []
    # first layer calculations  
    z_1 = all_w[0].dot(x) + all_b[0]
    a_1 = relu(z_1)
    all_z.append(z_1)
    all_a.append(a_1)
    
    # other layers calculations
    for i in range(1,len(all_w)):
        z_i = all_w[i].dot(all_a[i-1]) + all_b[i]
        if i != len(all_w)-1:
            a_i = relu(z_i)
        if i == len(all_w)-1:
            a_i = softmax(z_i)
        all_z.append(z_i)
        all_a.append(a_i)
        
    return all_z , all_a


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(all_z, all_a, all_w, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    all_dz = []
    all_dw = []
    all_db = []
    # last layer deltas
    dz_last = all_a[-1] - one_hot_Y
    dw_last = 1/m * dz_last.dot(all_a[-2].T)
    db_last = 1/m * np.array([np.sum(dz_last,1)]).T
    all_dz.insert(0,dz_last)
    all_dw.insert(0,dw_last)
    all_db.insert(0,db_last)
    
    # hidden layers deltas
    for i in range(len(all_a)-2,0,-1):
        dz_i = all_w[i+1].T.dot(all_dz[0]) * deriv_relu(all_z[i])
        dw_i = 1/m * dz_i.dot(all_a[i-1].T)
        db_i = 1/m * np.array([np.sum(dz_i,1)]).T
        all_dz.insert(0,dz_i)
        all_dw.insert(0,dw_i)
        all_db.insert(0,db_i)
    
    # first layer deltas
    dz_first = all_w[1].T.dot(all_dz[0]) * deriv_relu(all_z[0])
    dw_first = 1/m * dz_first.dot(X.T)
    db_first = 1/m * np.array([np.sum(dz_first,1)]).T
    all_dz.insert(0,dz_first)
    all_dw.insert(0,dw_first)
    all_db.insert(0,db_first)
    
    return all_dw, all_db

def update_params_Adam(all_w, all_b, all_dw, all_db, alpha,  beta1, beta2, epsilon, M_t_w, M_t_b, V_t_w, V_t_b, epoch):
    new_all_w = []
    new_all_b = []
    new_M_t_w = []
    new_M_t_b = []
    new_V_t_w = []
    new_V_t_b = []
    for i in range(len(all_w)):
        M_t_w_i = beta1 * M_t_w[i] + (1 - beta1) * all_dw[i]
        M_t_b_i = beta1 * M_t_b[i] + (1 - beta1) * all_db[i]
        
        V_t_w_i = beta2 * V_t_w[i] + (1 - beta2) * (all_dw[i]**2)
        V_t_b_i = beta2 * V_t_b[i] + (1 - beta2) * (all_db[i]**2)
        
        M_hat_w = M_t_w_i /( 1 - (beta1**epoch))
        M_hat_b = M_t_b_i /( 1 - (beta1**epoch))
        V_hat_w = V_t_w_i /( 1 - (beta2**epoch))
        V_hat_b = V_t_b_i /( 1 - (beta2**epoch))
        
        
        new_w_i = all_w[i] - (alpha * (M_hat_w / (np.sqrt(V_hat_w) + epsilon)))
        new_b_i = all_b[i] - (alpha * (M_hat_b / (np.sqrt(V_hat_b) + epsilon)))

        
        new_all_w.append(new_w_i)
        new_all_b.append(new_b_i)
        new_M_t_w.append(M_t_w_i)
        new_M_t_b.append(M_t_b_i)
        new_V_t_w.append(V_t_w_i)
        new_V_t_b.append(V_t_b_i)
    
    return new_all_w , new_all_b, new_M_t_w, new_M_t_b, new_V_t_w, new_V_t_b

def update_params(all_w, all_b, all_dw, all_db, alpha):
    new_all_w = []
    new_all_b = []
    for i in range(len(all_w)):
        new_w_i = all_w[i] - alpha * all_dw[i] #w_new = w_old - alpha * delta_w
        new_b_i = all_b[i] - alpha * all_db[i]
        new_all_w.append(new_w_i)
        new_all_b.append(new_b_i)
    
    return new_all_w , new_all_b

def get_predictions(output):
    return np.argmax(output, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent_adam(X, Y, alpha, beta1, beta2, epsilon, epochs,input_layer_nodes, hidden_layer_nodes, output_layer_nodes):
    
    print('__________You are using Adam as your optimizer__________')
    
    
    all_w , all_b = create_values(input_layer_nodes, hidden_layer_nodes, output_layer_nodes)
    # Adam values
    M_t_w = [np.zeros(i.shape) for i in all_w]
    M_t_b = [np.zeros(i.shape) for i in all_b]
    V_t_w = [np.zeros(i.shape) for i in all_w]
    V_t_b = [np.zeros(i.shape) for i in all_b]

    i = 1
    while (i <= epochs):
        all_z , all_a = feedforward(all_w, all_b, X)
        all_dw , all_db = backward_prop(all_z, all_a, all_w, X, Y)
        
        # using Adam optimizer
        new_all_w , new_all_b, new_M_t_w, new_M_t_b, new_V_t_w, new_V_t_b = update_params_Adam(all_w, all_b, all_dw, all_db, alpha, beta1, beta2, epsilon, M_t_w, M_t_b, V_t_w, V_t_b, i)
        all_w, all_b, M_t_w, M_t_b, V_t_w, V_t_b = new_all_w , new_all_b, new_M_t_w, new_M_t_b, new_V_t_w, new_V_t_b

        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(all_a[-1])
            print(get_accuracy(predictions, Y))
        i += 1
    return all_w, all_b

def gradient_descent_SGD(X, Y, alpha, epochs,input_layer_nodes, hidden_layer_nodes, output_layer_nodes):
    
    print('__________You are using SGD as your optimizer__________')
    
    all_w , all_b = create_values(input_layer_nodes, hidden_layer_nodes, output_layer_nodes)
    
    i = 1
    while (i <= epochs):
        all_z , all_a = feedforward(all_w, all_b, X)
        all_dw , all_db = backward_prop(all_z, all_a, all_w, X, Y)
        
        # using SGD
        new_all_w, new_all_b = update_params(all_w, all_b, all_dw, all_db, alpha)
        all_w, all_b = new_all_w, new_all_b
        
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(all_a[-1])
            print(get_accuracy(predictions, Y))
        i += 1
    return all_w, all_b


def run_network():
    X_data, Y_data = read_data()
    X_folds, Y_folds = cross_over(X_data,Y_data)
    input_layer_nodes = 784
    hidden_layer_nodes = [256,128]
    output_layer_nodes = 10
    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.99
    epsilon = 10 ** -8
    epochs = 10
    optimizer = 'adam'
    best_all_w, best_all_b = [], []
    all_scores = []
    best_score = 0
    for i in range(len(X_folds)):
        X_test = X_folds[i]
        Y_test = Y_folds[i]
        
        remaining_x_folds = X_folds.copy()
        remaining_y_folds = Y_folds.copy()
        
        remaining_x_folds.pop(i)
        remaining_y_folds.pop(i)
        
        X_train = np.concatenate(remaining_x_folds,axis=1)
        Y_train = np.concatenate(remaining_y_folds)
        
        if optimizer == 'adam':
            all_w, all_b = gradient_descent_adam(X_train, Y_train, alpha,  beta1, beta2, epsilon, epochs, input_layer_nodes, hidden_layer_nodes, output_layer_nodes)
            predictions = make_predictions(X_test, all_w, all_b)
            matrix = confusion_matrix(Y_test,predictions)
            f1 = f1_score(matrix)
            if f1 > best_score : 
                best_score = f1
                best_all_w = all_w
                best_all_b = all_b
            all_scores.append(f1)
            
        if optimizer == 'sgd':
            all_w, all_b = gradient_descent_SGD(X_data, Y_data, alpha, epochs, input_layer_nodes, hidden_layer_nodes, output_layer_nodes)
            predictions = make_predictions(X_test, all_w, all_b)
            matrix = confusion_matrix(Y_test,predictions)
            f1 = f1_score(matrix)
            if f1 > best_score :
                best_score = f1
                best_all_w = all_w
                best_all_b = all_b
            all_scores.append(f1)

    final_score = np.sum(all_scores) / len(all_scores)

    return best_all_w, best_all_b, final_score, X_data, Y_data, best_score


def make_predictions(X, all_w, all_b):
    all_z , all_a = feedforward(all_w, all_b, X)
    predictions = get_predictions(all_a[-1])
    return predictions

def test_prediction(index, all_w, all_b, X,Y):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], all_w, all_b)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def manual_test():
    all_w, all_b, final_score, X_data, Y_data, best_score= run_network()
    print()
    print('------------------------------------------------------\n')
    print('final resaults Are :')
    print()
    print(f'Avrage F1 Score for the network is : {final_score}')
    print(f'Best F1 Score is : {best_score}')
    while True:
        test_index = int(input('Enter an index number: '))
        test_prediction(test_index, all_w, all_b, X_data, Y_data)


manual_test()
