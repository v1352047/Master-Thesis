# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append("../../../dataset")  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from vehicles import load_vehicles
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_vehicles(normalize=True, one_hot_label=False)

    
    #データの選別
    x_train = x_train[5000: ]
    t_train = t_train[5000: ]

    return x_train, t_train


def init_network():
    with open("first-learn.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y



x, t = get_data()
network = init_network()
accuracy_cnt = 0
wrong_ans = []


for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1
    else:
        wrong_ans.append(i)


#間違えた問題のindexを保存
with open('wrong-list.pkl', 'wb') as f:
    pickle.dump(wrong_ans, f)


print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
print("counter:" + str(len(x)))
