# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append("../../../dataset")  # 親ディレクトリのファイルをインポートするための設定
import cupy as cp
import pickle
from common import config
from common.util import to_cpu, to_gpu
from fruits import load_fruits
from deep_convnet import DeepConvNet
from common.trainer import Trainer


#GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
#===============================================
config.GPU = True
#===============================================


(x_train, t_train), (x_test, t_test) = load_fruits(flatten=False)

x_train = to_gpu(x_train)
t_train = to_gpu(t_train)

#データの選別
x_train = x_train[5000: ]
t_train = t_train[5000: ]



network = DeepConvNet()  
network.load_params(file_name="first-learn.pkl")
accuracy = 0
wrong_ans = []


y = network.predict_for_select(x_train, num_limit=100)
p = np.argmax(y, 1)
wrong_ans = [key for key, value in enumerate(p == t_train) if not value]
accuracy = sum(p == t_train) / len(x_train)


#間違えた問題のindexを保存
with open('wrong-list.pkl', 'wb') as f:
    pickle.dump(wrong_ans, f)
