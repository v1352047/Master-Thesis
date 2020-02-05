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

#GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
#===============================================
config.GPU = True
#===============================================


def get_data():
    (x_train, t_train), (x_test, t_test) = load_fruits(flatten=False, one_hot_label=False)
    x_test = to_gpu(x_test)
    t_test = to_gpu(t_test)
    return x_test, t_test


x, t = get_data()
network = DeepConvNet()
network.load_params(file_name="first-learn-mini.pkl")
accuracy = network.accuracy(x, t, batch_size=50)

#accuracy_cnt = 0


#y = network.predict(x[:100])
#p = np.argmax(y) # 最も確率の高い要素のインデックスを取得
#accuracy_cnt = sum(p == t)


print("Accuracy:", accuracy)
print("counter:", len(x))
#print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
#print("counter:" + str(len(x)))
