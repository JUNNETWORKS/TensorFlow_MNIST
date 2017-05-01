# http://tensorflow.classcat.com/2016/03/09/tensorflow-cc-mnist-for-ml-beginners/
import tensorflow as tf

# MNIST データのダウンロード
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
"""
イメージ : x
ラベル : y

mnist.train 学習データ
mnist.train.images  学習イメージデータ 28*28   形状[55000, 784] 55000枚のデータと28*28の784次元ベクトル
mnist.train.labels  学習ラベル

mnist.test  テストデータ
mnist.test.images   テストイメージデータ
mnist.test.labels   テストラベル
"""

# モデルの作成
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)  # xとWの乗算の後bを加算し softmaxを適用

# 損失関数とオプティマイザー(最適化)を定義
y_ = tf.placeholder(tf.float32, [None, 10])     # 正解を入力したプレースホルダー
cross_entropy = -tf.reduce_sum(y_*tf.log(y))    # 交差エントロピー誤差

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)    #学習率0.01で交差エントロピー誤差を小さくしていく

# 学習
init = tf.global_variables_initializer()   # 学習を開始する前に変数の初期化を行う    追記: 初期化の関数が変わって "global_variables_initializer" になりました。
sess = tf.InteractiveSession()
sess.run(init)

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)  # ループ毎に100のランダムなデータのバッチを取得
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

# 訓練モデルのテスト
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))    # equalは2つの変数が等しければTrueを返し、異なればFalseを返す
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 平均を求める(正解数をデータ数で割る)
print("Accuracy: " + str(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))) # 実際にテストデータで精度を求める
