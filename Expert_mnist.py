# http://qiita.com/KojiOhki/items/64a2ee54214b01a411c7
# http://kivantium.hateblo.jp/entry/2015/11/18/233834
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# MNISTデータのダウンロード
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess = tf.InteractiveSession()

# モデルの作成
x = tf.placeholder(tf.float32, shape=[None, 784])   # 入力ノード
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 損失関数とオプティマイザー(最適化)を定義
y_ = tf.placeholder(tf.float32, shape=[None, 10])   # 出力ノード
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

sess.run(tf.global_variables_initializer())

# 学習
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))    # equalは2つの変数が等しければTrueを返し、異なればFalseを返す
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 平均を求める(正解数をデータ数で割る)
print("Accuracy: " + str(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))) # 実際にテストデータで精度を求める
# 精度は約91% であり、このコードはビギナー編と同じものである。



"""
    深層畳み込みネットワーク
    精度 99% を目指す
※追加知識
Variable は"変数"という意味
"""

# 重みを標準偏差0.1の正規分布で初期化
def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)
# バイアスを標準偏差0.1の正規分布で初期化
def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)
# 畳み込み層
def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
# プーリング層
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# 第1畳み込み層
# 第1レイヤー 5x5パッチで32の特徴を計算
# [5,5,1,32]は、5,5でパッチサイズを、1で入力チャンネル数、32で出力チャンネル
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 上記に対応するためにxを4次元にする
# 画像をモノクロにする(第2,第3の次元は画像の幅と高さ,)
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1) + b_conv1)   # x_imageと重みを畳み込みしバイアスを加えRelu関数を適用
h_pool1 = max_pool_2x2(h_conv1) # プーリング層1を作成

# 第2畳み込み層
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 結合層
# この時点で画像サイズは7*7にまで縮小されていて、画像全体の処理をするために1024個のニューロンと全結合
# 7*7まで縮小されてる理由は 2*2のプーリング層 を2回適用したので, [28,28] → [14,14] → [7,7] となる
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # tf.matmul は行列の乗算である

# 過学習軽減のためにドロップアウトをする   # プレースホルダーとはデータが格納される予定地である
keep_prob = tf.placeholder(tf.float32)
h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 最後にソフトマックス回帰のためのソフトマックス層を作成
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)

# モデルの訓練と評価
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # Adam法を使用してクロスエントロピー誤差を下げていく

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast: 新しい配列形式にする(この場合float32に変換)   # reduce_mean: 平均を求める
sess.run(tf.global_variables_initializer()) # 学習を開始する前に変数の初期化を行う

print("-------------- 畳み込みネットワーク --------------")
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:  # 100回ごとにログを表示
        train_accury = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob:1.0})   # eval: 文の呼び出し
        print("step %d, training accury : %g"%(i, train_accury))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accury: %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# 99.2%
