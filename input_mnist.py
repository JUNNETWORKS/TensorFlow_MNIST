import tensorflow as tf
import cv2
import numpy as np

NUM_CLASSES = 10

def interence(imegs_placeholder, keep_prob):
    """ 予測モデルを作成する関数

    引数:
      images_placeholder: 画像のplaceholder
      keep_prob: dropout率のplaceholder

    返り値:
      y_conv: 各クラスの確率(のようなもの)

     with tf.name_scope("xxx") as scope:
         これでTensorBoard上に一塊のノードとし表示される
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
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

    # プーリング層
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # 入力層を28*28*1に変形
    x_image = tf.reshape(imegs_placeholder, [-1, 28, 28, 1])

    # 畳み込み層1の作成
    with tf.name_scope("conv1") as scope:
        W_conv1 = weight_variable([3,3,1,16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # 畳み込み層2の作成
    with tf.name_scope("conv2") as scope:
        W_conv2 = weight_variable([3,3,16,16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # プーリング層1の作成
    with tf.name_scope("pool1") as scope:
        h_pool1 = max_pool_2x2(h_conv2)

    # 畳み込み層3の作成
    with tf.name_scope("conv3") as scope:
        W_conv3 = weight_variable([3,3,16,32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    # 畳み込み層4の作成
    with tf.name_scope("conv4") as scope:
        W_conv4 = weight_variable([3,3,32,32])
        b_conv4 = bias_variable([32])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    # プーリング層2の作成
    with tf.name_scope("pool2") as scope:
        h_pool2 = max_pool_2x2(h_conv4)

    # 畳み込み層5の作成
    with tf.name_scope("conv5") as scope:
        W_conv5 = weight_variable([3,3,32,64])
        b_conv5 = bias_variable([64])
        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

    # 畳み込み層6の作成
    with tf.name_scope("conv6") as scope:
        W_conv6 = weight_variable([3,3,64,64])
        b_conv6 = bias_variable([64])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    # プーリング層3の作成
    with tf.name_scope("pool3") as scope:
        h_pool3 = max_pool_2x2(h_conv6)

    # 結合層1の作成
    with tf.name_scope("fc1") as scope:
        W_fc1 = weight_variable([4*4*64, 1024])   #参照: http://qiita.com/tadOne/items/48302a399dcad44c69c8
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        # dropout1の設定
        h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 結合層2の作成
    with tf.name_scope("fc2") as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
#        h_fc2 = tf.nn.relu(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)
        # dropout2の設定
#        h_fc_2_drop = tf.nn.dropout(h_fc2, keep_prob)

#    # 結合層の3(?)の作成
#    with tf.name_scope("fc3") as scope:
#        W_fc3 = weight_variable([1024, NUM_CLASSES])
#        b_fc3 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    with tf.name_scope("softmax") as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv

if __name__ == "__main__":

    # 画像読み込み
    img = input("画像パスを入力してください >")
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    ximage = img.flatten().astype(np.float32)/255.0 #形式を変更

    # 式に用いる変数設定
    x_image = tf.placeholder("float", shape=[None, 784])    # 入力
    y_label = tf.placeholder("float", shape=[None, 10])
    keep_prob = tf.placeholder("float")

    logits = interence(x_image, keep_prob)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('./ckpt')
    saver.restore(sess, ckpt.model_checkpoint_path) # 変数データの読み込み

    pred = np.argmax(logits.eval(feed_dict={x_image: [ximage], keep_prob: 1.0})[0])
    print(pred)
