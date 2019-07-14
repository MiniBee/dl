import tensorflow as tf 

class Model():
    """
    表情识别中使用VGGNet-16作为基本模型
    """
    def __init__(self, batch_size=32, is_training=True):
        """
        初始化类
        """
        self.batch_size = batch_size 
        self.is_training = is_training
        self.build_model()
        self.init_sess() 
    def build_model(self):
        """
        构建计算图
        """
        self.graph = tf.Graph() 
        def block(net, n_conv, n_chl, blockID):
            """
            定义多个CNN组合单元
            """
            with tf.variable_scope("block%d"%blockID):
                for itr in range(n_conv):
                    net = tf.layers.conv2d(net, 
                                           n_chl, 3, 
                                           activation=tf.nn.relu, 
                                           padding="same")
                net = tf.layers.max_pooling2d(net, 2, 2)
            return net 
        with self.graph.as_default():
            # 人脸数据
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="inputs")
            # 表情序列，用0-6数字表示
            # self.target = tf.placeholder(tf.int32, 
            #                              [self.batch_size],
            #                              name="target") 
            self.target_onehot = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="target_onehot")
            net = block(self.inputs, 2, 64, 1)
            net = block(net, 2, 128, 2)
            # net = block(net, 2, 256, 3)
            # net = block(net, 2, 512, 4)
            # net = block(net, 2, 512, 5)
            net = tf.layers.flatten(net)
            net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
            net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
            self.logits = tf.layers.dense(net, 10, activation=None)
            # self.y_pred = tf.nn.softmax(self.logits)

            # 计算loss函数
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.target_onehot, 
                logits=self.logits
            )
            self.loss = tf.reduce_mean(self.loss)
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.target_onehot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # 优化
            self.step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
            self.all_var = tf.global_variables() 
            self.init = tf.global_variables_initializer() 
            self.saver = tf.train.Saver()
    def init_sess(self, restore=None):
        """
        初始化会话
        """
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        if restore != None:
            self.saver.restore(self.sess, restore)