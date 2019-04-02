import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import itertools
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

class MLP_Nnet(object):
    def __init__(self, HiddenNodes=25, epochs=10000, LCDM=True):
        tf.reset_default_graph()
        self.N_EPOCHS = epochs
        self.h_size = HiddenNodes
        self.LCDM = LCDM
        self.grad_stepsize = 1e-3
        if LCDM:
            self.dirName = 'MetaGraphs/LCDM_TT_Cls'
            self.fileN = self.dirName + '/LCDM_TT_Cls_Graph_Global_'
        else:
            # Fix this
            self.dirName = 'MetaGraphs/ALPHA_Tb_Global'
            self.fileN = self.dirName + '/ALPHA_21cm_Graph_Global_'
        
        if not os.path.exists(self.dirName):
            os.mkdir(self.dirName)
    
    def init_weights(self, shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, stddev=0.4)
        return tf.Variable(weights)

    def forwardprop(self, X, w_1, w_2, w_3):
        """
        Forward-propagation.
        """
        hid1 = tf.nn.sigmoid(tf.matmul(X, w_1))
        hid2 = tf.nn.sigmoid(tf.matmul(hid1, w_2))
        yhat = tf.matmul(hid2, w_3)
        return yhat

    def get_data(self, frac_test=0.3):
        self.scalar = StandardScaler()
        self.scalar_Y = StandardScaler()
        if self.LCDM:
            fileNd = 'Data/LCDM_TT_Cls.dat'
            inputN = 6

        dataVals = np.loadtxt(fileNd)
        np.random.shuffle(dataVals)
        data = dataVals[:, :inputN]
        target = dataVals[:, inputN:]
        
        dataSTD = self.scalar.fit_transform(data)
        self.train_size = (1.-frac_test)*len(dataVals[:,0])
        self.test_size = frac_test*len(dataVals[:,0])
        # Prepend the column of 1s for bias
        N, M  = data.shape
        all_X = np.ones((N, M + 1))
        all_X[:, 1:] = dataSTD

        return train_test_split(all_X, target, test_size=frac_test, random_state=RANDOM_SEED)

    def main_nnet(self):#, train_nnet=True, eval_nnet=False, evalVec=[], keep_training=False):
        
        self.train_X, self.test_X, self.train_y, self.test_y = self.get_data()
        
        # Layer's sizes
        self.x_size = self.train_X.shape[1]   # 
        self.y_size = self.train_y.shape[1]   # 

        # Symbols
        self.X = tf.placeholder("float", shape=[None, self.x_size], name='X')
        self.y = tf.placeholder("float", shape=[None, self.y_size])

        # Weight initializations
        self.w_1 = self.init_weights((self.x_size, self.h_size))
        self.w_2 = self.init_weights((self.h_size, self.h_size))
        self.w_3 = self.init_weights((self.h_size, self.y_size))

        # Forward propagation
        self.yhat = self.forwardprop(self.X, self.w_1, self.w_2, self.w_3)
        
        tf.add_to_collection("activation", self.yhat)
        # Backward propagation
        
        self.cost = tf.reduce_sum(tf.square((self.y - self.yhat), name="cost"))
        # Error Check
        self.perr_train = tf.reduce_sum(tf.abs((self.y - self.yhat) / self.y))
        self.perr_test = tf.reduce_sum(tf.abs((self.y - self.yhat) / self.y))
        self.updates = tf.train.GradientDescentOptimizer(self.grad_stepsize).minimize(self.cost)
        
        self.saveNN = tf.train.Saver()

        return

    def train_NN(self, evalVec, keep_training=False):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if keep_training:
                self.saveNN.restore(sess, self.fileN)
                print 'Model Restored.'
            BATCH_SIZE = 20
            train_count = len(self.train_X)
            for i in range(1, self.N_EPOCHS + 1):
                for start, end in zip(range(0, train_count, BATCH_SIZE),
                                      range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
                    sess.run(self.updates, feed_dict={self.X: self.train_X[start:end],
                                                      self.y: self.train_y[start:end]})

                if i % 100 == 0:
                    
                    train_accuracy = sess.run(self.perr_train, feed_dict={self.X: self.train_X, self.y: self.train_y})
                    test_accuracy = sess.run(self.perr_test, feed_dict={self.X: self.test_X, self.y: self.test_y})
                    
                    
                    print("Epoch = %d, train accuracy = %.7e, test accuracy = %.7e"
                          % (i + 1, train_accuracy/len(self.train_X), test_accuracy/len(self.test_X)))
                    if i == 100:
                        hold_train = train_accuracy/len(self.train_X)
                        hold_test = test_accuracy/len(self.test_X)
                    else:
                        if (hold_train - train_accuracy/len(self.train_X)) < 1e-3:
                            self.grad_stepsize / 2.
                        if (hold_test - test_accuracy/len(self.test_X)) < 0.:
                            print 'Potential Overtraining, breaking loop...'
                            break
                        if self.grad_stepsize < 1e-9:
                            print 'Step Size Below 1e-9, breaking loop...'
                            break
#                    predictions = sess.run(self.yhat, feed_dict={self.X: np.insert(self.scalar.transform(evalVec), 0, 1., axis=1)})
#                    print 'Current Predictions: ', predictions
            self.saveNN.save(sess, self.fileN)
        return

    def eval_NN(self, evalVec):
        with tf.Session() as sess:
            saverMeta = tf.train.import_meta_graph(self.fileN + '.meta')
            self.saveNN.restore(sess, self.fileN)
            predictions = sess.run(self.yhat, feed_dict={self.X: np.insert(self.scalar.transform(evalVec), 0, 1., axis=1)})
            if not self.globalTb:
                return np.power(10, predictions)
        return predictions

    def load_matrix_elems(self):
        with tf.Session() as sess:
            self.saveNN.restore(sess, self.fileN)
            self.Matrix1 = sess.run(self.w_1)
            self.Matrix2 = sess.run(self.w_2)
            self.Matrix3 = sess.run(self.w_3)
        return

    def rapid_eval(self, evalVec):
        inputV = np.insert(self.scalar.transform(evalVec), 0, 1., axis=1)
        h1 = self.sigmoid(np.matmul(inputV, self.Matrix1))
        h2 = self.sigmoid(np.matmul(h1, self.Matrix2))
        predictions = np.matmul(h2, self.Matrix3)
        
        return predictions

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))


class ImportGraph():
    def __init__(self, metaFile, dataFile, LCDM=True):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(metaFile + '.meta')
            saver.restore(self.sess, metaFile)
            self.activation = tf.get_collection('activation')[0]

        self.scalar = StandardScaler()
        dataIN = np.loadtxt(dataFile)
        if LCDM:
            n_in = 6
        input_v = dataIN[:, :n_in]
        std_input_v = self.scalar.fit_transform(input_v)
        return

    def run_yhat(self, data):
        inputV = np.insert(self.scalar.transform(data), 0, 1., axis=1)
        return self.sess.run(self.activation, feed_dict={"X:0": inputV})

