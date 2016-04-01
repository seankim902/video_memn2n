# -*- coding: utf-8 -*-


import os
import time
import pandas as pd
import numpy as np

from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import tensorflow as tf


def prepare_data(seqs_fn, maxlen=None, dim_img=4096):
    n_samples = len(seqs_fn)
    seqs_x = map(lambda fn: np.load(fn.replace("npy","mp4.npy")), seqs_fn)

    x = np.zeros((n_samples, maxlen, dim_img))
        
    for idx, s_x in enumerate(seqs_x):  
        x[idx][:len(s_x)] = s_x

    return x
    

def get_minibatch_indices(n, batch_size, shuffle=False):

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size
    if (minibatch_start != n):   
        minibatches.append(idx_list[minibatch_start:])
    return minibatches
    
    
    
class MemN2N(object):
    
    def __init__(self, config):
        self.img_dim = config.img_dim
        self.mem_dim = config.mem_dim
        self.mem_size = config.mem_size
        self.n_hop = config.n_hop
        self.n_y = config.n_y
        self.batch_size = config.batch_size
        self.do_prob = config.do_prob # keep prob
        self.nl = config.nl # keep prob
        self.learning_rate = config.learning_rate

        self.A = tf.Variable(tf.random_normal([self.img_dim, self.mem_dim]))
        self.C = tf.Variable(tf.random_normal([self.img_dim, self.mem_dim]))

        self.T_A = tf.Variable(tf.linspace(0.0, np.float32(self.mem_size-1), self.mem_size))
        #self.T_C = tf.Variable(tf.random_normal([self.mem_size, self.mem_dim]))
        self.T_C = tf.Variable(tf.linspace(0.0, np.float32(self.mem_size-1), self.mem_size))
        self.W_u = tf.Variable(tf.random_normal([self.mem_dim, self.mem_dim]))
        
        self.W_o = tf.Variable(tf.random_normal([self.mem_dim, self.n_y]))
        self.b_o = tf.Variable(tf.random_normal([self.n_y]))
        

    def build_model(self):
        self.saver = tf.train.Saver()
        
        videos = tf.placeholder(tf.float32, [None, self.mem_size, self.img_dim], name="videos")
        u = tf.constant(0.1, shape=[self.batch_size, self.mem_dim], name="u")
        y = tf.placeholder(tf.float32, [None, self.n_y], name="y")

        temp_videos_A = tf.matmul(tf.reshape(videos, [-1, self.img_dim]), self.A)
        temp_videos_A = tf.reshape(temp_videos_A, [-1, self.mem_size, self.mem_dim])
        temp_videos_A = tf.transpose(temp_videos_A, perm=[0, 2, 1]) + self.T_A #  sample * mem_dim * mem_size
        videos_input_m = tf.transpose(temp_videos_A, perm=[0, 2, 1]) #  sample * mem_size * mem_dim
        temp_videos_C = tf.matmul(tf.reshape(videos, [-1, self.img_dim]), self.C)  
        temp_videos_C = tf.reshape(temp_videos_C, [-1, self.mem_size, self.mem_dim])
        temp_videos_C = tf.transpose(temp_videos_C, perm=[0, 2, 1]) + self.T_C #  sample * mem_dim * mem_size
        videos_output_m =  tf.transpose(temp_videos_C, perm=[0, 2, 1]) # sample * mem_size * mem_dim
    
        os = [] 
        for _ in range(self.n_hop):
            u = tf.expand_dims(u, -1) # sample * mem_dim -> sample * mem_dim * 1
            in_m = tf.batch_matmul(videos_input_m, u)
            in_m = tf.squeeze(in_m)

            in_probs = tf.nn.softmax(in_m) # sample * mem_size
    
            out_m = tf.mul(videos_output_m, tf.tile(tf.expand_dims(in_probs, 2), [1, 1, self.mem_dim]))
            o = tf.reduce_sum(out_m, 1) # sample * mem_dim
            
            u = tf.add(tf.matmul(tf.squeeze(u), self.W_u), o)


            if self.nl:
                u = tf.nn.relu(u)
            
            os.append(u)


        os[-1] = tf.nn.dropout(os[-1], self.do_prob)
        y_hat = tf.nn.xw_plus_b(os[-1], self.W_o, self.b_o)
        pred = tf.argmax(tf.nn.softmax(y_hat), 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(y, 1)), tf.float32))
        
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_hat, y)
        loss = tf.reduce_mean(cross_entropy)
        

        regularizers = (tf.nn.l2_loss(self.A) + tf.nn.l2_loss(self.C) +
                        tf.nn.l2_loss(self.T_A) + tf.nn.l2_loss(self.T_C) +
                        tf.nn.l2_loss(self.W_u) + tf.nn.l2_loss(self.W_o) + tf.nn.l2_loss(self.b_o))        
        loss += 5e-4 * regularizers
        
        global_step = tf.Variable(0)
        
        lr = tf.train.exponential_decay( self.learning_rate,  # Base learning rate.
                                         global_step,         # Current index
                                         200,                 # Decay step.
                                         0.96,                # Decay rate.
                                         staircase=True)
  
      
        
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        
        return videos, u, y, loss, lr, train_op, pred, accuracy




        
        


def train():
    config = get_config()

    os.chdir('/home/seonhoon/Desktop/workspace/liris_video/')
    
    train = pd.read_pickle('train.pkl')
    valid = pd.read_pickle('valid.pkl')

    train_x_fn = train['fn']
    train_y1 = train['v']
    train_y2 = train['a']
    
    valid_x_fn = valid['fn']
    valid_y1 = valid['v']
    valid_y2 = valid['a']
    
    train_y = train_y1
    valid_y = valid_y1

    valid_y = np_utils.to_categorical(valid_y, config.n_y).astype('float32')
    valid_x = prepare_data(valid_x_fn, config.steps, config.img_dim)
    valid_batch_indices=get_minibatch_indices(len(valid_x), config.batch_size, shuffle=False)

    
    print 'valid_x :', valid_x.shape
    print 'valid_y :',valid_y.shape

    
    with tf.Session() as sess:
        
        initializer = tf.random_normal_initializer(0, 0.05)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = MemN2N(config = config)
           
        videos, u, y, loss, lr, train_op, pred, accuracy = model.build_model()
        
        sess.run(tf.initialize_all_variables())
        #model.saver.restore(sess, config.model_ckpt_path+'-42')
        
        best = 0.4
        for i in range(config.epoch):
            start = time.time()

            batch_indices=get_minibatch_indices(len(train_x_fn), config.batch_size, shuffle=True)

            for j, indices in enumerate(batch_indices):
                # train_x_fn, train_y
                x_ = np.array([ train_x_fn[k] for k in indices])
                x_ = prepare_data(x_, config.steps, config.img_dim)
                y_ = np.array([ train_y[k] for k in indices])
                y_ = np_utils.to_categorical(y_, config.n_y).astype('float32')
                #u_ = np.ndarray([x_.shape[0], config.mem_dim], dtype=np.float32)
                
                
                _loss, _lr, _, _pred, acc = sess.run([loss, lr, train_op, pred, accuracy],
                                              {videos: x_,
                                               y: y_})
                if j % 30 == 0 :
                    print 'cost : ', _loss, ', accuracy : ', acc, ', lr : ', _lr, ', iter : ', j+1, ' in epoch : ',i+1
            print 'cost : ', _loss, ', accuracy : ', acc, ', lr : ', _lr, ', iter : ', j+1, ' in epoch : ',i+1,' elapsed time : ', int(time.time()-start)
            
            if config.valid_epoch is not None:  # for validation
                
                
                if (i+1) % config.valid_epoch == 0:
                    val_preds = []
                    for j, indices in enumerate(valid_batch_indices):
                        x_ = np.array([ valid_x[k,:,:] for k in indices])
                        y_ = np.array([ valid_y[k,:] for k in indices])
                        
                        _pred = sess.run(pred,
                                        {videos: x_,
                                         y: y_})
               
                        val_preds = val_preds + _pred.tolist()
                    valid_acc = np.mean(np.equal(val_preds, np.argmax(valid_y, 1)))
                    print '##### valid accuracy : ', valid_acc, ' after epoch ', i+1
                    if valid_acc > best :
                        best = valid_acc
                        print 'save model...',
                        model.saver.save(sess, config.model_ckpt_path, global_step=int(best*100))
                        print int(best*100)
        
        print 'best valid accuracy :', best
                                       




def get_config():
    class Config1(object):
        
        
        img_dim = 4096
        mem_dim = 512
        mem_size = steps = 30
        n_hop = 3
        n_y = 3
        batch_size = 100
        do_prob = 0.6
        nl = True
        learning_rate = 0.001
        epoch = 80
        
        valid_epoch = 1 # or None
        model_ckpt_path = '/home/seonhoon/Desktop/workspace/liris_video/mem_model.ckpt'
    return Config1()
    
    
    
def main(_):


    is_train = True  # if False then test
    
    if is_train :
        train()
    

if __name__ == "__main__":
  tf.app.run()