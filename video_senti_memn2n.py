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
    seqs_x = map(lambda fn: np.load('/home/seonhoon/Desktop/workspace/instagram/feats/feat_40000/'+fn+'.mp4.npy'), seqs_fn)

    x = np.zeros((n_samples, maxlen, dim_img))
        
    for idx, s_x in enumerate(seqs_x):  
        x[idx][:len(s_x)] = s_x  # [:80] ->  

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
        self.nhop = config.nhop
        self.n_y = config.n_y
        self.batch_size = config.batch_size
        self.nl_prob = config.nl_prob # non linearity, keep prob
        self.lr = config.lr

        self.A = tf.Variable(tf.random_normal([self.img_dim, self.mem_dim]))
        self.C = tf.Variable(tf.random_normal([self.img_dim, self.mem_dim]))

        self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.mem_dim]))
        self.T_C = tf.Variable(tf.random_normal([self.mem_size, self.mem_dim]))

        self.W = tf.Variable(tf.random_normal([self.mem_dim, self.n_y]))
        self.b = tf.Variable(tf.random_normal([self.n_y]))

    def build_model(self):
        videos = tf.placeholder(tf.float32, [None, self.mem_size, self.img_dim], name="videos")
        u = tf.constant(1./self.batch_size, shape=[self.mem_dim, 1])
        y = tf.placeholder(tf.float32, [None, self.n_y], name="y")

        temp_videos_A = tf.matmul(tf.reshape(videos, [-1, self.img_dim]), self.A)
        temp_videos_A = tf.reshape(temp_videos_A, [-1, self.mem_size, self.mem_dim])
        temp_videos_A = temp_videos_A + self.T_A
        videos_input_m = tf.reshape(temp_videos_A, [-1, self.mem_dim]) # (sample*mem_size) * mem_dim
        temp_videos_C = tf.matmul(tf.reshape(videos, [-1, self.img_dim]), self.C)  
        temp_videos_C = tf.reshape(temp_videos_C, [-1, self.mem_size, self.mem_dim])
        videos_output_m = temp_videos_C + self.T_C # sample * mem_size * mem_dim
    
        os = [] 
        for _ in range(self.nhop):
            in_m = tf.matmul(videos_input_m, u)
            in_m = tf.squeeze(in_m)
            in_m = tf.reshape(in_m, [-1, self.mem_size])

            in_probs = tf.nn.softmax(in_m) # sample * mem_size
    
            out_m = tf.mul(videos_output_m, tf.tile(tf.expand_dims(in_probs, 2), [1, 1, self.mem_dim]))
            o = tf.reduce_sum(out_m, 1) # sample * mem_dim
            
            o = o + tf.squeeze(u)

            if self.nl_prob:
                o = tf.nn.dropout(o, self.nl_prob)
                
            os.append(o)



        y_hat = tf.nn.xw_plus_b(os[-1], self.W, self.b)
        pred = tf.argmax(tf.nn.softmax(y_hat), 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(y, 1)), tf.float32))
        
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_hat, y)
        loss = tf.reduce_mean(cross_entropy)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return videos, y, loss, train_op, pred, accuracy







def train():
    config = get_config()
    os.chdir('/home/seonhoon/Desktop/workspace/instagram/')

    pilot = pd.read_pickle('whole2.pkl')

    x = pilot['mediaid']
    y = pilot['type']
    
    train_x_fn, valid_x_fn, train_y, valid_y = \
        train_test_split(x, y, test_size=0.1, random_state=43)    

        
    train_x_fn = train_x_fn.reset_index(drop=True)
    valid_x_fn = valid_x_fn.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    valid_y = valid_y.reset_index(drop=True)

    valid_y = np_utils.to_categorical(valid_y, config.n_y).astype('float32')
    valid_x = prepare_data(valid_x_fn, config.steps, config.img_dim)
    valid_batch_indices=get_minibatch_indices(len(valid_x), config.batch_size, shuffle=False)

    print 'valid_x :', valid_x.shape
    print 'valid_y :',valid_y.shape

    
    with tf.Session() as sess:
        
        initializer = tf.random_normal_initializer(0, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = MemN2N(config = config)
           
        videos, y, loss, train_op, pred, accuracy = model.build_model()
        
        sess.run(tf.initialize_all_variables())
        
        
        for i in range(config.epoch):
            start = time.time()

            batch_indices=get_minibatch_indices(len(train_x_fn), config.batch_size, shuffle=True)

            for j, indices in enumerate(batch_indices):
                # train_x_fn, train_y
                x_ = np.array([ train_x_fn[k] for k in indices])
                x_  = prepare_data(x_, config.steps, config.img_dim)
                y_ = np.array([ train_y[k] for k in indices])
                y_ = np_utils.to_categorical(y_, config.n_y).astype('float32')
                
                  
                
                cost, _, pr, acc = sess.run([loss, train_op, pred, accuracy],
                                              {videos: x_,
                                               y: y_})
                if j % 2 == 0 :
                    print 'cost : ', cost, ', accuracy : ', acc, ', iter : ', j+1, ' in epoch : ',i+1
            print 'cost : ', cost, ', accuracy : ', acc, ', iter : ', j+1, ' in epoch : ',i+1,' elapsed time : ', int(time.time()-start)
            
            if config.valid_epoch is not None:  # for validation
                
                
                if (i+1) % config.valid_epoch == 0:
                    val_preds = []
                    for j, indices in enumerate(valid_batch_indices):
                        x_ = np.array([ valid_x[k,:,:] for k in indices])
                        y_ = np.array([ valid_y[k,:] for k in indices])
                        
                        _pred = sess.run(pred,
                                        {x: x_,
                                         y: y_})
               
                        val_preds = val_preds + _pred.tolist()
                    valid_acc = np.mean(np.equal(val_preds, np.argmax(valid_y, 1)))
                    print '##### valid accuracy : ', valid_acc, ' after epoch ', i+1
                                       




def get_config():
    class Config1(object):
        
        
        img_dim = 4096
        mem_dim = 256
        mem_size = steps = 45
        nhop = 2
        n_y = 4
        batch_size = 256
        nl_prob = 0.9
        lr = 0.001
        epoch = 50
        
        valid_epoch = 1 # or None
        model_ckpt_path = '/home/seonhoon/Desktop/workspace/ImageQA/version_tensorflow/model/model.ckpt'
    return Config1()
    
    
    
def main(_):


    is_train = True  # if False then test
    
    if is_train :
        train()
    

if __name__ == "__main__":
  tf.app.run()