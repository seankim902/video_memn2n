# -*- coding: utf-8 -*-


import os
import time
import pandas as pd
import numpy as np

from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell


def prepare_data(seqs_fn, maxlen=None, dim_img=4096):
    n_samples = len(seqs_fn)
    seqs_x = map(lambda fn: np.load(fn), seqs_fn)

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
        self.hidden = config.hidden
        self.mem_dim = config.mem_dim
        self.mem_size = self.steps = config.mem_size
        self.n_hop = config.n_hop
        self.n_y = config.n_y
        self.batch_size = config.batch_size
        self.do_prob = config.do_prob # keep prob
        self.nl = config.nl 
        self.learning_rate = config.learning_rate

        #rnn_ = rnn_cell.GRUCell(self.hidden)
        rnn_ = rnn_cell.BasicLSTMCell(self.hidden, forget_bias=1.0)
        num_layers = 2
        if True :
            rnn_ = rnn_cell.DropoutWrapper(
                rnn_, output_keep_prob=0.7)
        my_rnn = rnn_cell.MultiRNNCell([rnn_] * num_layers)

        self.my_rnn = my_rnn #= rnn_
        self.init_state = my_rnn.zero_state(self.batch_size, tf.float32) #tf.zeros([batch_size, my_rnn.state_size])

        self.W_iemb = tf.get_variable("W_iemb", [self.img_dim, self.hidden])
        self.b_iemb = tf.get_variable("b_iemb", [self.hidden])
        
        self.global_step = tf.Variable(0, name='g_step')
        self.A = tf.Variable(tf.random_normal([self.hidden, self.mem_dim]), name='A')
        self.C = tf.Variable(tf.random_normal([self.hidden, self.mem_dim]))

#        self.T_A = tf.Variable(tf.linspace(0.0, np.float32(self.mem_size-1), self.mem_size))
#        self.T_C = tf.Variable(tf.linspace(0.0, np.float32(self.mem_size-1), self.mem_size))
        
        self.W_u = tf.Variable(tf.random_normal([self.mem_dim, self.mem_dim]))        
        self.W_o = tf.Variable(tf.random_normal([self.mem_dim, self.n_y]))
        self.b_o = tf.Variable(tf.random_normal([self.n_y]))
        


    def build_model(self):
        
        videos = tf.placeholder(tf.float32, [self.batch_size, self.steps, self.img_dim])
        x_ = tf.reshape(videos, [-1, self.img_dim])
        # sample * steps * dim -> (sample * steps) * dim
        x_ = tf.nn.xw_plus_b(x_, self.W_iemb, self.b_iemb)
        
        img_input = tf.reshape(x_, [self.batch_size, self.steps, self.hidden])


        hiddens = []
        hidden = state = self.init_state
        with tf.variable_scope("RNN", reuse=None):
            for i in range(self.steps): 
                if i > 0 :
                    tf.get_variable_scope().reuse_variables()
             
                (hidden, state) = self.my_rnn(img_input[:,i,:], state)
                hiddens.append(hidden)
        
        hiddens = tf.pack(hiddens)
        hiddens = tf.squeeze(hiddens)
        hiddens = tf.transpose(hiddens, perm=[1, 0, 2])                
                
        
        videos_ = hiddens
        u = tf.constant(0.1, shape=[self.batch_size, self.mem_dim], name="u")
        y = tf.placeholder(tf.float32, [None, self.n_y], name="y")

        temp_videos_A = tf.matmul(tf.reshape(videos_, [-1, self.hidden]), self.A)
        temp_videos_A = tf.reshape(temp_videos_A, [-1, self.mem_size, self.mem_dim])
        temp_videos_A = tf.transpose(temp_videos_A, perm=[0, 2, 1]) # + self.T_A #  sample * mem_dim * mem_size
        videos_input_m = tf.transpose(temp_videos_A, perm=[0, 2, 1]) #  sample * mem_size * mem_dim
        temp_videos_C = tf.matmul(tf.reshape(videos_, [-1, self.hidden]), self.C)  
        temp_videos_C = tf.reshape(temp_videos_C, [-1, self.mem_size, self.mem_dim])
        temp_videos_C = tf.transpose(temp_videos_C, perm=[0, 2, 1]) # + self.T_C #  sample * mem_dim * mem_size
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
                u = tf.nn.elu(u)
            
            os.append(u)

        train_hidden = valid_hidden = os[-1]
        train_hidden = tf.nn.dropout(train_hidden, self.do_prob)
        
        y_hat = tf.nn.xw_plus_b(train_hidden, self.W_o, self.b_o)
        pred = tf.argmax(tf.nn.softmax(y_hat), 1)
        
        v_y_hat = tf.nn.xw_plus_b(valid_hidden, self.W_o, self.b_o)
        v_pred = tf.argmax(tf.nn.softmax(v_y_hat), 1)       
            
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(y, 1)), tf.float32))
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_hat, y)
        
        loss = tf.reduce_mean(cross_entropy)

        regularizers = (tf.nn.l2_loss(self.A) + tf.nn.l2_loss(self.C) +
                        #tf.nn.l2_loss(self.T_A) + tf.nn.l2_loss(self.T_C) +
                        tf.nn.l2_loss(self.W_iemb) + tf.nn.l2_loss(self.b_iemb) +
                        tf.nn.l2_loss(self.W_u) + tf.nn.l2_loss(self.W_o) + tf.nn.l2_loss(self.b_o))        
        loss += 5e-3 * regularizers
        
        lr = tf.train.exponential_decay( self.learning_rate,  # Base learning rate.
                                         self.global_step,    # Current index
                                         200,                 # Decay step.
                                         0.96,                # Decay rate.
                                         staircase=True)
        
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=self.global_step)
        
        return videos, u, y, loss, train_op, lr, pred, v_pred, accuracy



def train():
    config = get_config()

    os.chdir('/home/devbox/workspace/liris/')
    
    train = pd.read_pickle('train.pkl')
    valid = pd.read_pickle('valid.pkl')

    train_x_fn = train['fn']
    train_y1 = train['v']
    train_y2 = train['a']
    
    valid_x_fn = valid['fn']
    valid_y1 = valid['v']
    valid_y2 = valid['a']
    

    train_y = train_y2
    valid_y = valid_y2

    valid_y = np_utils.to_categorical(valid_y, config.n_y).astype('float32')
    valid_x = prepare_data(valid_x_fn, config.steps, config.img_dim)
    valid_batch_indices=get_minibatch_indices(len(valid_x), config.batch_size, shuffle=False)


    # train data should be embedded in the loop if these are large
    train_y = np_utils.to_categorical(train_y, config.n_y).astype('float32')
    train_x = prepare_data(train_x_fn, config.steps, config.img_dim)
    batch_indices=get_minibatch_indices(len(train_x), config.batch_size, shuffle=True)

    
    print 'train_x :',valid_x.shape
    print 'valid_y :',valid_y.shape

    print 'train_x :', valid_x.shape
    print 'valid_y :',valid_y.shape


    
    with tf.Session() as sess:
        initializer = tf.random_normal_initializer(0, 0.05)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = MemN2N(config = config)          
            
        videos, u, y, loss, train_op, lr, pred, v_pred, accuracy = model.build_model()
        
        
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        #saver.restore(sess, config.model_ckpt_path+'-43')
        
        best = 0.
        for i in range(config.epoch):
            start = time.time()
            
            for j, indices in enumerate(batch_indices):
                # train_x_fn, train_y
                x_ = np.array([ train_x[k,:,:] for k in indices])
                y_ = np.array([ train_y[k,:] for k in indices])
                
                _loss,  _, _lr, acc = sess.run([loss, train_op, lr, accuracy],
                                              {videos: x_,
                                               y: y_})
                if j % 20 == 0 :
                    print 'cost : ', _loss, ', accuracy : ', acc, ', lr : ', _lr, ', iter : ', j+1, ' in epoch : ',i+1
            print 'cost : ', _loss, ', accuracy : ', acc, ', lr : ', _lr, ', iter : ', j+1, ' in epoch : ',i+1,' elapsed time : ', int(time.time()-start)
            
            if config.valid_epoch is not None:  # for validation
                
                if (i+1) % config.valid_epoch == 0:
                    val_preds = []
                    for j, indices in enumerate(valid_batch_indices):
                        x_ = np.array([ valid_x[k,:,:] for k in indices])
                        y_ = np.array([ valid_y[k,:] for k in indices])
                        
                        _pred = sess.run(v_pred,
                                        {videos: x_,
                                         y: y_})
               
                        val_preds = val_preds + _pred.tolist()
                    valid_acc = np.mean(np.equal(val_preds, np.argmax(valid_y, 1)))
                    print best, '##### valid accuracy : ', valid_acc, ' after epoch ', i+1
                    if valid_acc > best :
                        best = valid_acc
                        print 'save model...',
                        saver.save(sess, config.model_ckpt_path, global_step=int(best*100))
                        print int(best*100)
        
        print 'best valid accuracy :', best
                                       


def get_config():
    class Config1(object):
        
        img_dim = 4096
        hidden = 512
        mem_dim = 512
        mem_size = steps = 20
        n_hop = 2
        n_y = 3
        batch_size = 100
        do_prob = 0.6
        nl = True
        learning_rate = 0.001
        epoch = 70
        
        valid_epoch = 1 # or None
        model_ckpt_path = '/home/devbox/workspace/seonhoon/mem_model.ckpt'
    return Config1()
    
    
def main(_):

    is_train = True
    
    if is_train :
        train()

if __name__ == "__main__":
  tf.app.run()