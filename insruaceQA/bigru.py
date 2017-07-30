import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from utils import ortho_weight, uniform_weight
## define lstm model and reture related features


# return n outputs of the n lstm cells
def BIGRU(x, hidden_size):
	# biLSTM��
	# ���ܣ����bidirectional_lstm����
	# ������
	# 	x: [batch, height, width]   / [batch, step, embedding_size]
	# 	hidden_size: lstm���ز�ڵ����
	# �����
	# 	output: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]

	# input transformation
	input_x = tf.transpose(x, [1, 0, 2])
	# input_x = tf.reshape(input_x, [-1, w])
	# input_x = tf.split(0, h, input_x)
	input_x = tf.unstack(input_x)

	# define the forward and backward lstm cells
	gru_fw_cell = rnn_cell.GRUCell(hidden_size)
	gru_bw_cell = rnn_cell.GRUCell(hidden_size)
	output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(gru_fw_cell, gru_bw_cell, input_x, dtype=tf.float32)

	# output transformation to the original tensor type
	output = tf.stack(output)
	output = tf.transpose(output, [1, 0, 2])
	return output

def BIGRU_ATT(input_x, rnn_size, batch_size, is_att=False, summary_state=None):
    with tf.variable_scope("FW") as fw_scope:
        h_encoder = GRU_ATT(input_x, rnn_size, batch_size, fw_scope, summary_state, is_att)
    with tf.variable_scope("BW") as bw_scope:
        #h_encoder_rev = GRU_ATT(tf.reverse(input_x, [False, True, False]), rnn_size, batch_size, bw_scope, summary_state, is_att)
        h_encoder_rev = GRU_ATT(tf.reverse(input_x, [1]), rnn_size, batch_size, bw_scope,summary_state, is_att)
    #h_encoder_rev = tf.reverse(h_encoder_rev, [False, True, False])
    h_encoder_rev = tf.reverse(h_encoder_rev, [1])
    output = tf.concat(axis=2, values=[h_encoder, h_encoder_rev])
    return output#(batch_size, steps, rnn_size * 2)

def GRU_ATT(input_x, rnn_size, batch_size, vscope, summary_state=None, is_att=True):#input(batch_size, steps, embedding_size)
    num_steps = int(input_x.get_shape()[1])
    embedding_size = int(input_x.get_shape()[2])
    output = []#(steps, batch_size, rnn_size)
    with tf.variable_scope(vscope):
        #define parameter
        W = tf.get_variable("W", initializer=tf.concat(axis=1, values=[uniform_weight(embedding_size, rnn_size), uniform_weight(embedding_size, rnn_size)]))
        U = tf.get_variable("U", initializer=tf.concat(axis=1, values=[ortho_weight(rnn_size), ortho_weight(rnn_size)]))
        b = tf.get_variable("b", initializer=tf.zeros([2 * rnn_size]))
        Wx = tf.get_variable("Wx", initializer=uniform_weight(embedding_size, rnn_size))
        Ux = tf.get_variable("Ux", initializer=ortho_weight(rnn_size))
        bx = tf.get_variable("bx", initializer=tf.zeros([rnn_size]))
        M = tf.get_variable("M", initializer=tf.concat(axis=1, values=[uniform_weight(2 * rnn_size, rnn_size), uniform_weight(2 * rnn_size, rnn_size)]))#attention weight
        h_ = tf.zeros([batch_size, rnn_size])
        one = tf.fill([batch_size, rnn_size], 1.)
        state_below = tf.transpose(tf.matmul(input_x, tf.tile(tf.reshape(W, [1, embedding_size, 2 * rnn_size]), [batch_size, 1, 1])) + b, perm=[1, 0, 2])
        state_belowx = tf.transpose(tf.matmul(input_x, tf.tile(tf.reshape(Wx, [1, embedding_size, rnn_size]), [batch_size, 1, 1])) + bx, perm=[1, 0, 2])#(steps, batch_size, rnn_size)
        for time_step in range(num_steps):
            preact = tf.matmul(h_, U)
            preact = tf.add(preact, state_below[time_step])
            if is_att and summary_state is not None:
                preact = preact + tf.matmul(summary_state, M)#add attention
            
            r = tf.nn.sigmoid(_slice(preact, 0, rnn_size))
            u = tf.nn.sigmoid(_slice(preact, 1, rnn_size))

            preactx = tf.matmul(h_, Ux)
            preactx = tf.multiply(preactx, r)
            preactx = tf.add(preactx, state_belowx[time_step])
            h = tf.tanh(preactx)

            h_ = tf.add(tf.multiply(u, h_), tf.multiply(tf.subtract(one, u), h))
            output.append(h_)
    output = tf.transpose(output, perm=[1, 0, 2])
    return output#(batch_size, steps, rnn_size)

def _slice(_x, n, dim):
    if len(_x.get_shape()) == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]
