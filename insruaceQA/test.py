import tensorflow as tf

a=tf.fill([10],0)

with tf.Session() as sess:
    print(sess.run(a))