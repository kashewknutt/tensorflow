import tensorflow as tf
print("This is me working with tensorflow for the first time!!")
print(tf.version)

t = tf.zeros([5,5,5,5])
#print(t)
t = tf.reshape(t, [625])
print(t)
