from scipy.stats import truncnorm
import tensorflow as tf

tf_tn = tf.random.truncated_normal([512, 100], stddev=0.1)
sp_tn = truncnorm.std([512, 100], scale=0.1)

print(tf_tn, sp_tn)
