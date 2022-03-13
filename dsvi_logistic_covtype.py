import math

import numpy as np
import scipy.special as spsp
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = loadmat('original-code/Stein-Variational-Gradient-Descent/data/covertype.mat')
X_input = data['covtype'][:, 1:]
y_input = data['covtype'][:, 0]
y_input[y_input == 2] = 0

X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2)

N = X_train.shape[0]
M = X_train.shape[1]
T = 2

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

mu = tf.Variable(tf.zeros(M))
sigma0 = tf.Variable(tf.ones(M))
e = tf.random_normal([M])
z = e * sigma0 + mu

std_multi_normal = tf.contrib.distributions.MultivariateNormalDiag([0.0] * M)
log_p_z = std_multi_normal.log_prob(z)
log_p_d_given_z = tf.reduce_sum(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.matmul(X, tf.reshape(z, [-1, 1])))) / 5000

# acc, acc_op = tf.metrics.accuracy(labels=y, predictions=tf.matmul(X, tf.reshape(z, [-1, 1])))


dist_q = tf.contrib.distributions.MultivariateNormalDiag(mu, sigma0)  # negative scales are allowed.
# entropy_q = tf.contrib.bayesflow.entropy.entropy_shannon(dist_q)
entropy_q = dist_q.entropy()

loss = (-entropy_q - log_p_d_given_z - log_p_z) / 5000
optimizer = tf.train.GradientDescentOptimizer(100)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# X_data = np.random.normal(size=(N,M))
# w_true = np.random.normal(size=M)
# y_data = np.random.binomial(1, spsp.expit(X_data.dot(w_true))).reshape(-1,1)

iteration = 0
perm = np.random.permutation(N)

for t in range(T):
    for iter in range(math.ceil(N / 5000)):
        batch = [i % N for i in range(iter * 5000, (iter + 1) * 5000)]
        ridx = perm[batch]
        iteration += 1

        Xs = X_train[ridx, :]
        Ys = y_train[ridx]
        grad = optimizer.compute_gradients(log_p_d_given_z, [mu, sigma0])
        update = optimizer.apply_gradients(grad)
        # update = optimizer.minimize(sess.run(loss, {X: Xs, y: Ys}), [mu, sigma0])
        sess.run(update, {X: Xs, y: Ys})
        # print(sess.run(log_p_d_given_z, {X:X_data,y:y_data}))
        if iteration % 10 == 0:
            print("epoch:{} iteration:{}, loss:{}".format(t + 1, iteration, sess.run(log_p_d_given_z, {X: Xs, y: Ys})))
            acc, acc_op = tf.compat.v1.metrics.accuracy(labels=y,
                                              predictions=tf.argmax(tf.sigmoid(tf.matmul(X, tf.reshape(z, [-1, 1]))),
                                                                    1))
            sess.run(tf.local_variables_initializer())
            accuracy = (sess.run(acc_op, {X: X_test, y: y_test}))
            print(accuracy)
# # print(w_true)
print(sess.run(mu))
print(sess.run(sigma))
