import tensorflow as tf
import numpy as np


def l21_norm(W):
    return tf.reduce_sum(tf.norm(W))


def get_group_regularization(w_not_bias):
    # mlp_model is the neural network model being trained

    def const_coeff(W):
        return tf.sqrt(tf.cast(tf.size(W), tf.float32))
    return tf.reduce_sum([tf.multiply(const_coeff(W), l21_norm(W)) for W in w_not_bias])


def get_L1_norm(w_bias):
    variables = [tf.reshape(v, [-1]) for v in w_bias]
    variables = tf.concat(variables, axis=0)
    return tf.norm(variables, ord=1)


def sparse_group_lasso(weights):
    w_not_bias = []
    w_bias = []
    for W in weights.keys():
        curr = weights[W].detach().numpy()
        curr = tf.convert_to_tensor(curr)
        w_bias.append(curr)
        if 'bias' not in W:
            w_not_bias.append(curr)
    grouplasso = get_group_regularization(w_not_bias)  # group lasso function
    l1 = get_L1_norm(w_bias)  # l1 function
    # sparse group lasso function (group lasso + l1)
    sparse_lasso = grouplasso + l1
    return sparse_lasso


"""import tensorflow as tf


def l21_norm(W):
    return tf.reduce_sum(tf.norm(W, axis=1))


def get_group_regularization(mlp_model):
    # mlp_model is the neural network model being trained
    def const_coeff(W): return tf.sqrt(
        tf.cast(W.get_shape().as_list()[1], tf.float32))
    return tf.reduce_sum([tf.multiply(const_coeff(W), l21_norm(W)) for W in mlp_model.trainable_variables if 'bias' not in W.name])


def get_L1_norm(mlp_model):
    variables = [tf.reshape(v, [-1]) for v in mlp_model.trainable_variables]
    variables = tf.concat(variables, axis=0)
    return tf.norm(variables, ord=1)


def sparse_group_lasso(mlp_model):
    grouplasso = get_group_regularization(mlp_model)  # group lasso function
    l1 = get_L1_norm(mlp_model)  # l1 function
    # sparse group lasso function (group lasso + l1)
    sparse_lasso = grouplasso + l1
    return sparse_lasso
"""
