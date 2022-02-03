import tensorflow as tf 

def l21_norm(W): 
    return tf.reduce_sum(tf.norm(W, axis=1))

def get_group_regularization(mlp_model):
	# mlp_model is the neural network model being trained
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1] ,tf.float32))
    return tf.reduce_sum([tf.multiply(const_coeff(W), l21_norm(W)) for W in mlp_model.trainable_variables if 'bias' not in W.name])

def get_L1_norm(mlp_model):
    variables = [tf.reshape(v ,[-1]) for v in mlp_model.trainable_variables]
    variables = tf.concat(variables, axis= 0)
    return tf.norm(variables, ord = 1)

def sparse_group_lasso(mlp_model):
    grouplasso = get_group_regularization(mlp_model) #group lasso function 
    l1 = get_L1_norm(mlp_model) # l1 function 
    sparse_lasso = grouplasso + l1 #sparse group lasso function (group lasso + l1)
    return sparse_lasso
