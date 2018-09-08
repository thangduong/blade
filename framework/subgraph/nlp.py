import tensorflow as tf
import framework.subgraph.core as core

def dump_variables():
  total_size = 0
  for var in tf.trainable_variables():
    var_size = np.prod(var.get_shape().as_list())
    print("%s : %s parameters, %s mb" % (var.name, str(var_size), str(var_size / (1024 * 1024))))
    total_size += var_size


def expand_list(value, list_to_match):
  if not (type(value) is list):
    result = [value] * len(list_to_match)
  else:
    result = value
  return result


def variable_with_weight_decay(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.05), wd=0.0, vlist=None):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  if vlist==None:
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
  else:
    var = vlist.pop(0)
  return var


def conv2d_array(input, sizes, widths, heights, keep_probs=None, w_wds=0.005, b_wds=0.000,
               w_initializers=tf.truncated_normal_initializer(stddev=0.05),
               b_initializers=tf.truncated_normal_initializer(stddev=0.05),
               name="conv2d_array",
               vlistin = None, vlistout = None):
  w_wd_list = expand_list(w_wds, sizes)
  b_wd_list = expand_list(b_wds, sizes)
  keep_prob_list = expand_list(keep_probs, sizes)
  w_initializer_list = expand_list(w_initializers, sizes)
  b_initializer_list = expand_list(b_initializers, sizes)
  input_size = input.get_shape().as_list()[-1]
  print('embedding_dimension = ' + str(input_size))
  with tf.variable_scope(name):
    for i, (width, height, size, w_wd, b_wd, keep_prob, w_initializer, b_initializer) in enumerate(
            zip(widths, heights, sizes, w_wd_list, b_wd_list, keep_prob_list, w_initializer_list, b_initializer_list)):
      with tf.variable_scope('conv%s' % str(i)) as scope:
        w = variable_with_weight_decay('w', [width, height, input_size, size],
                                       initializer=w_initializer, wd=w_wd,vlist=vlistin)
        if vlistout is not None:
          vlistout.append(w)
        b = variable_with_weight_decay('b', [size],
                                       initializer=b_initializer, wd=b_wd,vlist=vlistin)
        if vlistout is not None:
          vlistout.append(b)
        w_out = tf.nn.conv2d(input, filter=w, strides=[1,1,1,1], padding='SAME')
        b_out = tf.nn.bias_add(w_out, b)
        out = tf.nn.relu(b_out)
        if keep_prob is not None and keep_prob < 1.0:
          [out], _ = core.dropout([out], keep_prob)
        input_size = size
        input = out
  return out


def conv1d_array(input, sizes, widths, strides=None,  keep_probs=None, w_wds=0.005, b_wds=0.000,
               w_initializers=tf.truncated_normal_initializer(stddev=0.05),
               b_initializers=tf.truncated_normal_initializer(stddev=0.05),
               name="conv1d_array",
               vlistin = None, vlistout = None):
  w_wd_list = expand_list(w_wds, sizes)
  b_wd_list = expand_list(b_wds, sizes)
  keep_prob_list = expand_list(keep_probs, sizes)
  w_initializer_list = expand_list(w_initializers, sizes)
  b_initializer_list = expand_list(b_initializers, sizes)
  input_size = input.get_shape().as_list()[-1]
  if strides is None:
    strides = 1
  strides = expand_list(strides,sizes)
#  print('conv1d_array: embedding_dimension = ' + str(input_size))
  out = input
  with tf.variable_scope(name):
    for i, (width, size, w_wd, b_wd, keep_prob, w_initializer, b_initializer, stride) in enumerate(
            zip(widths, sizes, w_wd_list, b_wd_list, keep_prob_list, w_initializer_list, b_initializer_list, strides)):
      with tf.variable_scope('conv%s' % str(i)) as scope:
        w = variable_with_weight_decay('w', [width, input_size, size],
                                       initializer=w_initializer, wd=w_wd,vlist=vlistin)
        if vlistout is not None:
          vlistout.append(w)
        b = variable_with_weight_decay('b', [size],
                                       initializer=b_initializer, wd=b_wd,vlist=vlistin)
        if vlistout is not None:
          vlistout.append(b)
        print(w)
        w_out = tf.nn.conv1d(input, w, stride, 'SAME')
        b_out = tf.nn.bias_add(w_out, b)
        out = tf.nn.relu(b_out)
        if keep_prob is not None and keep_prob < 1.0:
          [out],_ = core.dropout([out], keep_prob)
        input_size = size
        input = out
  return out

def conv1d(input, sizes, widths, strides=None, keep_probs=None, w_wds=0.005, b_wds=0.000,
               w_initializers=tf.truncated_normal_initializer(stddev=0.05),
               b_initializers=tf.truncated_normal_initializer(stddev=0.05),
               name="conv1d",
               vlistin = None, vlistout = None):
  w_wd_list = expand_list(w_wds, sizes)
  b_wd_list = expand_list(b_wds, sizes)
  keep_prob_list = expand_list(keep_probs, sizes)
  w_initializer_list = expand_list(w_initializers, sizes)
  b_initializer_list = expand_list(b_initializers, sizes)
  input_size = input.get_shape().as_list()[-1]
  if strides is None:
    strides = 1
  strides = expand_list(strides,sizes)
  print('conv1d: embedding_dimension = ' + str(input_size))
  outlist = []
  with tf.variable_scope(name):
    for i, (width, size, w_wd, b_wd, keep_prob, w_initializer, b_initializer, stride) in enumerate(
            zip(widths, sizes, w_wd_list, b_wd_list, keep_prob_list, w_initializer_list, b_initializer_list, strides)):
      with tf.variable_scope('conv%s' % str(i)) as scope:
        w = variable_with_weight_decay('w', [width, input_size, size],
                                       initializer=w_initializer, wd=w_wd,vlist=vlistin)
        if vlistout is not None:
          vlistout.append(w)
        b = variable_with_weight_decay('b', [size],
                                       initializer=b_initializer, wd=b_wd,vlist=vlistin)
        if vlistout is not None:
          vlistout.append(b)
        print(w)
        print(stride)
        w_out = tf.nn.conv1d(input, w, stride, 'SAME')
        b_out = tf.nn.bias_add(w_out, b)
        out = tf.nn.relu(b_out)
        if keep_prob is not None and keep_prob < 1.0:
          [out],_ = core.dropout([out], keep_prob)
        outlist.append(out)
  return outlist

def gated_linear_unit(input):
	None