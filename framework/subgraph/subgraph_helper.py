import tensorflow as tf
import copy

def get_dict_value(dict_var, key, default_value=None):
    """
    This is like dict.get function except it checks that the dict_var is a dict
    in addition to dict.get.
    @param dict_var: the variable that is either a dict or something else
    @param key: key to look up in dict
    @param default_value: return value if dict_var is not of type dict or key is not in dict_var
    @return:
      either default_value or dict_var[key]
    """
    if (isinstance(dict_var, dict) and key in dict_var):
        return dict_var[key]
    else:
        return default_value

def get_variable(name, shape, out_vars=None, in_vars=None):
    '''
    Get or create a variable.  This simply wraps up Tensorflow's get_variable.

    Returns the tf variable.  if out_vars is a dict, then out_vars[name] also contains
    the tf variable.  if in_vars is a dict and in_vars[name] is a tf var, then the return value
    is in_vars[name].

    @param name : name of variable
    @param shape : shape of variable
    @param out_vars : dict of str -> tf_variable
    @param in_vars : dict of str -> tf variable
    '''
    result_var = None
    if (isinstance(in_vars, dict) and name in in_vars):
        result_var = in_vars[name]
    else: # note: default to normal as a hack for now
        result_var = tf.get_variable(name=name, initializer=tf.random_normal(shape, stddev=0.35))
    if isinstance(out_vars, dict):
        out_vars[name] = result_var
    return result_var
