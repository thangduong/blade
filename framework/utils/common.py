import copy
import os

def get_dict_value(dict_var, key, default_value=None, add_if_not_in_map=True):
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
        result = dict_var[key]
    else:
        result = default_value
        if add_if_not_in_map:
            dict_var[key] = result
    return result
    
def extend_dict(old_dict, new_dict):
    """
      Extend old_dict with entries in new_dict with keys not already in old_dict
    @param old_dict: dict to extend
    @param new_dict: dict with new entries to add
    @return:
      A completely new dict that is an extended version of old_dict with new_dict
    """
    if old_dict and new_dict:
        result_dict = copy.copy(new_dict)
        result_dict.update(old_dict)
        return result_dict
    elif old_dict:
        return copy.copy(old_dict)
    else:
        return {}
    
def load_param_file(param_file, default_params=None, variable_name='params'):
    """
    Load parameters from file and extend with default_params
    @param param_file:
    @param default_params:
    @return:
    """
    param_file_ext = os.path.splitext(param_file)[1].lower()
    if (os.path.isfile(param_file) and param_file_ext=='.py'):
        vars = {}
        with open(param_file) as f:
            exec (f.read(), vars)
        return extend_dict(get_dict_value(vars,variable_name), default_params)
    elif default_params:
        return copy(default_params)
    else:
        return {}

def fix_param_file_path(filepath, base_dir): #=os.path.realpath(__file__)):

#    print('[%s,%s,%s]'% (filepath,os.path.dirname(base_dir),os.path.dirname(filepath)))
    if os.path.dirname(filepath) == '':
        filepath = os.path.join(os.path.dirname(base_dir), filepath)
    return filepath
