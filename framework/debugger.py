from framework.model.evaluator import Evaluator
import scipy.misc
import os

class Debugger(object):
    """
    A debugger that attaches to an evaluator and allows for inspection of
    variables.
    """
    def __init__(self, evaluator, output_dir='./'):
        self._evaluator = evaluator
        self._current_inputs = None
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @classmethod
    def load(cls, pbtxt_filename, checkpoint_filename, output_dir='./'):
        return cls(Evaluator.load(pbtxt_filename, checkpoint_filename), output_dir=output_dir)

    def set_inputs(self, inputs):
        self._current_inputs = inputs

    def get_variable_names(self):
        variable_names = []
        for n in self._evaluator._tf_session.graph.as_graph_def().node:
            if (n.op == 'Variable'):
                current_variable_name = n.name + ':0'
                variable_names.append(current_variable_name)
        return variable_names

    def get_variables(self, variable_names):
        variables = []
        for current_variable_name in names:
            current_variable = self._evaluator._tf_session.graph.get_tensor_by_name(current_variable_name)
            variables.append(current_variable)
        return variables

    def eval(self, output_tensor_name):
        [result] = self._evaluator.eval(self._current_inputs, [output_tensor_name])
        return result

    def dump_ndarray_as_images(self, name, ndarray, rgb=False):
        """
        Dump a tensor as images
        """
        dir_name = self._output_dir + '/' + name
        os.makedirs(dir_name, exist_ok=True)

        # if rgb
        # <minibatch><x><y><rgb><filter_count>
        # else
        # <minibatch><x><y><filter_count>

        shape = ndarray.shape
        for elt_index in range(0,shape[0]):
            for filter_index in range(0,shape[-1]):
                # there probably is a quicker more elegant way to do this
                if rgb:
                    image_array = ndarray[elt_index,:,:,:,filter_index]
                else:
                    image_array = ndarray[elt_index,:,:,filter_index]
                out_filename = dir_name + '/' + str(elt_index) + '_' + str(filter_index) + '.png'
                print('writing to ' + out_filename)
                scipy.misc.toimage(image_array, cmin=-1, cmax=1).save(out_filename)

    def dump_tensor_as_images(self, tensor, rgb=False):
        None
