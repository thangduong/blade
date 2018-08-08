import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os

class Evaluator(object):
    """
    A general tensorflow network evaluator

    Example usage:

    from framework.model import Evaluator

    multiply_model = Evaluator.load("model_graph.pb", "model_variables.ckpt")
    [five_times_10] = multiply_model.eval({'x':5,'y':10},['product'])

    """

    def __init__(self, tf_session, graph_prefix =""):
        self._tf_session = tf_session
        self._graph_prefix = graph_prefix

    @classmethod
    def load2(cls, ckpt_filepath):
        tf_graph = tf.Graph()
        tf_session = tf.Session(graph=tf_graph)
        with tf_graph.as_default():
            with tf_session.as_default():
                saver = tf.train.import_meta_graph(ckpt_filepath + '.meta')
                saver.restore(tf_session, ckpt_filepath)
        return cls(tf_session)

    @classmethod
    def load_graphdef(cls, graphdef_filename):
        tf_graph = tf.Graph()
        tf_session = tf.Session(graph=tf_graph)
        model_name = os.path.splitext(os.path.basename(graphdef_filename))[0]
        with tf_graph.as_default():
            with tf_session.as_default():
                with tf.gfile.GFile(graphdef_filename, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name=model_name)
        return cls(tf_session, graph_prefix="%s/"%model_name)

    @classmethod
    def load(cls, pb_filename, ckpt_filename=None, model_dir=None):
        """
        Load a model into an evaluator and return the evalutor

        Parameters
        ----------
        pb_filename : str

        ckpt_filename : str

        """
        if isinstance(model_dir,str):
            pb_filename = '%s/%s' % (model_dir, pb_filename)
            if not (ckpt_filename == None):
                ckpt_filename = '%s/%s' % (model_dir, ckpt_filename)
        if (ckpt_filename == None):
            tf_graph = tf.Graph()
            tf_session = tf.Session(graph=tf_graph)
            saver = tf.train.Saver()
            saver.restore(tf_session, pb_filename)
        else:
            tf_graph = tf.Graph()
            tf_session = tf.Session(graph=tf_graph)
            with tf_graph.as_default():
                with tf_session.as_default():
                    with gfile.FastGFile(pb_filename,'rb') as f:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())
                        result = tf.import_graph_def(graph_def, name='')
                    before_list = [v.name for v in tf.global_variables()]
                    variables_to_restore = []
                    for n in tf_session.graph.as_graph_def().node:
                        if n.op == 'VariableV2':
    #                        print(n)
                            current_variable = tf_session.graph.get_tensor_by_name(n.name + ':0')
                            if not(current_variable.name in before_list):
                                variables_to_restore.append(current_variable)
                                tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES,current_variable)
                    saver = tf.train.Saver(variables_to_restore) # 'Saver' misnomer! Better: Persister!
                    saver.restore(tf_session, ckpt_filename)  # now OK
        return cls(tf_session)

    @classmethod
    def restore_variables(cls, inference, ckpt_pathname, params=None):
        """
        Create the model using the inference function given in the parameter and restore variables into that
        model.
        @param inference: inference function of the model
        @param ckpt_pathname: full path name of check point file
        @param params: parameters variable to be passed into the inference function
        @return:
        the evaluator
        """

        tf_graph = tf.Graph()
        tf_session = tf.Session(graph=tf_graph)
        with tf_graph.as_default():
            with tf_session.as_default():
                inference()
                before_list = [v.name for v in tf.all_variables()]
                variables_to_restore = []
                for n in tf_session.graph.as_graph_def().node:
                    if n.op == 'Variable':
                        current_variable = tf_session.graph.get_tensor_by_name(n.name + ':0')
                        if not (current_variable.name in before_list):
                            variables_to_restore.append(current_variable)
                            tf.add_to_collection(tf.GraphKeys.VARIABLES, current_variable)
                saver = tf.train.Saver(variables_to_restore)  # 'Saver' misnomer! Better: Persister!
                saver.restore(tf_session, ckpt_pathname)  # now OK
        return cls(tf_session)

    def get_inputs(self):
        """
        """
        result = []
        for n in self._tf_session.graph.as_graph_def().node:
            if (n.op == 'Placeholder'):
                result.append({'name':n.name,
                               'dtype':n.attr['dtype'].type,
                               'shape': n.attr['shape'].shape})
        return result
    
    def get_outputs(self):
        name_list = []
        for n in self._tf_session.graph.as_graph_def().node:
            name_list.append(n.name)
#            print([n.op, n.name])
        for n in self._tf_session.graph.as_graph_def().node:
            for i in n.input:
                if i in name_list:
                    name_list.remove(i)
        return name_list

    def dump_variable_sizes(self):
        """
        Dump size of variables
        @return: None
        """
        with self._tf_session.graph.as_default():
            with self._tf_session.as_default():
                total_mp_size = 0
                trainable_variable_names = [v.name for v in tf.trainable_variables()]
                for variable in tf.global_variables():
                    print(variable.get_shape())
                    mb_size = np.prod([float(i) for i in variable.get_shape().as_list()]) / (1024)
                    total_mp_size += mb_size
                    output_row = [variable.name, str(variable.get_shape()), "{0:.2f}".format(mb_size),
                                  "{0:.2f}".format(total_mp_size), str(variable.name in trainable_variable_names)]
                    col_width = [20, 20, 10, 10, 10]
                    print(" ".join(col.rjust(col_width[i]) for i, col in enumerate(output_row)))

    def get_variable(self, name):
        v = None
        with self._tf_session.graph.as_default():
            with self._tf_session.as_default():
                vlist = [v for v in tf.global_variables() if v.name==name or v.name==('%s:0'%name)]
                if len(vlist)>0:
                    v = vlist[0]
        return v
    
    def get_variable_value(self, name):
        result = None
        with self._tf_session.as_default():
            v = self.get_variable(name)
            if not (v==None):
                result = v.eval()
        return result

    def dump_graph(self):
        """
        Dump graph for debugging purposes.
        @return:
        """
        for node in self._tf_session.graph.as_graph_def().node:
            print('%s (%s)' % (node.name, node.op))

    def save_graph_as_pbtxt(self, filename):
        tf.train.write_graph(self._tf_session.graph.as_graph_def(), '.',filename,as_text=True)

    def eval(self, inputs, output_names):
        """
        Evaluate a node in the model

        Parameters
        ----------
        inputs: dict str -> type

        output_names: array of str

        """

        if (isinstance(output_names, str)):
            output_names = [output_names]

        # find the input placeholder nodes and build feed_dict
        feed_dict = {self._tf_session.graph.get_tensor_by_name(self._graph_prefix+'is_training:0'):False}
        for input_name, input_value in inputs.items():
            cur_input = self._tf_session.graph.get_tensor_by_name(self._graph_prefix + input_name + ':0')
            feed_dict[cur_input] = input_value

        # find the output nodes
        output_tensors = []
        for name in output_names:
            output_tensor = self._tf_session.graph.get_tensor_by_name(self._graph_prefix + name+':0')
            output_tensors.append(output_tensor)

#        print(output_tensors)
#        print(feed_dict)
        result = (self._tf_session.run(output_tensors, feed_dict))
        return result
