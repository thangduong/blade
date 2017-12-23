"""

"""
from framework.evaluator import Evaluator
from framework.utils.data.text_indexer import TextIndexer

class TextEvaluator(object):
    def __init__(self, pb_filename, ckpt_filename, vocab_filename, inputs_to_index=None, model_dir=None):
        """

        @param pb_filename:
        @param ckpt_filename:
        @param vocab_filename:
        @param inputs_to_index:
        @param model_dir:
        @param fix_dropout_param:
        """
        print("Loading: %s %s" % (model_dir, pb_filename))
        self._evaluator = Evaluator.load(pb_filename, ckpt_filename, model_dir)
#        self._evaluator.dump_graph()
        self._inputs_to_index = inputs_to_index
        self._indexer = TextIndexer

    def eval(self, inputs, output_names):
        """
        Index the appropriate inputs and forward the call to Evaluator.eval
        @param inputs:
        @param output_names:
        @return:
        """
        indexed_inputs = {}
        for input_name, input_value in inputs.items():
            if isinstance(input_value, str) \
                    and (self._inputs_to_index is None or input_name in self._inputs_to_index) \
                    and not(self._indexer is None):
                indexed_inputs[input_name] = self._indexer.index_text(input_value)
            else:
                indexed_inputs[input_name] = input_value
        return self._evaluator.eval(indexed_inputs, output_names)


class TextModel(TextEvaluator):
    def __init__(self, pb_filename, ckpt_filename, vocab_filename, input_names, output_names, inputs_to_index=None, model_dir=None):
        """
        @param pb_filename:
        @param ckpt_filename:
        @param vocab_filename:
        @param input_names:
        @param output_names:
        @param inputs_to_index:
        @param model_dir:
        """
        super(TextModel, self).__init__(pb_filename, ckpt_filename, vocab_filename, inputs_to_index, model_dir)
        self._input_names = input_names
        self._output_names = output_names

    def eval(self, *args):
        """
        Evaluate.
        @param args: Number of arguments must be equal to the size of input_names
        @return:
        """
        inputs = {}
        for name, value in zip(self._input_names, args):
            inputs[name] = value
        return super(TextModel, self).eval(inputs, self._output_names)
