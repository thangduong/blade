"""
Sample code using the framework to fit a polynomial to data


The model itself is defined by the "inference" function.  The polynomial subgraph simply
builds a subnetwork that computes a polynomial.

Training and evaluation are demonstrated in this file.
"""
import tensorflow as tf
from framework.evaluator import Evaluator
import framework.utils.data.training_data as training_data
from framework.subgraph import subgraph_helper as sghelper
from framework.subgraph.core import input_data, rename_nodes
from framework.trainer import Trainer
import logging

MODEL_NAME = 'polynomial'
PB_FILENAME = 'polynomial.pb'
CKPT_FILENAME = 'polynomial.ckpt'
OUTPUT_DIR = './%s' % MODEL_NAME


def polynomial(in_nodes, order, in_vars=None, name='polynomial'):
    """
    A subgraph to build a network that computes a polynomial function of the input

    Only in_nodes[0] is used.  This subgraph computes:
        <name>/output = a0 + a1 * in_nodes[0] + a1 * in_nodes[0]^2 + ...
    where a_j are scalar variables.

    @param in_nodes:

    @param order:
    @param in_vars:
    @param name:
    @return:
    """
    with tf.variable_scope(name):
        out_vars = {}
        a0 = sghelper.get_variable('a0', [], out_vars, in_vars=in_vars)
        terms = [tf.scalar_mul(a0, tf.ones_like(in_nodes[0]))]
        for i in range(1,order+1):
            terms.append(tf.scalar_mul(sghelper.get_variable('a%d'%i, [], out_vars, in_vars=in_vars),
                                       tf.pow(in_nodes[0], tf.constant(float(i)))))
        return [tf.add_n(terms, name='output')], out_vars

def inference(params=None):
    """
    This function defines the TF computational model network
    @return:
    network_output_nodes as an array of tf nodes
    """""
    x, _ = input_data([None, 1], 'x')
    poly, _ = polynomial(x, 2, name='p')
    poly, _ = rename_nodes(poly, ['ybar'])
    return poly

def train_iteration_done(trainer, iteration, loss_value, done, run_results):
    # debug dump of the variables as we train
    if iteration % 100 == 0:
        a0 = [v for v in tf.all_variables() if v.name == 'p/a0:0']
        a1 = [v for v in tf.all_variables() if v.name == 'p/a1:0']
        a2 = [v for v in tf.all_variables() if v.name == 'p/a2:0']
        logging.info([trainer._training_data.current_epoch(), iteration,
                      loss_value, a0[0].eval(), a1[0].eval(), a2[0].eval()])
#
#       example of how to exit on condition:
#        if (loss_value < 1):
#            done = True
    return done

logging.basicConfig(level=logging.INFO)

# training code: generate some fake data, create a trainer with the data, run the trainer, and save the model
fake_data = training_data.generate_fake_1d_training_data(['x', 'y0'])
trainer = Trainer(inference=inference, model_output_location=OUTPUT_DIR,
                  name='quadratic', training_data=fake_data, train_iteration_done=train_iteration_done,batch_size=16)
trainer.run(num_epochs=5000,restore_latest_ckpt=False, save_network=False)
trainer.save(output_dir=OUTPUT_DIR,pb_filename=PB_FILENAME,ckpt_filename=CKPT_FILENAME)

# test evaluation code
e = Evaluator.load(model_dir=OUTPUT_DIR,pb_filename=PB_FILENAME,ckpt_filename=CKPT_FILENAME)
logging.info(e.eval({'x': 10}, 'ybar'))
