"""
Sample MLP trainer

Training and evaluation are demonstrated in this file.
"""
from framework.model.evaluator import Evaluator
import framework.utils.data.training_data as training_data
from framework.subgraph.core import input_data, rename_nodes, fully_connected_network
from framework.trainer import Trainer

MODEL_NAME = 'mlp1'
PB_FILENAME = 'mlp1.pb'
CKPT_FILENAME = 'mlp1.ckpt'
OUTPUT_DIR = '/tmp'

def inference():
    """
    This function defines the TF computational model network
    @return:
    network_output_nodes as an array of tf nodes
    """""
    x, _ = input_data([None, 1], 'x')
    fcn, _ = fully_connected_network(x, [10,10,1], dropout_keep_probs=.5)
#  expand, _   = fully_connected_layer(x, 10, name='expand')
#  contract, _ = fully_connected_layer(expand, 1, name='contract')
    ybar, _ = rename_nodes(fcn, ['ybar'])
    return ybar
  
def train_iteration_done(trainer, iteration, loss_value, done):
    # debug dump of the variables as we train
    if iteration % 1000 == 0:
#    print([v.name for v in tf.all_variables()])
        print([iteration, loss_value])
        if (loss_value < 0.15):
            done = True
    return done
  
  
# training code:
#   1 - generate some fake data,
#   2 - create a trainer with the data,
#   3 - run the trainer,
#   4 - and save the model
fake_data = training_data.generate_fake_1d_training_data(['x', 'y0'])
trainer = Trainer(inference=inference, batch_size=128, model_output_location=OUTPUT_DIR,
                  name=MODEL_NAME, training_data=fake_data, train_iteration_done=train_iteration_done)
trainer.run(restore_latest_ckpt=False, save_network=True)
trainer.save(output_dir=OUTPUT_DIR,pb_filename=PB_FILENAME,ckpt_filename=CKPT_FILENAME)
print("Training done")


# test evaluation code
e = Evaluator.load(model_dir=OUTPUT_DIR,pb_filename=PB_FILENAME,ckpt_filename=CKPT_FILENAME)
print(e.eval({'x': 10}, 'ybar'))
