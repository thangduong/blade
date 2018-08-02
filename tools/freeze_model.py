import framework.utils.common as utils
import sys
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

paramsfile = sys.argv[1]
params = utils.load_param_file(paramsfile)
dirname = os.path.dirname(paramsfile)
timestamp = time.time()
model_name = utils.get_dict_value(params, 'model_name')
output_graphdef_filename = os.path.join(dirname,model_name + ".graphdef")
inference_output_node = utils.get_dict_value(params,'inference_output_node', 'sm_decision')
freeze_graph = os.path.join(os.path.dirname(tf.__file__), 'python/tools','freeze_graph.py')
ckpt = os.path.join(utils.get_dict_value(params, 'output_location'),
										model_name + '.ckpt')
pbf = os.path.join(utils.get_dict_value(params, 'output_location'),
									 model_name + '.training.pb')
cmd = 'python3 %s --help' % freeze_graph

cmd = 'python3 %s --input_binary True --clear_devices True --input_graph %s ' \
		'--input_checkpoint %s --output_node_names %s --output_graph %s' % \
			(freeze_graph, pbf, ckpt, inference_output_node, output_graphdef_filename)
os.system(cmd)

release_timestamp_filename = os.path.join(utils.get_dict_value(params, 'output_location'),
									 'release.timestamp.txt')
with open(release_timestamp_filename,'w') as f:
	f.write('%s'%timestamp)

print("WROTE %s"%output_graphdef_filename)