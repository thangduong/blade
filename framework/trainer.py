"""
Basic trainer class that encapsolates training

"""
import tensorflow as tf
import os
import math
import copy
import framework.subgraph.core as core
import framework.utils.common as common_utils
import logging
import framework.evaluator
import framework.utils.common as utils
import framework.db.model_log_db as db
from time import time
import collections
import numpy as np

LEARNING_RATE_PARAM_NAME = 'learning_rate'
DEFAULT_LEARNING_RATE = 0.001
IS_TRAINING_PLACEHOLDER_NAME = 'is_training'
CHECKPOINT_SUBDIRECTORY = '/ckpt/'
CHECKPOINT_FILE_EXTENSION = '.ckpt'
MODEL_GRAPH_FILE_EXTENSION = '.pb'
ENABLE_REGULARIZATION_PARAM_NAME = "enable_regularization"

default_params = {
  IS_TRAINING_PLACEHOLDER_NAME: DEFAULT_LEARNING_RATE
}

def _default_train_iteration_done(trainer, epoch, index, iteration_count,
								  loss_value, training_done, run_results, params):

	stats = params['stats']
	next_batch_time = np.mean(stats['next_batch_time_list'])
	training_time = np.mean(stats['training_time_list'])
	overhead_time = np.mean(stats['overhead_time_list'])

	if iteration_count == 1:
			trainer._training_log_file = open(os.path.join(utils.get_dict_value(params, 'output_location'), 'training_log.txt'), 'w')


	msg = ("%02d, %04d, %s, %s, %0.4f, %0.5f, %0.5f, %0.5f" %
				 (epoch, iteration_count, time(), loss_value, next_batch_time,
					training_time, overhead_time,
					training_time / sum([next_batch_time, training_time, overhead_time])))
	if "eval_results" in params:
		eval_results = params['eval_results']
		for x in eval_results:
			msg += ", %0.4f"%x

	print('%s' % msg)
	trainer._training_log_file.write('%s\n' % msg)
	trainer._training_log_file.flush()

	if self._model_log_db is not None:
		self._model_log_db.on_update(loss_vale, msg)
	return False

class Trainer(object):
	"""
	A trainer class that takes 3 methods: inference, and train.
	Inference builds the network.  Inputs reads the input data.  Train
	updates the weights.

	Gating a sub network based on is_training using condition:
	  x, _ = input_data([None, 1], 'x')
	  is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
	  poly, _ = polynomial(x, 2, name='p')
	  poly = [tf.cond(is_training, lambda: poly[0], lambda: x)]
	  poly, _ = rename_nodes(poly, ['ybar'])
	  return poly
	"""
	def __init__(self, inference, training_data, loss=None,
				 batch_size=128, data_map=None, name='model',
				 model_output_location='/tmp', optimizer=None,
				 train_iteration=None, train_iteration_done=_default_train_iteration_done,
				 params=None):
		"""
		Constructor for the trainer class

		@type inference: function
		@param inference: The function that builds a network.  This function has no argument and returns a list of
			output nodes of the network.
		@param training_data:
		@param loss:
		@param batch_size:
		@param data_map:
		@param name:
		@param model_output_location:
		@param optimizer:
		@param train_iteration:
		@param train_iteration_done:
		@param optimizer_params:
		"""
		self._model_log_db = None
		self._model_name = name
		self._model_output_location = model_output_location
		self._model_ckpt_filename = os.path.join(self._model_output_location, self._model_name + CHECKPOINT_FILE_EXTENSION)
		self._inference = inference
		self._loss = loss
		self._batch_size = batch_size
		self._data_map = data_map
		self._training_data = training_data
		self._params = common_utils.extend_dict(params, default_params)
		self._train_iteration_done = train_iteration_done
		self._model_graph_def = None
		if loss:
			self._loss = loss
		else:
			self._loss = Trainer._l2_diff_loss
		if optimizer:
			self._optimizer = optimizer
		else:
			self._optimizer = Trainer._adam_optimizer
		if train_iteration:
			self._train_iteration = train_iteration
		else:
			self._train_iteration = Trainer._default_train_iteration

		# create session and build graph
		self._graph = tf.Graph()
		self._session = tf.Session(graph=self._graph)
		with self._graph.as_default():
			self._is_training = core.input_data([], IS_TRAINING_PLACEHOLDER_NAME, dtype=tf.bool)
			self._network_output_nodes = self._inference(self._params)
			self._learning_rate = tf.Variable(common_utils.get_dict_value(self._params, LEARNING_RATE_PARAM_NAME,
														DEFAULT_LEARNING_RATE), trainable=False)
			self._model_graph_def = copy.copy(self._graph.as_graph_def())

			# loss is an array of loss functions
			self._loss_nodes = self._loss(self._network_output_nodes, params=self._params)

			# build optimizer
			self._optimizer_nodes = self._optimizer(self._params, self._loss_nodes, self._learning_rate)

	def get_evaluator(self):
		return framework.evaluator.Evaluator(self._session)

	def get_network_output_nodes(self):
		return self._network_output_nodes

	def get_learning_rate(self):
		return self._learning_rate.eval()

	def set_learning_rate(self, new_learning_rate):
		self._session.run(self._learning_rate.assign(new_learning_rate))

	def decay_learning_rate(self, factor):
		self.set_learning_rate(self.get_learning_rate()*factor)
		return self.get_learning_rate()

	def get_nodes_by_name(self, node_names):
		result_nodes = []
		for node_name in node_names:
			node = self._session.graph.get_tensor_by_name(node_name+':0')
			if node is not None:
				result_nodes.append(node)
		return result_nodes

	def run(self, num_iterations=None,  num_epochs=None, restore_latest_ckpt=True,
			save_ckpt=True, mini_batches_between_checkpoint=1000, save_network=True,
			additional_nodes_to_evaluate=None, on_checkpoint_saved=None,
			mini_batches_between_sanity_check=25,
			log_to_db=True):
		"""
		Run the training loop.

		@param num_iterations: number of iterations to run the training.
				either num_iteration or num_epoch or neither are set, but not both.
		@param num_epochs: number of epochs to run the training.
				either num_iteration or num_epoch or neither are set, but not both.
		@param restore_latest_ckpt:  If true, restore the checkpoint file before resuming training.
		@param save_ckpt: If true, save checkpoint files every so often.
		@param mini_batches_between_checkpoint:  Number of minibatches processed before a checkpoint file is saved.
		@param save_network: If true, save the network as a pb file before training.
		@return:
		 None
		"""
		if log_to_db:
			self._model_log_db = db.ModelLogDb()
			self._model_log_db_id = self._model_log_db.begin_training(self._model_name, self._model_output_location)
			with open(os.path.join(self._model_output_location,
																							 'model_log_db_id.txt'),'w') as f:
				f.write("%s\n"%self._model_log_db_id)

		self._params['stats'] = {'next_batch_time_list': collections.deque(maxlen=10),
								 'training_time_list': collections.deque(maxlen=10),
								 'overhead_time_list': collections.deque(maxlen=10) }

		with self._graph.as_default():
			# save the inference and training network, too
			if save_network:
				tf.train.write_graph(self._model_graph_def, self._model_output_location, self._model_name + '.pb', False)
				tf.train.write_graph(self._graph.as_graph_def(), self._model_output_location, self._model_name + '.training.pb',
									 False)
			# build a data_map that maps from field name to placeholder
			data_map = {IS_TRAINING_PLACEHOLDER_NAME:self._is_training}
			for n in self._graph.as_graph_def().node:
				if (n.op == 'Placeholder'):
					placeholder_tensor = self._graph.get_tensor_by_name(n.name + ':0')
					input_name = n.name
					if isinstance(self._data_map, dict) and input_name in self._data_map:
						input_name = self._data_map[input_name]
					data_map[input_name] = placeholder_tensor

			self._current_datamap = data_map

			iteration_count = 0
			ckpt_dir = self._model_output_location + CHECKPOINT_SUBDIRECTORY
			if not os.path.exists(ckpt_dir):
				os.makedirs(ckpt_dir)

			with self._session.as_default():
				tf.global_variables_initializer().run()
				# if there's a checkpoint file, then restore it so we can re-start training
				# where it was left off.

#				coord = tf.train.Coordinator()
				# this is in case we have queues
#				threads = tf.train.start_queue_runners(sess=self._session, coord=coord)

				self._saver = tf.train.Saver()
#				print('restore_latest_ckpt %s ' % str(restore_latest_ckpt))
#				print('model meta file is %s ' % str(self._model_ckpt_filename + '.meta'))
#				print('is file is %s ' % str(os.path.isfile(self._model_ckpt_filename + '.meta')))
				if (restore_latest_ckpt and os.path.isfile(self._model_ckpt_filename + '.meta')):
					print('restoring %s'%self._model_ckpt_filename)
					self._saver.restore(self._session, self._model_ckpt_filename)

				training_done = False
				starting_epochs = self._training_data.current_epoch()
				first_batch = None
				time_after = time()
				while not training_done:
					time_before = time()
					overhead_time = time_before - time_after
					data_batch = self._training_data.next_batch(self._batch_size)
					time_after = time()
					next_batch_time = time_after - time_before

					if first_batch is None:
						first_batch = data_batch

					time_before = time()
					training_done, loss_value, run_results = self._train_iteration(self, self._session, data_map, self._network_output_nodes, self._loss_nodes,
																	  self._optimizer_nodes, data_batch, additional_nodes_to_evaluate)
					time_after = time()
					training_time = time_after - time_before

					iteration_count += 1

					if (mini_batches_between_sanity_check is not None) and (0 == (iteration_count-1)%mini_batches_between_sanity_check):
						training_done, _loss_value, _run_results = self._eval_iteration(self._session, data_map, self._network_output_nodes, self._loss_nodes,
																	  self._optimizer_nodes, first_batch, additional_nodes_to_evaluate)
#					first_batch = data_batch

						self._params['_loss_value'] = _loss_value
						self._params['_run_results'] = _run_results
					else:
						self._params.pop('_loss_value', None)
						self._params.pop('_run_results', None)

					# if exit after n iterations
					if (num_iterations and num_iterations <= iteration_count):
						training_done = True

					# if exit after n epochs
					if (num_epochs and (self._training_data.current_epoch()- starting_epochs > num_epochs)):
						training_done = True

					# call the train_iteration_done event handler
					if self._train_iteration_done:
						self._params['stats']['next_batch_time_list'].append(next_batch_time)
						self._params['stats']['training_time_list'].append(training_time)
						self._params['stats']['overhead_time_list'].append(overhead_time)
						training_done = self._train_iteration_done(self, self._training_data.current_epoch(),
																   self._training_data.current_index(), iteration_count,
																   loss_value, training_done, run_results,
																   params=self._params)

					# save check point
					if save_ckpt and (training_done or (0 == ((iteration_count - 1) % mini_batches_between_checkpoint))):
#						save_path = self._saver.save(self._session, ckpt_dir + self._model_name + '_' + str(iteration_count) + CHECKPOINT_FILE_EXTENSION)
						save_path = self._saver.save(self._session, self._model_ckpt_filename)
						if on_checkpoint_saved is not None:
							on_checkpoint_saved(self, self._params, save_path)
							if self._model_log_db is not None:
								self._model_log_db.on_checkpoint_saved(save_path)
		if self._model_log_db is not None:
			self._model_log_db.done_training()

	def test_batch(self, data_batch, nodes_to_evaluate):
		"""
		Default function to handle a single train iteration

		@param tf_session: the tf session
		@param data_map: a dict that maps from name of the data field in minibatch to name of placeholder
		@param network_output_nodes: output nodes of the network
		@param loss_nodes: loss nodes
		@param optimizer_nodes: optimizer operations
		@param data_batch: the data mini batch to be used for training
		@return:
		  training_done, loss_value
		  training_done will tell the main loop to exit.
		  loss_value = loss value after current training iteration.
		"""
	
		# create feed_dict from data_batch and data_map
		# data_batch is the actual data by columns
		# data_map maps from column names to the placeholder
		feed_dict = {}
		data_map = self._current_datamap
		for data_column_name, data_column in data_batch.items():
			if (data_column_name in data_map):
				feed_dict[data_map[data_column_name]] = data_column
		feed_dict[data_map[IS_TRAINING_PLACEHOLDER_NAME]] = True
		tf_nodes = []
		if isinstance(nodes_to_evaluate, list):
			for node in nodes_to_evaluate:
				n = node
				if isinstance(node, str):
					n = tf.get_default_graph().get_tensor_by_name(node + ':0')
					tf_nodes += [n]  # additional_nodes_to_evaluate
				#		print(feed_dict)
		run_results = self._session.run(tf_nodes, feed_dict)
		return run_results

	def save(self, output_dir=None, pb_filename=None, ckpt_filename=None):
		"""
		Save the model to a pb and ckpt

		@param output_dir: the output directory to save
		@param pb_filename: name of the pb file.  Exact path will be output_dir + '/' + pb_filename
		@param ckpt_filename: name of the ckpt_file.  Exact path will be output_dir + '/' + ckpt_filename
		@return:
		   None
		"""
		if output_dir is None:
			output_dir = self._model_output_location
		if pb_filename is None:
			pb_filename = self._model_name + MODEL_GRAPH_FILE_EXTENSION
		if ckpt_filename is None:
			ckpt_filename = self._model_name + CHECKPOINT_FILE_EXTENSION
		if not (self._model_graph_def is None):
			# if a model was created, then save pb
			print('saving to %s/%s' % (output_dir,pb_filename))
			tf.train.write_graph(self._model_graph_def, output_dir, pb_filename, False)
		if not (self._session is None) and not(self._saver is None):
			# if model was trained, then save ckpt
			print('saving to %s/%s' % (output_dir, ckpt_filename))
			save_path = self._saver.save(self._session, '%s/%s' % (output_dir, ckpt_filename))

	def dump_graph(self):
		"""
		Dump graph for debugging purposes
		@return:
		"""
		for node in self._model_graph_def.node:
			print('%s (%s)' % (node.name, node.op))

	@staticmethod
	def _l2_diff_loss(network_output_nodes, params=None):
		"""
		Default loss is an L2 loss that creates new inputs with names 'yn'

		@param network_output_nodes: list of tensorflow nodes
		@return:
		 a list of tensorflow nodes.  This list has equal length as network_output_nodes.
		"""
		loss = []
		for j, network_output_node in enumerate(network_output_nodes):
			y, _ = core.input_data(network_output_node.get_shape(), 'y%d'%j)
			loss.append(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y,network_output_node)))))
		return loss

	@staticmethod
	def _adam_optimizer(optimizer_param, loss_nodes, learning_rate, var_lists=None):
		"""
		Default optimizer uses adam optimizer

		@param optimizer_param: dict
		@param loss_nodes: list of tensorflow nodes
		@param var_lists: list of list of variables to optimize
		@return:
		a list of tensorflow optimizer nodes.  This list has equal length as loss_nodes.
		"""
		optimizer_nodes = []

		# if var_lists is None, then make it a list of None matching the # of loss nodes
		if var_lists==None:
			var_lists=[None]*len(loss_nodes)

		# just create adam optimizers
		for loss_node, var_list in zip(loss_nodes, var_lists):
			loss = loss_node

			if utils.get_dict_value(optimizer_param, ENABLE_REGULARIZATION_PARAM_NAME, False):
				reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
				reg_constant = 1.0 # already have wd, make this parametrizable
				loss += reg_constant * sum(reg_losses)
			min_node = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list)
			optimizer_nodes.append(min_node)
		return optimizer_nodes

	@staticmethod
	def _default_train_iteration(trainer, tf_session, data_map, network_output_nodes, loss_nodes, optimizer_nodes, data_batch, additional_nodes_to_evaluate=None):
		"""
		Default function to handle a single train iteration

		@param tf_session: the tf session
		@param data_map: a dict that maps from name of the data field in minibatch to name of placeholder
		@param network_output_nodes: output nodes of the network
		@param loss_nodes: loss nodes
		@param optimizer_nodes: optimizer operations
		@param data_batch: the data mini batch to be used for training
		@return:
		  training_done, loss_value
		  training_done will tell the main loop to exit.
		  loss_value = loss value after current training iteration.
		"""

		# create feed_dict from data_batch and data_map
		# data_batch is the actual data by columns
		# data_map maps from column names to the placeholder
		feed_dict = {}
		for data_column_name, data_column in data_batch.items():
			if (data_column_name in data_map):
				feed_dict[data_map[data_column_name]] = data_column
		feed_dict[data_map[IS_TRAINING_PLACEHOLDER_NAME]] = True

		nodes_to_evaluate = [loss_nodes[0], optimizer_nodes[0]]
		if isinstance(additional_nodes_to_evaluate, list):
			for node in additional_nodes_to_evaluate:
				n = node
				if isinstance(node, str):
					n = tf.get_default_graph().get_tensor_by_name(node + ':0')
				nodes_to_evaluate += [n] #additional_nodes_to_evaluate
#		print(feed_dict)
		run_results = tf_session.run(nodes_to_evaluate, feed_dict)
		loss_value = run_results[0]

		results_map = {}
		if additional_nodes_to_evaluate is not None:
			for node, result in zip(additional_nodes_to_evaluate, run_results[2:]):
				results_map[node] = result
		return False, loss_value, results_map

	@staticmethod
	def _rnn_train_iteration(trainer, tf_session, data_map, network_output_nodes, loss_nodes, optimizer_nodes, data_batch,
															 additional_nodes_to_evaluate=None):
		"""
		Default function to handle a single train iteration
			@param tf_session: the tf session
		@param data_map: a dict that maps from name of the data field in minibatch to name of placeholder
		@param network_output_nodes: output nodes of the network
		@param loss_nodes: loss nodes
		@param optimizer_nodes: optimizer operations
		@param data_batch: the data mini batch to be used for training
		@return:
		  training_done, loss_value
		  training_done will tell the main loop to exit.
		  loss_value = loss value after current training iteration.
		"""
		# create feed_dict from data_batch and data_map
		# data_batch is the actual data by columns
		# data_map maps from column names to the placeholder

		max_mb_on_state = utils.get_dict_value(trainer._params, 'max_mb_on_state', -1)
		if hasattr(trainer, 'state') and ((max_mb_on_state<0) or (trainer.mb_on_state < max_mb_on_state)):
			state = trainer.state
		else:
			print("RESETING STATE")
			trainer._training_data.reset_stream()
			state = np.zeros([trainer._params['num_layers'], 2, trainer._batch_size, trainer._params['cell_size']])
			trainer.mb_on_state = 0

		data_batch['state'] = state

		feed_dict = {}
		for data_column_name, data_column in data_batch.items():
			if (data_column_name in data_map):
				feed_dict[data_map[data_column_name]] = data_column
		feed_dict[data_map[IS_TRAINING_PLACEHOLDER_NAME]] = True

		nodes_to_evaluate = [loss_nodes[0], optimizer_nodes[0],
												 tf.get_default_graph().get_tensor_by_name('final_state:0')]
		if isinstance(additional_nodes_to_evaluate, list):
			for node in additional_nodes_to_evaluate:
				n = node
				if isinstance(node, str):
					n = tf.get_default_graph().get_tensor_by_name(node + ':0')
				nodes_to_evaluate += [n]  # additional_nodes_to_evaluate
			#		print(feed_dict)

		run_results = tf_session.run(nodes_to_evaluate, feed_dict)
		loss_value = run_results[0]

		results_map = {}
		if additional_nodes_to_evaluate is not None:
			for node, result in zip(additional_nodes_to_evaluate, run_results[3:]):
				results_map[node] = result

		trainer.mb_on_state += 1
		trainer.state = run_results[2]
		return False, loss_value, results_map

	def _eval_iteration(self, tf_session, data_map, network_output_nodes, loss_nodes, optimizer_nodes, data_batch, additional_nodes_to_evaluate=None):
		"""
		Default function to handle a single train iteration

		@param tf_session: the tf session
		@param data_map: a dict that maps from name of the data field in minibatch to name of placeholder
		@param network_output_nodes: output nodes of the network
		@param loss_nodes: loss nodes
		@param optimizer_nodes: optimizer operations
		@param data_batch: the data mini batch to be used for training
		@return:
		  training_done, loss_value
		  training_done will tell the main loop to exit.
		  loss_value = loss value after current training iteration.
		"""

		# create feed_dict from data_batch and data_map
		# data_batch is the actual data by columns
		# data_map maps from column names to the placeholder
		feed_dict = {}
		for data_column_name, data_column in data_batch.items():
			if (data_column_name in data_map):
				feed_dict[data_map[data_column_name]] = data_column
		feed_dict[data_map[IS_TRAINING_PLACEHOLDER_NAME]] = True
		nodes_to_evaluate = [loss_nodes[0]] #, optimizer_nodes[0]]
		if isinstance(additional_nodes_to_evaluate, list):
			for node in additional_nodes_to_evaluate:
				n = node
				if isinstance(node, str):
					n = tf.get_default_graph().get_tensor_by_name(node + ':0')
				nodes_to_evaluate += [n] #additional_nodes_to_evaluate
#		print(feed_dict)
		run_results = tf_session.run(nodes_to_evaluate, feed_dict)
		loss_value = run_results[0]

		results_map = {}
		if additional_nodes_to_evaluate is not None:
			for node, result in zip(additional_nodes_to_evaluate, run_results[1:]):
				results_map[node] = result
		return False, loss_value, results_map
