class BaseDataObject:
	def __init__(self):
		self._current_epoch = 0
		self._current_index = 0
		self._num_minibatches = 0

	def current_epoch(self):
		return self._current_epoch

	def current_index(self):
		return self._current_index