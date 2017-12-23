import timeit
import csv
import numpy as np
import random
import pickle
import logging
import random

class TrainingData(object):
    """
    A class to read delimited data files and generate mini batches for training.

    This class will read the entire data file into memory on construction.

    Example use:

    from framework.data.training_data import TrainingData
    malta_data = TrainingData.load_delimited_file(

    """

    def __init__(self, data, header=None, shuffle=True):
        """
        Constructor for TrainingData that takes the actual data and an optional header.

        @param data : array of array that is the actual data.  each array in the array is a row.
        @param header : array of strings that are headers for the columns
        @param shuffle : whether to shuffle
        """
        if header:
            self._header = header
        else:
            self._header = list(range(len(data[0])))
        self._data = data
        self._shuffle = shuffle
        self._current_index = 0                  # index of current row in current epoch
        self._epoch_count = 0
        self._header_map = {}
        for i in range(len(self._header)):
            self._header_map[self._header[i]] = i
        if self._shuffle:
            random.shuffle(self._data)

    def current_epoch(self):
        """
        @return:the # of epochs finished
        """
        return self._epoch_count

    def current_index(self):
        """
        @return: current index in the dataset
        """
        return self._current_index

    def epoch_size(self):
        """
        @return: size of the dataset
        """
        return len(self._data)

    def header(self):
        """
        @return: header labels
        """
        return self._header

    def data(self):
        """
        @return: the raw data
        """
        return self._data

    @classmethod
    def load_delimited_file(cls, filename, delimiter='\t', quotechar=None, has_header=False, header=None, data_normalizer=None, shuffle=True, progress_granularity=10000, rows_to_load=-1):
        """
        Read delimited file into a training data object
        @param filename: name of delimited file
        @param delimiter: delimiter
        @param quotechar: quote
        @param has_header: has header?
        @param header: if no header, use this as header
        @param data_normalizer: how to transform data as we read it.  This function should have 3 parameters:
            column index, column name, column data.  It should return the transformed column data.
        @param shuffle: should training data set be shuffled.
        @param progress_granularity: whether to log progress.  if this is <= 0, then no progress.  otherwise, it is
            the # of rows loaded in between progress.
        @return:
            the training data object
        """
        data = []
        logging.info('loading data file: %s' % filename)
        start_time = timeit.default_timer()
        with open(filename,'r') as infile:
            csv_file = csv.reader(infile, delimiter=delimiter)

            # read header
            if has_header:
                _header = next(csv_file)
                if not header:
                    header = _header

            # read data
            rows_loaded = 0
            for row in csv_file:
                if data_normalizer:
                    data_row = []
                    for col_index, col in enumerate(row):
                        data_row.append(data_normalizer(col_index, header[col_index] if has_header else col_index, col))
                else:
                    data_row = row
                data.append(data_row)
                rows_loaded += 1
                if rows_to_load>0 and rows_loaded>=rows_to_load:
                    break
                if progress_granularity>0 and (rows_loaded % progress_granularity == 0):
                    logging.info('loaded %d rows' % rows_loaded)
        elapsed = timeit.default_timer() - start_time
        logging.info('finished loading in %s seconds' % str(elapsed))
        return TrainingData(data, header, shuffle)

    @classmethod
    def load(cls, filename, **params):
        """
        Load data from file.  If filename ends with pkl, then load as pickle file.
        Otherwise, load as delimited file.
        @param filename:
        @param params:
        @return:
        """
        if (filename.endswith('.pkl')):
            result = cls.load_pkl_file(filename, **params)
        else:
            result = cls.load_delimited_file(filename, **params)
        return result

    @classmethod
    def load_pkl_file(cls, filename, shuffle=True, **kwargs):
        """
        Load pickle file as training data.  The pickle file
        needs to have two variables (data and header).  Ideally,
        it was saved with the save_pkl function.
        @param filename: name of the pickle file
        @param shuffle: true if data should be shuffled
        @return:
        TrainingData object
        """
        logging.info('loading data file: %s' % filename)
        start_time = timeit.default_timer()
        with open(filename, 'rb') as pkl_file:
            data, header = pickle.load(pkl_file)
            elapsed = timeit.default_timer() - start_time
            logging.info('finished loading in %s seconds' % str(elapsed))
        return TrainingData(data, header, shuffle)

    def save_as_pkl_file(self, filename):
        """
        Save data in current object to pickle file
        @param filename: name of pickle file to save
        @return:
        None
        """
        with open(filename, 'wb') as pkl_file:
            pickle.dump([self._data, self._header], pkl_file)

    def save_to_delimited_file(self, filename, delimiter='\t', eol='\n', header=None, write_header=False):
        """
        Save data to delimited file.  NOT YET IMPLEMENTED
        @param filename: name of destination file
        @param delimiter: delimiter used to separate columns
        @param eol: end of line delimiter
        @param header: headers to override the header in the class
        @param write_header: whether to write the header
        @return:
        """
        with open(filename, 'w') as file:
            if write_header:
                if not header:
                    header = self._header
                file.write(delimiter.join(map(str, header)))
                file.write(eol)
            for data_row in self._data:
                file.write(delimiter.join(map(str, data_row)))
                file.write(eol)

    def next_batch(self, batch_size, columns=None):
        """
        Get data for the next mini batch
        @param batch_size: size of the batch in # of records
        @param columns: columns to get.  if None (default), then get every column
        @return:
        a dict that maps column to data
        """
        if self._current_index==0 and self._shuffle:
            random.shuffle(self._data)

        if not columns:
            columns = self._header

        batch_data = {}

        for column in columns:
            data_size = 1
            data = self._data[0][self._header_map[column]]
            if isinstance(data, list):
                data_size = len(data)
            batch_data[column] = np.zeros([batch_size, data_size])

        row_index = 0
        while row_index < batch_size:
            for column in columns:
                batch_data[column][row_index,:] = self._data[self._current_index][self._header_map[column]]
            self._current_index += 1
            row_index += 1
            if self._current_index == len(self._data):
                # finished one epoch
                self._current_index = 0
                self._epoch_count += 1
                if self._shuffle:
                    random.shuffle(self._data)
        return batch_data

def generate_fake_1d_training_data(header=['x','y'],xlist=np.linspace(-10,10).tolist(),function=lambda x: (x*x+random.normalvariate(0,.1))):
    """
    Generate fake training 1 dimensional training data that contains (x,y) pairs.

    y = function(x).  By default this function is just a polynomial + noise.
    @param header: The labels for x and y.  Default to ['x', 'y'].
    @param xlist: x values to generate.
    @param function: the function that generates y
    @return:
        a TrainingData object that contains this fake data
    """
    data = []
    for x in xlist:
        y = function(x)
        data.append([x,y])
    return TrainingData(data, header)
