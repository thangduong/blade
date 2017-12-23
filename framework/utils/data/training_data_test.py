import copy
import unittest

import framework.utils.data.training_data as training_data


class TestTrainingData(unittest.TestCase):
    """ Unit test class to test the stuff in training_data.py
    """
    def test_training_data_class(self):
        """
        Test the TrainingData class.  The following methods are tested:
          constructor, next_batch, constructor with shuffle
        @return: None
        """
        header = ['number 1', 'number 2']
        raw_data = [[1,2],[3,4],[5,6],[7,8],[9,10]]
        raw_data_dup = copy.deepcopy(raw_data)
        data = training_data.TrainingData(raw_data, header, shuffle=False)
        self.assertSequenceEqual(raw_data, raw_data_dup)

        # make sure we can get batches of different sizes
        for batch_size in range(1,10):
            batch = data.next_batch(batch_size)
            self.assertEqual(len(batch), len(header))
            for col in header:
                self.assertEqual(len(batch[col]), batch_size)

        # make sure we can shuffle
        good_count = 0
        for try_count in range(1, 5):
            raw_data = copy.deepcopy(raw_data_dup)
            data = training_data.TrainingData(raw_data, header, shuffle=True)
            unequal_count = 0
            for raw_row in raw_data_dup:
                batch = data.next_batch(1)
                equal = True
                for col_index, raw_col in enumerate(raw_row):
                    if not (batch[header[col_index]][0] == raw_col):
                        equal = False
                if not equal:
                    unequal_count += 1
            if unequal_count > 0:
                good_count += 1
        self.assertTrue(good_count > 2)

    def test_training_data_save_load(self):
        """
        Test the saving and loading functions in the TrainingData class
        @return:
        """
        header = ['number 1', 'number 2']
        raw_data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        data = training_data.TrainingData(raw_data, header, shuffle=False)
        tsv_out_filename = '/tmp/raw_data.tsv'
        pkl_out_filename = '/tmp/raw_data.pkl'
        data.save_to_delimited_file(tsv_out_filename, write_header=True)
        data_tsv_loaded = training_data.TrainingData.load_delimited_file(tsv_out_filename, data_normalizer=lambda col, col_name, x: int(x), shuffle=False, has_header=True)
        self.assertSequenceEqual(data.data(), data_tsv_loaded.data())
        self.assertSequenceEqual(data.header(), data_tsv_loaded.header())

        data.save_as_pkl_file(pkl_out_filename)
        data_pkl_loaded = training_data.TrainingData.load_pkl_file(pkl_out_filename, shuffle=False)
        self.assertSequenceEqual(data.data(), data_pkl_loaded.data())
        self.assertSequenceEqual(data.header(), data_pkl_loaded.header())

    def test_generate_fake_1d_training_data(self):
        """
        test the training_data.generate_fake_1d_training_data function
        @return:
        """

        # interactive testing
        random_data = training_data.generate_fake_1d_training_data()

        batch = random_data.next_batch(3)
        self.assertEqual(len(batch), 2)
        self.assertEqual(len(batch['x']), 3)
        self.assertEqual(len(batch['y']), 3)


if __name__ == '__main__':
    unittest.main()
