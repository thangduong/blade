import unittest

import framework.utils.data.text_utils as data_utils


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        None

    def test_split_str_list(self):
        results = data_utils._split_str_list(['hello world', 'OMG!LOL', 'WTF,DUDE!'], [' ','!', ','])

    def tokenize_text(self):
        test_string = 'OMG!LOL!  DUDE!  WTF@ hello world, programming language.  testing! 1,2, 3'
        results = data_utils.tokenize_text(test_string)

    def test_normalize_text(self):
        input_string = 'OMG!LOL!  DUDE!  WTF@ hello world, programming language.  testing! 1,2, 3'
        expected_output_string = 'OMG ! LOL ! DUDE ! WTF @ hello world , programming language . testing ! 1 , 2 , 3'
        output_string = data_utils.normalize_text(input_string)
        print('test_normalize_text:')
        print('input string:    "%s"' % input_string)
        print('expected output: "%s"' % expected_output_string)
        print('output string:   "%s"' % output_string)

        self.assertEqual(expected_output_string, output_string)

    def test_extract_vocab(self):
        input_string = 'OMG!LOL!  DUDE!  WTF@ hello world, programming language.  testing! 1,2, 3'
        data_utils.extract_vocab()




if __name__ == '__main__':
    unittest.main()
