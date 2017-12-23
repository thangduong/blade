from framework.evaluator import Evaluator

MODEL_NAME = 'polynomial'
PB_FILENAME = 'polynomial.pb'
CKPT_FILENAME = 'polynomial.ckpt'
OUTPUT_DIR = '/tmp'

# test evaluation code
e = Evaluator.load(model_dir=OUTPUT_DIR,pb_filename=PB_FILENAME,ckpt_filename=CKPT_FILENAME)
for x in [1,2,3,4,5,6,7,8,9,10]:
    print('f(%f) = %f' % (x, e.eval({'x': x}, 'ybar')[0]))
