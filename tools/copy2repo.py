# copy model to repo after release is called
from shell_command import shell_call
import framework.utils.common as utils
import gflags
import json
import sys
import os


def callcmd(cmd):
	print("executing: %s"%cmd)
	shell_call(cmd)

def copy2repo(source_dir, target_dir, model_name, release_dir_name="release"):
	"""
	Copy from <source_dir>/<model_name> to <target_dir>/<model_name> if doesn't exist in target.
	If exists in target, then copy <source_dir>/<model_name>/release to <target_dir>/<model_name>/release.<num>
	where <num> is the next unique number.
	@param source_dir: source directory
	@param target_dir: target directory
	@param model_name: name of model
	@return: <num>, -1 if source_model_dir doesn't exist, -2 if no release dir in source_model_dir
	"""
	target_model_dir = os.path.join(target_dir, model_name)
	source_model_dir = os.path.join(source_dir, model_name)
	if not os.path.isdir(source_model_dir):
		return -1, source_model_dir
	if not os.path.isdir(target_dir):
		return -3, target_dir, target_model_dir

	cmd = 'cp -rvf \"%s\" \"%s\"'%(source_model_dir,target_dir)
	callcmd(cmd)

	release_num = 0
	found = False
	while not found:
		target_release_dir = os.path.join(target_model_dir, '%s.%s'%(release_dir_name, release_num))
		if os.path.isdir(target_release_dir):
			release_num+=1
		else:
			found=True

	source_release_dir = os.path.join(source_model_dir, release_dir_name)
	if not os.path.isdir(source_release_dir):
		return -2, source_release_dir, target_model_dir

	cmd = 'cp -rvf \"%s\" \"%s\"'%(source_release_dir,target_release_dir)
	callcmd(cmd)
	return release_num, None, target_model_dir

def add_release_num_to_json(json_file, release_num):
	# load
	with open(json_file, 'r', encoding='utf-8') as data_file:
		data = json.loads(data_file.read())

	# modify
	data['release_num'] = release_num

	# save back
	with open(json_file, 'w', encoding='utf-8') as data_file:
		json.dump(data, data_file)

def main(argv):
	try:
		argv = FLAGS(argv)  # parse flags
	except gflags.FlagsError as e:
		print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))
		sys.exit(1)

	params = utils.load_param_file(FLAGS.paramsfile)
	model_dirname = os.path.dirname(FLAGS.paramsfile)
	src_dir = os.path.abspath(os.path.join(model_dirname, '..'))
	target_dir = FLAGS.target_dir
	model_name = params['model_name']
	release_dir_name = FLAGS.release_dir_name
	if release_dir_name == "":
		release_dir_name = model_name
	release_num, dir_name, target_model_dir = copy2repo(src_dir, target_dir, model_name, release_dir_name)
	if release_num == -1:
		print("source dir %s doesn't exist" % dir_name)
	elif release_num == -2:
		print("release source dir %s doesn't exist" % dir_name)
	elif release_num == -3:
		print("target dir %s doesn't exist" % dir_name)
	else:
		add_release_num_to_json(os.path.join(target_model_dir,release_dir_name,'params.json'), release_num)
		add_release_num_to_json(os.path.join(target_model_dir,'%s.%s'%(release_dir_name,release_num),'params.json'), release_num)
		print("SUCCESSFULLY COPY MODEL %s.%s TO REPO"%(model_name, release_num))

if __name__ == '__main__':
	gflags.DEFINE_string('paramsfile', 'params.py', 'parameters file')
	gflags.DEFINE_string('target_dir', '/models/dlframework_models', 'directory of repo')
	gflags.DEFINE_string('release_dir_name', 'release', 'release dir name.  if blank, use model name')
	FLAGS = gflags.FLAGS
	main(sys.argv)
