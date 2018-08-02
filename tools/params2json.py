import json, pickle
import sys, os
import framework.utils.common as utils


def params2json(params_file, json_filename):
	params = utils.load_param_file(params_file)
	with open(json_filename, 'w') as fo:
		ignore_list = []
		for f in params:
			if not (isinstance(params[f], list) or isinstance(params[f], dict) or
								isinstance(params[f], int) or isinstance(params[f], str)
							or isinstance(params[f], float)
							):
				ignore_list.append(f);
		for f in ignore_list:
			print("IGNORING %s" % f)
			del params[f]
		json.dump(params, fo)

if __name__ == "__main__":
	params_file = sys.argv[1]
	json_filename, _ = os.path.splitext(params_file)
	json_filename += '.json'
	print("FROM: %s\nTO: %s\n"%(params_file, json_filename))
	params2json(params_file, json_filename)
	print("Wrote: %s"%json_filename)