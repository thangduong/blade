import json, pickle
import sys, os


def pkl2json(pkl_filename, json_filename):
	with open(pkl_filename, "rb") as fi:
		with open(json_filename, 'w') as fo:
			data = pickle.load(fi)
			json.dump(data, fo)


if __name__ == "__main__":
	pkl_filename = sys.argv[1]
	json_filename, _ = os.path.splitext(pkl_filename)
	json_filename += '.json'
	print("FROM: %s\nTO: %s\n"%(pkl_filename, json_filename))
	pkl2json(pkl_filename, json_filename)
	print("Wrote: %s"%json_filename)