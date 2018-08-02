# build release package
from shell_command import shell_call
import framework.utils.common as utils
import sys
import os
paramsfile = sys.argv[1]
model_dirname = os.path.dirname(paramsfile)
script_path = os.path.dirname(sys.argv[0])
freeze_cmd = 'python3 %s/freeze_model.py %s' % (script_path, os.path.join(model_dirname,'params.py'))
vocab_cmd = 'python3 %s/pkl2json.py %s' % (script_path, os.path.join(model_dirname,'vocab.pkl'))
if os.path.exists(os.path.join(model_dirname,'keywords.pkl')):
	keywords_cmd = 'python3 %s/pkl2json.py %s && cp %s/keywords.json %s/release' % (script_path, os.path.join(model_dirname,'keywords.pkl'), model_dirname, model_dirname)
else:
	keywords_cmd = ''
params_cmd = 'python3 %s/params2json.py %s' % (script_path, paramsfile)

params = utils.load_param_file(paramsfile)
model_name = params['model_name']

release_dir_name = model_name
release_cmds = []
release_cmds.append('mkdir %s' % os.path.join(model_dirname, release_dir_name))

release_files = [	model_dirname + '/*.graphdef',
									model_dirname + '/*.json',
									model_dirname + '/release.timestamp.txt',
								]
for src_file in release_files:
	release_cmds.append('cp -rvf %s %s' % (src_file, os.path.join(model_dirname, release_dir_name)))

copy2repo = 'python3 %s/copy2repo.py --paramsfile %s --release_dir_name %s'%(script_path, os.path.join(model_dirname,'params.py'), release_dir_name)

cmds = [freeze_cmd, vocab_cmd, params_cmd] + release_cmds + [keywords_cmd, copy2repo]
for c in cmds:
	if len(c)>0:
		print("EXECUTING: %s"%c)
		shell_call(c)
