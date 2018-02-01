import os
os.environ['TDSDUMP'] = 'stdout'
#import pymssql
import pyodbc
import socket

class ModelLogDb:
	_all_objects = []

	def __init__(self):
		try:
			dsn = "modelrepo"
			user = "thduon"
			password = "m0del_rep0"
			db = "model_repo"
			self._db = pyodbc.connect("Driver={ODBC Driver 13 for SQL Server};"
														"Server=modelrepo.database.windows.net;"
														"Database=model_repo;"
														"UID=thduon;PWD=m0del_rep0;"
																)
		except Exception as e:
			print(e)
			print("WARNING: FAILED TO CONNECT TO DATABASE")
			self._db = None
		self._id = -1
		self._done = False
		ModelLogDb._all_objects.append(self)

	def begin_training(self, model_name, output_location):
		if self._db is None:
			return
		cursor = self._db.cursor()
		training_gpu_env = ''
		if 'CUDA_VISIBLE_DEVICES'  in os.environ:
			training_gpu_env = os.environ['CUDA_VISIBLE_DEVICES']
		training_host = socket.gethostname()
		model_name = model_name
		pid = os.getpid()
		public_ip = os.popen('curl ifconfig.me -s').read().rstrip().lstrip()
		sql = "INSERT INTO [dbo].[model_training] " \
		"([model_name], [training_start], [training_host], [training_gpu_env], [training_pid], [training_host_ip], [training_state], [output_location]) " \
		" VALUES " \
					"('%s',CURRENT_TIMESTAMP,'%s','%s',%s,'%s',0,'%s')" \
					%(model_name, training_host, training_gpu_env, pid, public_ip, output_location)
		cursor.execute(sql)
		self._db.commit()

		cursor.execute("SELECT SCOPE_IDENTITY()")
		row = cursor.fetchone()
		self._id = row[0]
		return self._id

	def update_db(self, entries):
		if self._db is None:
			return
		cursor = self._db.cursor()
		update_fields = ""
		for key, value in entries.items():
			if len(update_fields) > 0:
				update_fields += ","
			if isinstance(value,str) and not(value=="CURRENT_TIMESTAMP"):
				update_fields += "%s='%s'"%(key,value)
			else:
				update_fields += "%s=%s"%(key,value)
		sql = "update [dbo].[model_training] SET %s WHERE id=%s"%(update_fields,self._id)
		print(sql)
		cursor.execute(sql)
		self._db.commit()

	def on_update(self, loss, log_line):
		self.update_db({'last_loss':loss, 'last_training_log_line':log_line,'last_update':'CURRENT_TIMESTAMP'})

	def done_training(self):
		self.update_db({'training_state':10,'training_state_change_time':'CURRENT_TIMESTAMP'})
		self._done = True

	def __del__(self):
		ModelLogDb._all_objects.remove(self)

	def on_checkpoint_saved(self, save_path):
		self.update_db({'last_checkpoint_time':'CURRENT_TIMESTAMP'})

	def _at_exit(self):
		# update exit time
		if not self._done:
			self.update_db({'training_state':9,'training_state_change_time':'CURRENT_TIMESTAMP'})

	@staticmethod
	def _static_at_exit():
		for o in ModelLogDb._all_objects:
			o._at_exit()

import atexit
atexit.register(ModelLogDb._static_at_exit)

if __name__ == "__main__":
	db = ModelLogDb()
	params = {'model_name':'testing'}
	db.begin_training("mymodel","test")
	db.done_training()