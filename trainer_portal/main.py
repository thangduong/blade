#!/usr/bin/python3
# this is the main entry point
from flask import Flask
from flask import request
from flask import Flask, request, send_from_directory, redirect
import framework.db.model_log_db as db
import json
models = []

app = Flask(__name__)

model_log_db = db.ModelLogDbViewer()
default_handler = lambda obj: (
    str(obj)
)
@app.route('/jobs', methods=['GET'])
def jobs():
	rows, colnames = model_log_db.get_all_trainers()
	colmap = {}
	for i,name in enumerate(colnames):
		colmap[name] = i
	data = []
	for x in rows:
		data.append(list(x))
	result = {'cols':colmap, 'colnames':colnames, 'data':data}
	return json.dumps(result, default=default_handler)

@app.route('/html/<path:path>')
def send_html(path):
    return send_from_directory('html', path)

@app.route('/html/')
def html():
    return send_from_directory('html', 'index.html')

@app.route('/')
def root():
	return redirect("/html", code=302)

if __name__ == '__main__':
	print("ALL LOADED!")
	app.run(host="0.0.0.0",debug=True,port=8088,threaded=True)

