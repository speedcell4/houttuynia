PROJECT_DIR = ${CURDIR}
BIN_DIR = ${PROJECT_DIR}/venv/bin
PYTHON = PYTHONPATH=${PROJECT_DIR} ${BIN_DIR}/python3.6

${BIN_DIR}/python3:
	virtualenv --python=python3.6 venv

${BIN_DIR}/jupyter:
	$(PYTHON) -m pip install jupyter

${BIN_DIR}/tensorboard:
	$(PYTHON) -m pip install tensorflow==1.8.0 tensorboard

${PROJECT_DIR}/.dep: ${BIN_DIR} ${PROJECT_DIR}/requirements.txt
	$(PYTHON) -m pip install -r ${PROJECT_DIR}/requirements.txt --no-cache
	touch ${PROJECT_DIR}/.dep

.PHONY:
notebook: ${BIN_DIR}/jupyter
	${BIN_DIR}/jupyter notebook --no-browser --ip 0.0.0.0

.PHONY:
tensorboard: ${BIN_DIR}/tensorboard
	${BIN_DIR}/tensorboard --logdir ${PROJECT_DIR}/out --host 0.0.0.0

.PHONY:
test: ${BIN_DIR}/python3 ${PROJECT_DIR}/.dep ${PROJECT_DIR}/tests
	$(PYTHON) -m unittest discover -s ${PROJECT_DIR}/tests -t ${PROJECT_DIR}/tests

.PHONY:
iris: ${BIN_DIR}/python3 ${PROJECT_DIR}/.dep ${PROJECT_DIR}/examples/iris.py
	$(PYTHON) ${PROJECT_DIR}/examples/iris.py