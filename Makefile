main: init

init:
	python3 setup.py build_ext --inplace
#	cp core.*.so fast_ultrametrics/
test: init
	python3 test.py	
