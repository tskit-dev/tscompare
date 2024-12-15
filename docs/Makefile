
# Need to set PYTHONPATH so that we pick up the local tscompare
PYPATH=${PWD}/..
TSC_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import tscompare; print(tscompare.__version__.split("+")[0])')

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for tscompare version ${TSC_VERSION}
	sed -i s/__TSCOMPARE__VERSION__/${TSC_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR _build
