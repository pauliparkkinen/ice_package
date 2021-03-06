# $Id: Makefile,v 1.6 2008/10/29 01:01:35 ghantoos Exp $
#

PYTHON=`which python`
MPICC=`which mpicc`
DESTDIR=/
BUILDIR=$(CURDIR)/debian/ice_package
PROJECT=icepackage
VERSION=1.0.0

all:
		@echo "make source - Create source package"
		@echo "make install - Install on local system"
		@echo "make buildrpm - Generate a rpm package"
		@echo "make builddeb - Generate a deb package"
		@echo "make clean - Get rid of scratch and byte files"

source:
		CC=$(MPICC) $(PYTHON) setup.py sdist $(COMPILE)

install:
		CC=$(MPICC) $(PYTHON) setup.py build_ext -i
		CC=$(MPICC) $(PYTHON) setup.py install --root $(DESTDIR) $(COMPILE)

buildrpm:
		#CC=$(MPICC) $(PYTHON) setup.py build_ext -i
		CC=$(MPICC) $(PYTHON) setup.py bdist_rpm

builddeb:
		# build the source package in the parent directory
		# then rename it to project_version.orig.tar.gz
		CC=$(MPICC) $(PYTHON) setup.py sdist $(COMPILE) --dist-dir=../ --prune
		rename -f 's/$(PROJECT)-(.*)\.tar\.gz/$(PROJECT)_$$1\.orig\.tar\.gz/' ../*
		# build the package
		dpkg-buildpackage -i -I -rfakeroot

clean:
		CC=$(MPICC) $(PYTHON) setup.py clean
		#$(MAKE) -f $(CURDIR)/debian/rules clean
		rm -rf build/ MANIFEST
		find . -name '*.pyc' -delete

