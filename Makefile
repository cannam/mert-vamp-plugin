
LIBTORCH_INSTALL := ${HOME}/code/mert/pytorch-install

LIBTORCH_INCDIR := ${LIBTORCH_INSTALL}/include
LIBTORCH_CSRC_DIR := ${LIBTORCH_INCDIR}/torch/csrc
LIBTORCH_CSRC_INCDIR := ${LIBTORCH_CSRC_DIR}/api/include
LIBTORCH_LIBDIR = ${LIBTORCH_INSTALL}/lib
LIBTORCH_DEP_LIBDIR = ${LIBTORCH_INSTALL}/lib64

experiment:	experiment.cpp
		g++ -g experiment.cpp -o experiment -I${LIBTORCH_CSRC_INCDIR} -I${LIBTORCH_INCDIR} -I${LIBTORCH_CSRC_DIR} -L${LIBTORCH_LIBDIR} -L${LIBTORCH_DEP_LIBDIR} -Wl,--rpath ${LIBTORCH_LIBDIR} -ltorch -ltorch_cpu -lc10 -lsndfile
