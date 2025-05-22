
MKL_PATH	:= /opt/intel/oneapi/mkl/latest

OPTS 	:= -O3 -fopenmp -DUSE_CBLAS -DUSE_MKL -I${MKL_PATH}/include  # -fopt-info-vec-optimized
#OPTS	:= -g -fsanitize=address
#OPTS	:= -g 

OTHERLIBS	:= -L${MKL_PATH}/lib -Wl,-Bstatic -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -Wl,-Bdynamic

test:	main.cpp tensor.hpp model.hpp Makefile
	g++ ${OPTS} main.cpp ../data/weights.a -o $@ -lsndfile ${OTHERLIBS}

clean:
	rm test
