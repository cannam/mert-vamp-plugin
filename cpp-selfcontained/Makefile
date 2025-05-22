
OPTS 	:= -O3 -fopenmp -DUSE_CBLAS

#OPTS	:= -g -fsanitize=address
#OPTS	:= -g 

OTHERLIBS	:= -lcblas

test:	main.cpp tensor.hpp model.hpp Makefile
	g++ ${OPTS} main.cpp ../data/weights.a -o $@ -lsndfile ${OTHERLIBS}

clean:
	rm test
