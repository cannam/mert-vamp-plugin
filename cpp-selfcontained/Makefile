
OPTS 	:= -O3 -fopenmp  # -fopt-info-vec-optimized
#OPTS	:= -g -fsanitize=address
#OPTS	:= -g 

test:	main.cpp tensor.hpp model.hpp Makefile
	g++ ${OPTS} main.cpp ../data/weights.a -o $@ -lsndfile ${OTHERLIBS}

clean:
	rm test
