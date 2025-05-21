
OPTS 	:= -O3 -fopenmp
#OPTS	:= -g -fsanitize=address
#OPTS	:= -g 

experiment:	experiment.cpp withoutlib.cpp Makefile
		g++ ${OPTS} experiment.cpp ../data/weights.a -o experiment -lsndfile ${OTHERLIBS}

