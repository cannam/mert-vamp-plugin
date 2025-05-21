
WEIGHTS	:= $(subst .cpp,.o,$(wildcard weights*.cpp))

weights.a: ${WEIGHTS}
	ar rcf $@ $^

