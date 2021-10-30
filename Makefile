CC=nvcc
C_FLAGS=
TARGET=main

${TARGET}:
	${CC} ${C_FLAGS} ${TARGET}.cu -o ${TARGET}

clean:
	-rm -rf ${TARGET}

.PHONY: clean
