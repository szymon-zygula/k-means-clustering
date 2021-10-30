CC=nvcc
C_FLAGS=
FILES=main.cu
TARGET=kmeans_clustering

${TARGET}:
	${CC} ${C_FLAGS} ${FILES} -o ${TARGET}

clean:
	-rm -rf ${TARGET}

.PHONY: clean
