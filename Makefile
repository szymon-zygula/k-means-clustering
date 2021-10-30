CC=nvcc
C_FLAGS=--gpu-architecture sm_60
FILES=main.cu
TARGET=kmeans_clustering

${TARGET}:
	${CC} ${C_FLAGS} ${FILES} -o ${TARGET}

clean:
	-rm -rf ${TARGET}

.PHONY: clean
