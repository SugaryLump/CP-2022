CC				= nvcc
BIN				= bin/
SRC				= src/
INCLUDES		= include/
EXEC			= k_means

CFLAGS			= "-lm" -Iincludes -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets -Xcompiler -fopenmp -g

.DEFAULT_GOAL 	= k_means

$(BIN):
	mkdir -p $(BIN)

k_means: $(SRC)k_means.cu $(BIN)
	$(CC) $(CFLAGS) $(SRC)k_means.cu -o $(BIN)k_means

#$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
#	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

run:
	./$(BIN)$(EXEC) 10000000 4