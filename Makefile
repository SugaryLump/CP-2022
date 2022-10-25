CC				= gcc
BIN				= bin/
SRC				= src/
INCLUDES		= include/
EXEC			= k_means

CFLAGS			= "-lm" "-g" -Iincludes -Ofast -ftree-vectorize -ffast-math -msse2 -fopt-info-vec-missed -fno-omit-frame-pointer -std=gnu99

.DEFAULT_GOAL 	= k_means

$(BIN):
	mkdir -p $(BIN)

k_means: $(SRC)k_means.c $(BIN)
	$(CC) $(CFLAGS) $(SRC)k_means.c -o $(BIN)k_means

#$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
#	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

run:
	./$(BIN)$(EXEC) 10000000 4
