CC				= gcc
BIN				= bin/
SRC				= src/
INCLUDES		= include/
EXEC			= k_means

CFLAGS			= "-lm" "-g"

.DEFAULT_GOAL 	= k_means

k_means: $(SRC)k_means.c
	$(CC) $(CFLAGS) $(SRC)k_means.c -o $(BIN)k_means

#$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
#	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

run:
	./$(BIN)$(EXEC) 10000000 4