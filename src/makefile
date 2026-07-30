CC = gcc
CFLAGS = -Wall -Wextra -O2
LDLIBS = -lm

main: main.c
	$(CC) $(CFLAGS) main.c -o nn $(LDLIBS)

run: main
	./nn

data:
	sh get-data.sh

clean:
	rm -f nn

.PHONY: run data clean
