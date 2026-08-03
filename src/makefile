CC = gcc
CFLAGS = -Wall -Wextra -O2
LDLIBS = -lm -lOpenCL

ifeq ($(OS),Windows_NT)
	TARGET = nn.exe
else
	TARGET = nn
endif

main: main.c cl_ops.c cl_ops.h
	$(CC) $(CFLAGS) main.c cl_ops.c -o nn $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

data:
	sh get-data.sh

clean:
	rm -f nn

.PHONY: run data clean
