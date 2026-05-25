CC      = mpicc
CFLAGS  = -O3 -Wall -Wextra
LDFLAGS = -ligraph -lm

PROG    = rw
OBJS    = main.o walker.o routing.o graph_io.o intmap.o

all: $(PROG)

$(PROG): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	$(RM) -f $(PROG) $(OBJS)

.PHONY: all clean
