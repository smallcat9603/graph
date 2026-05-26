CC      = mpicc
CFLAGS  = -O3 -Wall -Wextra
LDFLAGS = -lm

PROG    = rw
OBJS    = main.o walker.o routing.o graph_io.o intmap.o scheduler.o node_scheduler.o

all: $(PROG)

$(PROG): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	$(RM) -f $(PROG) $(OBJS)

.PHONY: all clean
