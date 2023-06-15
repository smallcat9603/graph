CC = mpicc
RM = /bin/rm
PROG = rw

CFLAGS = -O3
LDFLAGS = -ligraph -lm

all : ${PROG}

rw : rw.c algo.c file.c rt.c
	${CC} -o $@ $^ ${CFLAGS} ${LDFLAGS}

clean :
	${RM} -f ${PROG} 
