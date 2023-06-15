CC = mpicc
RM = /bin/rm
PROG = rw

CFLAGS = -O3 -ligraph -lm
LDFLAGS = ${CFLAGS}

all : ${PROG}

rw : rw.c algo.c file.c rt.c
	${CC} -o $@ $^ ${LDFLAGS}
	
clean :
	${RM} -f ${PROG} 






