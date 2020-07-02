CC=mpicc

DEPS_PROG=alltoall_personalized.c

program:
	$(CC) $(DEPS_PROG)
	@echo "Usage: mpirun -n <no_of_processes> ./a.out"

clean:
	rm -rf *.o *.out