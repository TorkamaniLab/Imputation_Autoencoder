#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

char *join(int argc, char *const argv[], int *size_p) {
    int size = argc == 1 ? 1 : argc-1; // space separations+ null terminator
    char *buf;

    for(int i=1; i<argc; i++) {
        int slen = strlen(argv[i]);
        size += slen;
    }
    if( (buf = malloc(size)) == NULL) {
        fprintf(stderr, "Memory error!\n");
        exit(1);
    }
    int loc = 0;
    for(int i=1; i<argc; i++) {
        int slen = strlen(argv[i]);
        memcpy(buf+loc, argv[i], slen);
        loc += slen+1;
        buf[loc-1] = ' ';
    }
    buf[size-1] = 0;
    *size_p = size;
    return buf;
}

int main(int argc, char *argv[]) {
    int rank, size;
    char *buf = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
        buf = join(argc, argv, &size);
    }
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank != 0) {
        if( (buf = malloc(size)) == NULL) {
            fprintf(stderr, "Memory error!\n");
            exit(1);
        }
    }
    MPI_Bcast(buf, size, MPI_CHAR, 0, MPI_COMM_WORLD);
    printf("%s\n", buf);
    free(buf);

    MPI_Finalize();
    return 0;
}
