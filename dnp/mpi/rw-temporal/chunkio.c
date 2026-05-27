#include "chunkio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int file_exists(const char* path) {
    return access(path, F_OK) == 0;
}

char** resolve_chunks(const char* logical, int* count) {
    char** list = NULL;
    int    n = 0;

    if (file_exists(logical)) {
        list = (char**) malloc(sizeof(char*));
        list[0] = strdup(logical);
        n = 1;
    } else {
        int  cap = 8;
        char buf[600];
        list = (char**) malloc(sizeof(char*) * cap);
        for (int i = 0; ; i++) {
            snprintf(buf, sizeof(buf), "%s.part%03d", logical, i);
            if (!file_exists(buf)) break;
            if (n == cap) {
                cap *= 2;
                list = (char**) realloc(list, sizeof(char*) * cap);
            }
            list[n++] = strdup(buf);
        }
    }

    *count = n;
    return list;
}

void free_chunks(char** chunks, int count) {
    if (!chunks) return;
    for (int i = 0; i < count; i++) free(chunks[i]);
    free(chunks);
}
