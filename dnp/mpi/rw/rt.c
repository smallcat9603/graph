//
// by smallcat 20230609
// 
// 
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rt.h"

void insert_rt(rt **dict, int idx, int key, int* value, int num) {
  rt *newDict = realloc(*dict, (idx + 1) * sizeof(rt));
  if (newDict == NULL) {
    printf("Error: Failed to allocate memory.\n");
    exit(0);
  }

  *dict = newDict;
  (*dict)[idx].src = key;
  (*dict)[idx].dst_proc = value;
  (*dict)[idx].num = num;
}

int get_rt(rt *dict, int rt_size, int key) {
  for (int i = 0; i < rt_size; i++) {
    if (dict[i].src == key) {
      return i;
    }
  }
  return -1;
}

//parse line in rt file, e.g., 58 "[(107, 1), (1684, 1), (3173, 1)]"
void parseLine(const char *line, int *src, int **dst_proc, int *num) {
  sscanf(line, "%d", src);

  const char *start = strchr(line, '[');
  const char *end = strchr(line, ']');
  if (start == NULL || end == NULL || end <= start) {
    *dst_proc = NULL;
    *num = 0;
    exit(0);
  }

  int length = end - start - 1;
  if(length == 0) {
    *dst_proc = NULL;
    *num = 0;
    exit(0);
  }

  //temp: (107, 1), (1684, 1), (3173, 1)
  char *temp = (char *)malloc(length + 1);
  strncpy(temp, start + 1, length);
  temp[length] = '\0';

  *num = 0;
  for(int i = 0; i < length; i++) {
    if (temp[i] == '(') {
      (*num)++;
    }
  }
  *num *= 2;

  *dst_proc = (int *)malloc(*num * sizeof(int));

  char *token = strtok(temp, ")");
  sscanf(token, "(%d, %d", &(*dst_proc)[0], &(*dst_proc)[1]);
  int index = 2;
  while (token = strtok(NULL, ")")) {
      sscanf(token, ", (%d, %d", &(*dst_proc)[index], &(*dst_proc)[index+1]);
      index += 2;
  }

  free(temp);
}

void read_rt(const char* file, rt** dict, int* rt_size){
  FILE* fp;
  fp = fopen(file, "r");
  if (fp == NULL) {
      printf("there is something wrong with opening %s\n", file);
      exit(0);
  }

  int idx = 0;
  char *line = NULL;
  size_t line_length = 0;
  
  while (getline(&line, &line_length, fp) != -1) {
    int src;
    int* dst_proc;
    int num;
    parseLine(line, &src, &dst_proc, &num);
    insert_rt(dict, idx, src, dst_proc, num);
    idx++;
  }
  *rt_size = idx;
  
  free(line);  
  fclose(fp);
}
