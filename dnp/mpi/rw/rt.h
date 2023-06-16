//
// by smallcat 20230609
// 
// 
//

#ifndef RT_H  
#define RT_H

//58 "[(107, 1), (1684, 1), (3173, 1)]" --> src = 58, dst_proc[] = {107, 1, 1684, 1, 3173, 1}
typedef struct {
    int src;
    int* dst_proc;
    int num; //number of elements in dst_proc
} rt;

void insert_rt(rt **dict, int idx, int key, int* value, int num);
int get_rt(rt *dict, int rt_size, int key);
void parseLine(const char *line, int *src, int **dst_proc, int *num);
void read_rt(const char* file, rt** dict, int* rt_size);

#endif