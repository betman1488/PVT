#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

int count = 0;
    
double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double parallel(int n, int *v) {
    double t = omp_get_wtime();

    
    #pragma omp parallel
    {
       int lc = 0;
       #pragma omp for
       for (int i = 0; i < n; i++) {
           if (v[i] == 3) 
               lc++;
       }
       #pragma omp atomic
	   count += lc;
    }   
    
    t = wtime() - t;
    return t;	
}

double serial(int n, int *v) { 
    double t = wtime();
    for (int i = 0; i < n; i++) {
        if (v[i] == 3)
            count++;
    }
    t = omp_get_wtime() - t;
    return t;
}


int main(int argc, char *argv[])
{
    int n = 100000000;
    int *v = malloc(sizeof(*v) * n);
    for (int i = 0; i < n; i++)
        v[i] = rand() % 30;
        
    double t1 = serial(n, v);
    double t2 = parallel(n, v);
    
    printf("Counter (n = %d)\n", n);    
    printf("Count = %d\n", count);    
    printf("Time (sec) serial: %.6f || Time (sec) parallel: %.6f\n", t1, t2);
    printf("Speed up: %.6f\n", t1/t2);
    
    free(v);
    return 0;
}
