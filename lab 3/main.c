#define _POSIX_C_SOURCE 1

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

const double PI = 3.14159265358979323846;
const int n = 100000000;

double getrand(unsigned int *seed)
{
    return (double)rand_r(seed) / RAND_MAX;
}
double func(double x, double y)
{
	return 3 * pow(y, 2) * pow(sin(x), 2); 
}

double serial()
{
	printf("Serial: n = %d\n", n);
	double t = omp_get_wtime();
    unsigned int seed = 0;
    int in = 0;
    double s = 0;
    for (int i = 0; i < n; i++) {
		double x = getrand(&seed) * PI;
		double y = getrand(&seed);
		if (y <= sin(x)) {
			in++;
			s += func(x, y);
		}
    }
    double v = PI * in / n;
    double res = v * s / in;
   	t = omp_get_wtime() - t;
    printf("Result: %.12f\n\n", res);
    return t;
}

double parallel()
{
    printf("Parallel: n = %d\n", n);
	double t = omp_get_wtime();
	int in = 0;
    double s = 0;
	#pragma omp parallel 
	{
		double s_loc = 0;
		int in_loc = 0;
		unsigned int seed = omp_get_thread_num();
		#pragma	omp for nowait
			for (int i = 0; i < n; i++) {
				double x = getrand(&seed) * PI;
				double y = getrand(&seed);
				if (y <= sin(x)) {
					in_loc++;
					s_loc += func(x, y);
				}
			}
		#pragma	omp atomic 
		s += s_loc;
		#pragma	omp atomic 
		in += in_loc;
    }
    double v = PI * in / n;
    double res = v * s / in;
    t = omp_get_wtime() - t;
    printf("Result: %.12f\n\n", res, n);
	return t;

}


int main(int argc, char **argv)
{
	
	double t1 = serial();
	double t2 = parallel();
	printf ("Rez time(sec.): %.12f (serial) | %.12f (parallel)\n", t1, t2);
	printf ("Speed up (sec.): %.12f\n", t1/t2);
    return 0;

}
