#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <inttypes.h>
#include <math.h>

struct thread_data_fs {
	double sum;
	uint8_t padding[64 - sizeof(double)];
};

struct thread_data {
	double sum;
};

double func (double x)
{
  return exp (-x * x);
}

double serial(double a, double b, int n) 
{

	double t = omp_get_wtime();
	
	double h = (b - a) / n;
	double s = 0.0;
	for (int i = 0; i < n; i++)
		s += func(a + h * (i + 0.5));
	s *= h;
	t = omp_get_wtime() - t;
	return t;
	
}

double parallel_fs(double a, double b, int n) 
{
	
	double h = (b - a) / n;
	double s = 0.0;
	double t = omp_get_wtime ();
	struct thread_data sumloc[omp_get_max_threads()];
	
	#pragma omp parallel
	{
		int nthreads = omp_get_num_threads ();
		int tid = omp_get_thread_num ();
		int points_per_thread = n / nthreads;
		int lo = tid * points_per_thread;
		int hi = (tid == nthreads - 1) ? n - 1 : lo + points_per_thread;
		
		sumloc[tid].sum = 0.0;
		
		for (int i = lo; i <= hi; i++)
			sumloc[tid].sum += func (a + h * (i + 0.5));

		#pragma omp atomic 
		s += sumloc[tid].sum;		

	}
	t = omp_get_wtime() - t;
	return t;
}

double parallel(double a, double b, int n) 
{
	
	double h = (b - a) / n;
	double s = 0.0;
	double t = omp_get_wtime ();
	double sumloc[omp_get_max_threads ()];

	#pragma omp parallel
	{
		int nthreads = omp_get_num_threads ();
		int tid = omp_get_thread_num ();
		int points_per_thread = n / nthreads;
		int lo = tid * points_per_thread;
		int hi = (tid == nthreads - 1) ? n - 1 : lo + points_per_thread;
		
		sumloc[tid] = 0.0;
		
		for (int i = lo; i <= hi; i++)
			sumloc[tid] += func (a + h * (i + 0.5));
		

		#pragma omp atomic 
		s += sumloc[tid];
	
	}
	
	t = omp_get_wtime() - t;
	return t;

}

int main (int argc, char **argv)
{
	const double a = -4.0;
	const double b = 4.0;
	const int n = 10000000;
	
	printf ("Numerical integration: [%f, %f], n = %d\n", a, b, n);
	
	double t1 = serial (a, b ,n);
	double t2 = parallel (a, b, n);
	double t3 = parallel_fs (a, b, n);
	
	printf ("Elapsed time (sec.): %.12f (serial) | %.12f (parallel) | %.12f (parallel f.s.)\n", t1, t2, t3);
	printf ("Elapsed time (sec.): %.12f\n", t1/t2);
	printf ("Elapsed time (sec.): %.12f\n", t1/t3);

	return 0;
	
}
