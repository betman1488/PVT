#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>
#include <sys/time.h>

#define EPS 0.001
#define PI 3.14159265358979323846
#define IND(i, j) ((i) * nx + (j))

double
wtime ()
{
  struct timeval t;
  gettimeofday (&t, NULL);
  return (double) t.tv_sec + (double) t.tv_usec * 1E-6;
}

double
serial (int rows, int cols, const char *filename)
{
  double ttotal = -wtime ();
// Allocate memory for grids [0..ny-1][0..nx - 1]
  int ny = rows;
  int nx = cols;
  double talloc = -wtime ();
//double * local_grid = cmalloc(ny * nx, sizeof(*local_grid));
//double *local_newgrid = cmalloc(ny * nx, sizeof(*local_newgrid));
  double *local_grid = malloc (ny * nx * sizeof (*local_grid));
  double *local_newgrid = malloc (ny * nx * sizeof (*local_newgrid));
  talloc += wtime ();
  double tinit = -wtime ();
// Fill boundary points:
//-left and right borders are zero filled
//- top border:u(x, 0) = sin(pi * x)
//- bottom border:u(x, 1) = sin(pi * x) * exp(-pi)
  double dx = 1.0 / (nx - 1.0);
// Initialize top border: u(x, 0) = sin(pi * x)
  for (int j = 0; j < nx; j++)
    {
      int ind = IND (0, j);
      local_newgrid[ind] = local_grid[ind] = sin (PI * dx * j);
    }
  // Initialize bottom border: u(x, 1) = sin(pi * x) * exp(-pi)
  for (int j = 0; j < nx; j++)
    {
      int ind = IND (ny - 1, j);
      local_newgrid[ind] = local_grid[ind] = sin (PI * dx * j) * exp (-PI);
    }
  // Initialize inner cells (we can use calloc for zeroing)
  for (int i = 1; i < ny - 1; i++)
    {
      for (int j = 1; j < nx - 1; j++)
	{
	  local_newgrid[IND (i, j)] = 0.0;
	  local_grid[IND (i, j)] = 0.0;
	}
    }
  tinit += wtime ();
  int niters = 0;
  for (;;)
    {
      niters++;
      // Update interior points
      for (int i = 1; i < ny - 1; i++)
	{
	  for (int j = 1; j < nx - 1; j++)
	    {
	      local_newgrid[IND (i, j)] =
		(local_grid[IND (i - 1, j)] +
		 local_grid[IND (i + 1, j)] +
		 local_grid[IND (i, j - 1)] +
		 local_grid[IND (i, j + 1)]) * 0.25;
	    }
	}
      double maxdiff = -DBL_MAX;
      for (int i = 1; i < ny - 1; i++)
	{
// Check termination condition
	  for (int j = 1; j < nx - 1; j++)
	    {
	      int ind = IND (i, j);
	      maxdiff = fmax (maxdiff, fabs (local_grid[ind]
					     - local_newgrid[ind]));
	    }
	}
      double *p = local_grid;
// Swap grids (after terminationlocal_grid will contain result)
      local_grid = local_newgrid;
      local_newgrid = p;
      if (maxdiff < EPS)
	break;
    }
  ttotal += wtime ();
  /*printf("# Heat 2D (serial): grid: rows %d, cols %d\n", rows, cols);
     printf("# niters %d, total time (sec.): %.6f\n", niters, ttotal);
     printf("#talloc: %.6f,tinit: %.6f, titers: %.6f\n", talloc, tinit,
     ttotal - talloc - tinit); */
// Save grid
  if (filename)
    {
      FILE *fout = fopen (filename, "w");
      if (!fout)
	{
	  perror ("Can't open file");
	  exit (EXIT_FAILURE);
	}
      for (int i = 0; i < ny; i++)
	{
	  for (int j = 0; j < nx; j++)
	    fprintf (fout, "%.4f ", local_grid[IND (i, j)]);
	  fprintf (fout, "\n");
	}
      fclose (fout);
    }
  return ttotal;
}

double
parallel_v1 (int rows, int cols, const char *filename)
{
  double ttotal = -omp_get_wtime ();
  int ny = rows;
  int nx = cols;
  double talloc = -omp_get_wtime ();
//double * local_grid = cmalloc(ny * nx, sizeof(*local_grid));
//double *local_newgrid = cmalloc(ny * nx, sizeof(*local_newgrid));
  double *local_grid = malloc (ny * nx * sizeof (*local_grid));
  double *local_newgrid = malloc (ny * nx * sizeof (*local_newgrid));
  talloc += omp_get_wtime ();
  double tinit = -omp_get_wtime ();
// Fill boundary points:
//-left and right borders are zero filled
//- top border:u(x, 0) = sin(pi * x)
//- bottom border:u(x, 1) = sin(pi * x) * exp(-pi)
  double dx = 1.0 / (nx - 1.0);
// Initialize top border: u(x, 0) = sin(pi * x)
#pragma omp parallel for
  for (int j = 0; j < nx; j++)
    {
      int ind = IND (0, j);
      local_newgrid[ind] = local_grid[ind] = sin (PI * dx * j);
    }
// Initialize bottom border: u(x, 1) = sin(pi * x) * exp(-pi)
#pragma omp parallel for
  for (int j = 0; j < nx; j++)
    {
      int ind = IND (ny - 1, j);
      local_newgrid[ind] = local_grid[ind] = sin (PI * dx * j) * exp (-PI);
    }
#pragma omp parallel for
  for (int i = 1; i < ny - 1; i++)
    {
      for (int j = 1; j < nx - 1; j++)
	{
	  local_newgrid[IND (i, j)] = 0.0;
	  local_grid[IND (i, j)] = 0.0;
	}
    }
  tinit += omp_get_wtime ();
  int niters = 0;
  for (;;)
    {
      niters++;
#pragma omp parallel for
      for (int i = 1; i < ny - 1; i++)
	{
	  for (int j = 1; j < nx - 1; j++)
	    {
	      local_newgrid[IND (i, j)] =
		(local_grid[IND (i - 1, j)] + local_grid[IND (i + 1, j)] +
		 local_grid[IND (i, j - 1)] +
		 local_grid[IND (i, j + 1)]) * 0.25;
	    }
	}
      double maxdiff = -DBL_MAX;
#pragma omp parallel for reduction (max:maxdiff)
      for (int i = 1; i < ny - 1; i++)
	{
	  for (int j = 1; j < nx - 1; j++)
	    {
	      int ind = IND (i, j);
	      maxdiff = fmax (maxdiff, fabs (local_grid[ind]
					     - local_newgrid[ind]));
	    }
	}
      double *p = local_grid;
      local_grid = local_newgrid;
      local_newgrid = p;
      if (maxdiff < EPS)
	break;
    }
  ttotal += omp_get_wtime ();
  /*printf("# Heat 2D (parallel): grid: rows %d, cols %d\n", rows, cols);
     printf("# niters %d, total time (sec.): %.6f\n", niters, ttotal);
     printf("#talloc: %.6f,tinit: %.6f, titers: %.6f\n", talloc, tinit,
     ttotal - talloc - tinit); */
// Save grid
  if (filename)
    {
      FILE *fout = fopen (filename, "w");
      if (!fout)
	{
	  perror ("Can't open file");
	  exit (EXIT_FAILURE);
	}
      for (int i = 0; i < ny; i++)
	{
	  for (int j = 0; j < nx; j++)
	    fprintf (fout, "%.4f ", local_grid[IND (i, j)]);
	  fprintf (fout, "\n");
	}
      fclose (fout);
    }
  return ttotal;
}

double
parallel_v2 (int rows, int cols, const char *filename)
{
  double ttotal = -omp_get_wtime ();
  int ny = rows;
  int nx = cols;
  double talloc = -omp_get_wtime ();
//double * local_grid = cmalloc(ny * nx, sizeof(*local_grid));
//double *local_newgrid = cmalloc(ny * nx, sizeof(*local_newgrid));
  double *local_grid = malloc (ny * nx * sizeof (*local_grid));
  double *local_newgrid = malloc (ny * nx * sizeof (*local_newgrid));
  talloc += omp_get_wtime ();
  double tinit = -omp_get_wtime ();
// Fill boundary points:
//-left and right borders are zero filled
//- top border:u(x, 0) = sin(pi * x)
//- bottom border:u(x, 1) = sin(pi * x) * exp(-pi)
  double dx = 1.0 / (nx - 1.0);
// Initialize top border: u(x, 0) = sin(pi * x)
#pragma omp parallel for
  for (int j = 0; j < nx; j++)
    {
      int ind = IND (0, j);
      local_newgrid[ind] = local_grid[ind] = sin (PI * dx * j);
    }
// Initialize bottom border: u(x, 1) = sin(pi * x) * exp(-pi)
#pragma omp parallel for
  for (int j = 0; j < nx; j++)
    {
      int ind = IND (ny - 1, j);
      local_newgrid[ind] = local_grid[ind] = sin (PI * dx * j) * exp (-PI);
    }
#pragma omp parallel for
  for (int i = 1; i < ny - 1; i++)
    {
      for (int j = 1; j < nx - 1; j++)
	{
	  local_newgrid[IND (i, j)] = 0.0;
	  local_grid[IND (i, j)] = 0.0;
	}
    }
  tinit += omp_get_wtime ();

  double maxdiff;
#pragma omp parallel
  {
// Thread-private copy of shared objects
    double *grid = local_grid;
    double *newgrid = local_newgrid;
    int niters = 0;
    for (;;)
      {
#pragma omp barrier
// All threads finished to check break condition
	maxdiff = -DBL_MAX;
#pragma omp barrier
// All threads updated maxdiff and ready to start reduction
#pragma omp for reduction (max:maxdiff)
	for (int i = 1; i < ny - 1; i++)
	  {
	    for (int j = 1; j < nx - 1; j++)
	      {
		int ind = IND (i, j);
		newgrid[ind] =
		  (grid[IND (i
			     -
			     1, j)] + grid[IND (i + 1, j)] +
		   grid[IND (i, j - 1)] + grid[IND (i, j + 1)]) * 0.25;
		maxdiff = fmax (maxdiff, fabs (grid[ind] - newgrid[ind]));
	      }
	  }
	double *p = grid;
// Swap grids (after termination grid will contain result)
	grid = newgrid;
	newgrid = p;
	niters++;
	if (maxdiff < EPS)
	  break;
      }
// for iters
#pragma omp barrier
#pragma omp master
    {
      ttotal += omp_get_wtime ();
      /*printf("# Heat 2D (OMP %d): grid: rows %d, cols %d\n",
         omp_get_num_threads(), rows, cols);
         printf("# niters %d, total time (sec.): %.6f\n", niters, ttotal);
         printf("#talloc: %.6f,tinit: %.6f, titers: %.6f\n", talloc, tinit,
         ttotal - talloc - tinit); */
// Restore shared objects
      local_grid = grid;
    }
  }

  if (filename)
    {
      FILE *fout = fopen (filename, "w");
      if (!fout)
	{
	  perror ("Can't open file");
	  exit (EXIT_FAILURE);
	}
      for (int i = 0; i < ny; i++)
	{
	  for (int j = 0; j < nx; j++)
	    fprintf (fout, "%.4f ", local_grid[IND (i, j)]);
	  fprintf (fout, "\n");
	}
      fclose (fout);
    }
  return ttotal;
}

int
main (int argc, char *argv[])
{
  int rows = (argc > 1) ? atoi (argv[1]) : 100;
  int cols = (argc > 2) ? atoi (argv[2]) : 100;
  const char *filename = (argc > 3) ? argv[3] : NULL;
  if (cols < 1 || rows < 1)
    {
      fprintf (stderr, "Invalid size of grid: rows %d, cols %d\n", rows,
	       cols);
      exit (EXIT_FAILURE);
    }
  double t1 = serial (rows, cols, filename);
  double t2 = parallel_v1 (rows, cols, filename);
  double t3 = parallel_v2 (rows, cols, filename);
  printf ("s(n) v1 = %.6f \ns(n) v2 = %.6f \n", t1 / t2, t1 / t3);
  return 0;
}
