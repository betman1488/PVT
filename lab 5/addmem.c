#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

const float G = 6.67e-11;

typedef struct particle {
    float x, y, z;
} part;

double wtime()
{
    struct timeval t;
    gettimeofday (&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void calculate_forces(part *p, part *f[], float *m, int n)
{
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    for (int i = 0; i < n; i++) {
        f[tid][i].x = 0;
        f[tid][i].y = 0;
        f[tid][i].z = 0;
    }
    #pragma omp for schedule(dynamic, 8)
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            float dist = sqrtf(powf(p[i].x - p[j].x, 2) + 
                    powf(p[i].y - p[j].y, 2) + powf(p[i].z - p[j].z, 2));
            float mag = (G * m[i] * m[j]) / powf(dist, 2);
            part dir = {
                .x = p[j].x - p[i].x,
                .y = p[j].y - p[i].y,
                .z = p[j].z - p[i].z
            };
            f[tid][i].x += mag * dir.x / dist;
            f[tid][i].y += mag * dir.y / dist;
            f[tid][i].z += mag * dir.z / dist;
            f[tid][j].x -= mag * dir.x / dist;
            f[tid][j].y -= mag * dir.y / dist;
            f[tid][j].z -= mag * dir.z / dist;
        }
    }

    #pragma omp single
    {
        for (int i = 0; i < n; i++)
            for (int tid = 1; tid < nthreads; tid++) {
                f[0][i].x += f[tid][i].x;
                f[0][i].y += f[tid][i].y;
                f[0][i].z += f[tid][i].z;
            }
    }
}

void move_particles(part *p, part *f[], part *v, float *m, int n, double dt)
{
    #pragma omp for
    for (int i = 0; i < n; i++) {
        part dv = {
            .x = f[0][i].x / m[i] * dt,
            .y = f[0][i].y / m[i] * dt,
            .z = f[0][i].z / m[i] * dt,
        };
        part dp = {
            .x = (v[i].x + dv.x / 2) * dt,
            .x = (v[i].y + dv.y / 2) * dt,
            .x = (v[i].z + dv.z / 2) * dt,
        };
        v[i].x += dv.x;
        v[i].y += dv.y;
        v[i].z += dv.z;
        p[i].x += dp.x;
        p[i].y += dp.y;
        p[i].z += dp.z;
    }
}

int main(int argc, char **argv)
{
    double ttotal, tinit = 0, tforces = 0, tmove = 0;
    ttotal = wtime();
    int n = (argc > 1) ? atoi(argv[1]) : 100;
    char *filename = (argc> 2) ? argv[2] : NULL;

    tinit -= wtime();
    part *p = (part *) malloc(sizeof(*p) * n);
    part *f[omp_get_max_threads()]; 
    for (int i = 0; i < omp_get_max_threads(); i++)
        f[i] = malloc(sizeof(part) * n);
    part *v = (part *) malloc(sizeof(*v) * n);
    float *m = (float *) malloc(sizeof(*m) * n);
    #pragma omp for
    for (int i = 0; i < n; i++) {
        unsigned int seed = omp_get_thread_num();
        p[i].x = rand_r(&seed) / RAND_MAX - 0.5;
        p[i].y = rand_r(&seed) / RAND_MAX - 0.5;
        p[i].z = rand_r(&seed) / RAND_MAX - 0.5;
        v[i].x = rand_r(&seed) / RAND_MAX - 0.5;
        v[i].y = rand_r(&seed) / RAND_MAX - 0.5;
        v[i].z = rand_r(&seed) / RAND_MAX - 0.5;
        m[i] = rand_r(&seed) / RAND_MAX * 10 + 0.01;
    }
    tinit += wtime();
    double dt = 1e-5;
    for (double t = 0; t < 1; t += dt) {
        tforces -= wtime();
        calculate_forces(p, f, m, n);
        tforces += wtime();
        tmove -= wtime();
        move_particles(p, f, v, m, n, dt);
        tmove += wtime();
    }
    ttotal = wtime() - ttotal;
    printf("ttotal = %.6f, tinit = %.6f, tforces = %.6f, tmove = %.6f\n",
            ttotal, tinit, tforces, tmove);
    ttotal = wtime(), tinit = 0, tforces = 0, tmove = 0;
    if (filename) {
        FILE *fout = fopen(filename, "w");
        if (!fout) {
            fprintf(stderr, "Failed open file\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < n; i++)
            fprintf(fout, "%15f %15f %15f\n", p[i].x, p[i].y, p[i].z);
        fclose(fout);
    }

    free(m);
    free(v);
    free(f);
    free(p);

    return 0;
}
