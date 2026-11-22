// floor: board-side floors for the input scan, picked by argv[0] basename.
//  rd: 8 threads pread every CSV in 1 MB blocks into a private buffer and sum it
//  nl: rd + AVX2 masks for '\n' ',' '"' (popcounts), the cheapest possible tokenizer
//  mm: mmap every file, threads sum 2 MB ranges (page faults on tmpfs/ext4 cache)
//  wb: rd + each thread streams 32 B per line into a growing 2 MB-page arena (record write cost)
//  pw: run /v/base under perf counters (inherit), print X: lines; on failure print the errno
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double T0;
#define PH(n) fprintf(stderr, "%-10s %7.1f ms\n", n, (now() - T0) * 1e3)

typedef struct { int fd; long off, len; char *map; } Job;
static Job *J; static long NJ, CAP; static volatile long NEXT;
static int MODE; // 0 rd 1 nl 2 mm 3 wb
static uint64_t SINK[64 * 8];

static void addfile(const char *p) {
  int fd = open(p, O_RDONLY); if (fd < 0) return; struct stat st; fstat(fd, &st);
  long blk = MODE == 2 ? (2 << 20) : (1 << 20);
  char *m = NULL;
  if (MODE == 2 && st.st_size) m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  for (long o = 0; o < st.st_size; o += blk) {
    if (NJ == CAP) { CAP = CAP ? CAP * 2 : 4096; J = realloc(J, CAP * sizeof *J); }
    J[NJ++] = (Job){fd, o, st.st_size - o < blk ? st.st_size - o : blk, m};
  }
}
static void adddir(const char *dd, const char *sub) {
  char d[600]; snprintf(d, sizeof d, "%s/%s", dd, sub); DIR *x = opendir(d); struct dirent *e;
  while (x && (e = readdir(x))) if (strstr(e->d_name, ".csv")) { char p[1200]; snprintf(p, sizeof p, "%s/%s", d, e->d_name); addfile(p); }
  if (x) closedir(x);
}

static uint64_t scan_sum(const char *b, long n) {
  uint64_t s = 0; const uint64_t *w = (const uint64_t *)b; long k = n / 8;
  for (long i = 0; i < k; i++) s += w[i];
  return s;
}
static uint64_t scan_nl(const char *b, long n) {
  uint64_t c = 0; long i = 0;
#ifdef __AVX2__
  const __m256i nl = _mm256_set1_epi8('\n'), cm = _mm256_set1_epi8(','), q = _mm256_set1_epi8('"');
  for (; i + 32 <= n; i += 32) {
    __m256i v = _mm256_loadu_si256((const __m256i *)(b + i));
    uint32_t a = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, nl)), d = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, cm)),
             e = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, q));
    c += __builtin_popcount(a) + ((uint64_t)__builtin_popcount(d) << 20) + ((uint64_t)__builtin_popcount(e) << 40);
  }
#endif
  for (; i < n; i++) c += b[i] == '\n';
  return c;
}

static void *worker(void *arg) {
  long t = (long)arg; char *buf = aligned_alloc(4096, (1 << 20) + 64); uint64_t s = 0;
  char *ar = NULL; size_t ap = 0, acap = 0;
  if (MODE == 3) { acap = (size_t)2 << 30; ar = mmap(NULL, acap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0); madvise(ar, acap, MADV_HUGEPAGE); }
  for (;;) {
    long j = __atomic_fetch_add(&NEXT, 1, __ATOMIC_RELAXED); if (j >= NJ) break;
    Job *x = &J[j]; const char *b;
    if (MODE == 2) b = x->map + x->off;
    else { long g = 0; while (g < x->len) { ssize_t r = pread(x->fd, buf + g, x->len - g, x->off + g); if (r <= 0) break; g += r; } b = buf; }
    if (MODE == 1) s += scan_nl(b, x->len);
    else if (MODE == 3) {
      // one 32 B record per line, written at the line end
      long ls = 0;
      for (long i = 0; i < x->len; i++) if (b[i] == '\n') {
        if (ap + 32 <= acap) { uint64_t *r = (uint64_t *)(ar + ap); r[0] = i; r[1] = ls; r[2] = j; r[3] = s; ap += 32; }
        ls = i + 1;
      }
      s += ap;
    } else s += scan_sum(b, x->len);
  }
  SINK[t * 8] = s;
  return NULL;
}

static long pe(struct perf_event_attr *a) { return syscall(SYS_perf_event_open, a, 0, -1, -1, 0); }

int main(int argc, char **argv) {
  T0 = now();
  const char *nm = strrchr(argv[0], '/'); nm = nm ? nm + 1 : argv[0];
  const char *dd = getenv("DATA_DIR"); if (!dd) dd = "/data";
  int nt = getenv("NCPU") ? atoi(getenv("NCPU")) : 8; if (nt < 1 || nt > 64) nt = 8;
  if (!strcmp(nm, "pw")) {
    static const struct { uint32_t type; uint64_t cfg; const char *n; } C[] = {
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"},
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instr"},
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "llc_miss"},
      {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "dtlb_rmiss"},
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "br_miss"},
      {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "task_ms"},
    };
    enum { NC = sizeof C / sizeof *C }; long fd[NC];
    for (int i = 0; i < NC; i++) {
      struct perf_event_attr a; memset(&a, 0, sizeof a); a.size = sizeof a; a.type = C[i].type; a.config = C[i].cfg;
      a.inherit = 1; a.exclude_kernel = 1; a.exclude_hv = 1;
      fd[i] = pe(&a); if (fd[i] < 0) fprintf(stderr, "X: pe %s errno %d\n", C[i].n, errno);
    }
    pid_t p = fork();
    if (p == 0) { for (int i = 0; i < NC; i++) if (fd[i] >= 0) close(fd[i]); char *av[] = {"/v/base", NULL}; execv("/v/base", av); _exit(127); }
    int st; waitpid(p, &st, 0);
    fprintf(stderr, "X:");
    for (int i = 0; i < NC; i++) { uint64_t v = 0; if (fd[i] >= 0 && read(fd[i], &v, 8) == 8) fprintf(stderr, " %s %.1fM", C[i].n, i == NC - 1 ? v / 1e6 : v / 1e6); else fprintf(stderr, " %s -", C[i].n); }
    fprintf(stderr, "\n");
    return WIFEXITED(st) ? WEXITSTATUS(st) : 3;
  }
  MODE = !strcmp(nm, "nl") ? 1 : !strcmp(nm, "mm") ? 2 : !strcmp(nm, "wb") ? 3 : 0;
  adddir(dd, "events"); adddir(dd, "clicks"); adddir(dd, "links");
  PH("map");
  pthread_t th[64];
  for (long t = 1; t < nt; t++) pthread_create(&th[t], NULL, worker, (void *)t);
  worker(0);
  for (long t = 1; t < nt; t++) pthread_join(th[t], NULL);
  PH("parse");
  uint64_t s = 0; for (int t = 0; t < nt; t++) s += SINK[t * 8];
  fprintf(stderr, "X: %s jobs %ld sink %llx\n", nm, NJ, (unsigned long long)s);
  _exit(0);
}
