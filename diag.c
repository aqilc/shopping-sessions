// diag: board-side profiler. Prints the machine facts, times /sess on dev and on a
// 20x tagged replica of dev built in /scratch, then exits 1 so the dev check shows logs.
#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void cat(const char *p, const char *grep, int maxl) {
  FILE *f = fopen(p, "r"); char l[1024]; int n = 0;
  if (!f) { fprintf(stderr, "[%s: unreadable]\n", p); return; }
  while (fgets(l, sizeof l, f) && n < maxl)
    if (!grep || strstr(l, grep)) { fprintf(stderr, "%s: %s", p, l); n++; }
  fclose(f);
}

static void run(const char *data, const char *out, const char *ncpu, const char *tag) {
  mkdir(out, 0777);
  char e1[512], e2[512], e3[64];
  snprintf(e1, sizeof e1, "DATA_DIR=%s", data); snprintf(e2, sizeof e2, "OUTPUT_DIR=%s", out);
  snprintf(e3, sizeof e3, "NCPU=%s", ncpu);
  char *env[] = {e1, e2, e3, "SESS_PROF=1", NULL}, *argv[] = {"/sess", NULL};
  fprintf(stderr, "== run %s data=%s NCPU=%s\n", tag, data, ncpu);
  double t = now(); pid_t p = fork();
  if (p == 0) { execve("/sess", argv, env); _exit(127); }
  int st; waitpid(p, &st, 0);
  fprintf(stderr, "== %s wall %.1f ms exit %d\n", tag, (now() - t) * 1e3, WEXITSTATUS(st));
}

// Copy every CSV in src/sub to dst/sub K times, prefixing the first two fields with a
// per-copy tag (inside the quote if the field is quoted) so ids stay distinct per copy.
static void rep(const char *src, const char *dst, const char *sub, int K, int nf) {
  char sd[512], dd[512]; snprintf(sd, sizeof sd, "%s/%s", src, sub); snprintf(dd, sizeof dd, "%s/%s", dst, sub);
  mkdir(dd, 0777);
  DIR *d = opendir(sd); struct dirent *e;
  if (!d) return;
  while ((e = readdir(d))) {
    if (!strstr(e->d_name, ".csv")) continue;
    char p[1024]; snprintf(p, sizeof p, "%s/%s", sd, e->d_name);
    int fd = open(p, O_RDONLY); struct stat st; fstat(fd, &st);
    if (st.st_size == 0) { close(fd); continue; }
    char *m = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    for (int k = 0; k < K; k++) {
      char q[1024]; snprintf(q, sizeof q, "%s/k%02d-%s", dd, k, e->d_name);
      FILE *o = fopen(q, "w"); setvbuf(o, NULL, _IOFBF, 1 << 20);
      char tg[4] = {'a' + k / 26, 'a' + k % 26, '_', 0};
      const char *s = m, *end = m + st.st_size; int hdr = 1;
      while (s < end) {
        const char *nl = memchr(s, '\n', end - s); if (!nl) nl = end;
        if (hdr || k == 0) { fwrite(s, 1, nl - s, o); hdr = 0; }
        else {
          const char *c = s;
          for (int f = 0; f < nf && c <= nl; f++) {
            const char *fe = memchr(c, ',', nl - c); if (!fe) fe = nl;
            if (fe > c && *c == '"') { fputc('"', o); fputs(tg, o); fwrite(c + 1, 1, fe - c - 1, o); }
            else if (fe > c) { fputs(tg, o); fwrite(c, 1, fe - c, o); }
            if (fe < nl) fputc(',', o);
            c = fe + 1;
          }
          if (c < nl) fwrite(c, 1, nl - c, o);
        }
        if (nl < end) fputc('\n', o);
        s = nl + 1;
      }
      fclose(o);
    }
    munmap(m, st.st_size);
  }
  closedir(d);
}

static void cp(const char *src, const char *dst, const char *name) {
  char a[512], b[512]; snprintf(a, sizeof a, "%s/%s", src, name); snprintf(b, sizeof b, "%s/%s", dst, name);
  FILE *i = fopen(a, "r"), *o = fopen(b, "w"); char buf[65536]; size_t n;
  if (!i || !o) return;
  while ((n = fread(buf, 1, sizeof buf, i))) fwrite(buf, 1, n, o);
  fclose(i); fclose(o);
}

int main(void) {
  cpu_set_t cs; sched_getaffinity(0, sizeof cs, &cs);
  fprintf(stderr, "== affinity cpus %d, nproc_onln %ld, NCPU env %s\n", CPU_COUNT(&cs), sysconf(_SC_NPROCESSORS_ONLN), getenv("NCPU"));
  cat("/proc/cpuinfo", "model name", 1); cat("/proc/cpuinfo", "flags", 1);
  cat("/proc/meminfo", "MemTotal", 1);
  cat("/sys/kernel/mm/transparent_hugepage/enabled", NULL, 2);
  cat("/sys/kernel/mm/transparent_hugepage/defrag", NULL, 2);
  cat("/sys/fs/cgroup/cpu.max", NULL, 2); cat("/sys/fs/cgroup/cpuset.cpus.effective", NULL, 2);
  cat("/proc/mounts", NULL, 40);
  cat("/proc/version", NULL, 1);
  const char *dd = getenv("DATA_DIR"); if (!dd) dd = "/data";
  const char *sc = getenv("SCRATCH_DIR"); if (!sc) sc = "/scratch";
  const char *nc = getenv("NCPU"); if (!nc) nc = "8";
  char out[512], big[512], bout[512];
  snprintf(out, sizeof out, "%s/o-dev", sc); snprintf(big, sizeof big, "%s/big", sc); snprintf(bout, sizeof bout, "%s/o-big", sc);
  for (int i = 0; i < 3; i++) run(dd, out, nc, "dev");
  int K = getenv("DIAG_K") ? atoi(getenv("DIAG_K")) : 20;
  double t = now(); mkdir(big, 0777);
  rep(dd, big, "events", K, 2); rep(dd, big, "clicks", K, 2); rep(dd, big, "links", K, 2);
  cp(dd, big, "products.csv"); cp(dd, big, "campaigns.csv"); cp(dd, big, "manifest.json");
  fprintf(stderr, "== replica x%d built in %.1f ms\n", K, (now() - t) * 1e3);
  for (int i = 0; i < 3; i++) run(big, bout, nc, "big");
  run(big, bout, "4", "big-nt4");
  run(big, bout, "16", "big-nt16");
  // A real answer for the dev check too, then fail on purpose so logs are shown.
  run(dd, getenv("OUTPUT_DIR") ? getenv("OUTPUT_DIR") : "/output", nc, "final-dev");
  fprintf(stderr, "== diag done, exiting 1 on purpose\n");
  return 1;
}
