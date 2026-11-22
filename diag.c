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
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static char SUM[16384]; static int SL;
#define SUMF(...) (SL += snprintf(SUM + SL, sizeof SUM - SL, __VA_ARGS__))

static void cat(const char *p, const char *grep, int maxl) {
  FILE *f = fopen(p, "r"); char l[4096]; int n = 0;
  if (!f) { SUMF("[%s unreadable]\n", p); return; }
  while (fgets(l, sizeof l, f) && n < maxl)
    if (!grep || strstr(l, grep)) { const char *b = strrchr(p, '/'); SUMF("%s: %.150s", b ? b + 1 : p, l); if (!strchr(l, '\n') || strlen(l) > 150) SUMF("\n"); n++; }
  fclose(f);
}

static int RUNI, XCAP;
// Per run: wall from fork to reap, phase deltas between the main stamps only (sub-timer lines are skipped,
// so a phase's time is never hidden behind its own sub-stamps), start (fork -> main's T0), exit (last stamp -> reap),
// user/sys cpu, max rss, minor faults.
enum { NPH = 8 };
typedef struct { double w, ph[NPH], st, ex, ut, sy, rss, flt; int x; } R;
static const char *PHN[NPH] = {"small", "links", "map", "parse", "dedup", "sess", "write", "release"};
static R run(const char *bin, const char *data, const char *ncpu) {
  char e1[512], e2[512], e3[64], lf[600], out[600];
  const char *sc = getenv("SCRATCH_DIR"); if (!sc) sc = "/scratch";
  snprintf(out, sizeof out, "%s/o%02d", sc, RUNI); mkdir(out, 0777);
  snprintf(e1, sizeof e1, "DATA_DIR=%s", data); snprintf(e2, sizeof e2, "OUTPUT_DIR=%s", out);
  snprintf(e3, sizeof e3, "NCPU=%s", ncpu);
  snprintf(lf, sizeof lf, "%s/run%02d.log", sc, RUNI++);
  char *env[] = {e1, e2, e3, "SESS_PROF=1", NULL}, *argv[] = {(char *)bin, NULL};
  double t = now(); pid_t p = fork();
  if (p == 0) { int fd = open(lf, O_WRONLY | O_CREAT | O_TRUNC, 0666); dup2(fd, 2); execve(bin, argv, env); _exit(127); }
  int st; struct rusage ru; wait4(p, &st, 0, &ru);
  R r = {0}; r.w = (now() - t) * 1e3; r.x = WIFEXITED(st) ? WEXITSTATUS(st) : -WTERMSIG(st);
  r.ut = ru.ru_utime.tv_sec * 1e3 + ru.ru_utime.tv_usec * 1e-3; r.sy = ru.ru_stime.tv_sec * 1e3 + ru.ru_stime.tv_usec * 1e-3;
  r.rss = ru.ru_maxrss / 1024.0; r.flt = ru.ru_minflt / 1000.0;
  FILE *f = fopen(lf, "r"); char l[256], nm[64]; double v, prev = 0, last = 0;
  while (f && fgets(l, sizeof l, f)) {
    if (XCAP && !strncmp(l, "X:", 2)) { const char *b = strrchr(bin, '/'); SUMF("%s %.200s", b ? b + 1 : bin, l); if (!strchr(l, '\n')) SUMF("\n"); continue; }
    if (l[0] == ' ') continue; // sub-timer
    if (sscanf(l, "%63s %lf ms", nm, &v) != 2) continue;
    if (!strcmp(nm, "t0abs")) { r.st = v - t * 1e3; continue; }
    for (int i = 0; i < NPH; i++) if (!strcmp(nm, PHN[i])) { r.ph[i] = v - prev; prev = last = v; }
  }
  if (f) fclose(f);
  r.ex = r.w - r.st - last;
  return r;
}
static int dcmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double med(double *v, int n) { double t[16]; memcpy(t, v, n * sizeof *t); qsort(t, n, sizeof *t, dcmp); return t[n / 2]; }

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
  SUMF("aff %d onln %ld NCPU %s\n", CPU_COUNT(&cs), sysconf(_SC_NPROCESSORS_ONLN), getenv("NCPU"));
  cat("/sys/fs/cgroup/memory.max", NULL, 1);
  const char *dd = getenv("DATA_DIR"); if (!dd) dd = "/data";
  const char *sc = getenv("SCRATCH_DIR"); if (!sc) sc = "/scratch";
  const char *nc = getenv("NCPU"); if (!nc) nc = "8";
  char big[512]; snprintf(big, sizeof big, "%s/big", sc);
  R d = run("/v/base", dd, nc); SUMF("dev base w%.0f x%d st%.1f ex%.1f pa%.0f\n", d.w, d.x, d.st, d.ex, d.ph[3]);
  int K = getenv("DIAG_K") ? atoi(getenv("DIAG_K")) : 60;
  { // cap K so the replica fits in 80% of /scratch free space
    long long src = 0; const char *subs[] = {"events", "clicks", "links"};
    for (int i = 0; i < 3; i++) { char sd[512]; snprintf(sd, sizeof sd, "%s/%s", dd, subs[i]); DIR *d = opendir(sd); struct dirent *e;
      while (d && (e = readdir(d))) { char p[1024]; struct stat st; snprintf(p, sizeof p, "%s/%s", sd, e->d_name); if (!stat(p, &st) && S_ISREG(st.st_mode)) src += st.st_size; }
      if (d) closedir(d); }
    struct statvfs v; long long fr = statvfs(sc, &v) ? -1 : (long long)v.f_bavail * v.f_frsize;
    int kmax = src > 0 && fr > 0 ? (int)(fr * 0.8 / (src * 1.08)) : K;
    SUMF("src %lld MB scratch free %lld MB kmax %d\n", src >> 20, fr >> 20, kmax);
    if (K > kmax) K = kmax;
  }
  double t = now(); mkdir(big, 0777);
#ifdef GEN_E
  { // generator data (real skew: full-size whales, shared-device persons, chains), not a replica
    char ev[32], cl[32], sd[32]; int ge = getenv("DIAG_GEN") ? atoi(getenv("DIAG_GEN")) : GEN_E; snprintf(ev, sizeof ev, "%d", ge); snprintf(cl, sizeof cl, "%d", ge / 4); snprintf(sd, sizeof sd, "%d", GEN_SEED);
    char *argv[] = {"/gen", "--seed", sd, "--events", ev, "--clicks", cl, "--output", big, "--threads", "8", NULL};
    pid_t p = fork(); if (p == 0) { int fd = open("/dev/null", O_WRONLY); dup2(fd, 1); execv("/gen", argv); _exit(127); }
    int st; waitpid(p, &st, 0); (void)K;
    SUMF("gen seed %d events %d x%d %.0f ms\n", GEN_SEED, ge, WEXITSTATUS(st), (now() - t) * 1e3);
  }
#else
  rep(dd, big, "events", K, 2); rep(dd, big, "clicks", K, 2); rep(dd, big, "links", K, 2);
  cp(dd, big, "products.csv"); cp(dd, big, "campaigns.csv"); cp(dd, big, "manifest.json");
  SUMF("replica x%d %.0f ms\n", K, (now() - t) * 1e3);
#endif
  static const char *V[] = {"base", "sa", "sak4", "sak4w", "saw", "p4k4", "sabh", "sa16", "nost", "base2"};
  enum { NV = 10, NR = 6 };
  static R res[NV][NR];
  run("/v/base", big, nc); // warm-up, discarded
  for (int r = 0; r < NR; r++)
    for (int v = 0; v < NV; v++) { char b[64]; snprintf(b, sizeof b, "/v/%s", V[v]); XCAP = 0; res[v][r] = run(b, big, nc); XCAP = 0; }
  for (int v = 0; v < NV; v++) {
    R *x = res[v]; double w[NR], q[NR], mn = 1e9;
    for (int r = 0; r < NR; r++) { w[r] = x[r].w; if (w[r] < mn) mn = w[r]; }
    int xx = 0; for (int r = 0; r < NR; r++) xx |= x[r].x;
    SUMF("%s w%.0f min%.0f x%d", V[v], med(w, NR), mn, xx);
#define MF(fld, fmt, lab) do { for (int r = 0; r < NR; r++) q[r] = x[r].fld; SUMF(" " lab fmt, med(q, NR)); } while (0)
    for (int i = 0; i < NPH; i++) { for (int r = 0; r < NR; r++) q[r] = x[r].ph[i]; SUMF(" %.2s%.0f", PHN[i], med(q, NR)); }
    MF(st, "%.1f", "st"); MF(ex, "%.1f", "ex"); MF(ut, "%.0f", "u"); MF(sy, "%.0f", "s"); MF(rss, "%.0f", "M"); MF(flt, "%.0f", "f");
    SUMF("\n");
  }
  // A real answer for the dev check too, then fail on purpose so logs are shown.
  const char *od = getenv("OUTPUT_DIR"); if (!od) od = "/output";
  { char e1[512], e2[512], e3[64]; snprintf(e1, sizeof e1, "DATA_DIR=%s", dd); snprintf(e2, sizeof e2, "OUTPUT_DIR=%s", od); snprintf(e3, sizeof e3, "NCPU=%s", nc);
    char *env[] = {e1, e2, e3, NULL}, *argv[] = {"/v/base", NULL};
    pid_t p = fork(); if (p == 0) { execve("/v/base", argv, env); _exit(127); } int st; waitpid(p, &st, 0); SUMF("final-dev x%d\n", WEXITSTATUS(st)); }
  SUMF("diag done, exit 1 on purpose\n");
  fputs(SUM, stderr);
  return 1;
}
