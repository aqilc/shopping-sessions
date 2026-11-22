// Shopping Sessions engine v3: one static binary, no deps.
// Pipeline: links (parallel union-find) -> parse events+clicks (SIMD tokenizer,
// person resolved at parse) -> dedup by event-id partition (marks dead payloads)
// -> sessionize by person partition -> per-thread Parquet.
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define INL static inline __attribute__((always_inline))

static void die(const char *m) { fprintf(stderr, "fatal: %s (%s)\n", m, strerror(errno)); _exit(2); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("oom"); return p; }
static void *xcalloc(size_t a, size_t b) { void *p = calloc(a ? a : 1, b ? b : 1); if (!p) die("oom"); return p; }
static void *xrealloc(void *p, size_t n) { p = realloc(p, n ? n : 1); if (!p) die("oom"); return p; }

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int PROF; static double T0;
#define PH(name) do { if (PROF) fprintf(stderr, "%-10s %7.1f ms\n", name, (now_s() - T0) * 1e3); } while (0)

// ---------- hashing / strings ----------
INL uint64_t ld8(const char *p) { uint64_t w; memcpy(&w, p, 8); return w; }
INL uint64_t mum(uint64_t a, uint64_t b) { __uint128_t r = (__uint128_t)a * b; return (uint64_t)r ^ (uint64_t)(r >> 64); }
#define HS0 0xa0761d6478bd642full
#define HS1 0xe7037ed1a0b428dbull
#define HS2 0x8ebc6af09c88c6e3ull
#define HS3 0x589965cc75374cc3ull
// Reads up to 7 bytes past p+n: every buffer we hash is padded.
INL uint64_t hstr(const char *p, uint32_t n) {
  uint64_t a, b;
  if (LIKELY(n <= 16)) {
    if (n >= 8) { a = ld8(p); b = ld8(p + n - 8); }
    else { a = n ? ld8(p) & (~0ull >> (64 - 8 * n)) : 0; b = 0; }
    uint64_t h = mum(a ^ HS1, b ^ HS0 ^ n);
    return mum(h ^ HS2, (uint64_t)n ^ HS3) | 1;
  }
  uint64_t h = HS0 ^ n;
  while (n > 16) { h = mum(ld8(p) ^ HS1, ld8(p + 8) ^ h); p += 16; n -= 16; }
  a = ld8(p + n - 16 + 0); b = ld8(p + n - 8);
  h = mum(a ^ HS2, b ^ h);
  return mum(h ^ HS3, HS1) | 1;
}
INL int str_eq(const char *a, uint32_t an, const char *b, uint32_t bn) { return an == bn && memcmp(a, b, an) == 0; }
static inline int str_cmp(const char *a, uint32_t an, const char *b, uint32_t bn) {
  uint32_t m = an < bn ? an : bn;
  int c = memcmp(a, b, m);
  if (c) return c;
  return an < bn ? -1 : (an > bn ? 1 : 0);
}
// string ref: 48-bit pointer | len << 48
INL uint64_t mkref(const char *p, uint32_t n) { return (uint64_t)(uintptr_t)p | ((uint64_t)n << 48); }
INL const char *refp(uint64_t r) { return (const char *)(uintptr_t)(r & 0xFFFFFFFFFFFFull); }
INL uint32_t refn(uint64_t r) { return (uint32_t)(r >> 48); }

// ---------- files ----------
typedef struct { const char *p; size_t n; } MFile;
#define PAD 4096
// Map a file with at least PAD readable zero bytes after its end.
static MFile map_file(const char *path) {
  MFile f = {0, 0};
  int fd = open(path, O_RDONLY);
  if (fd < 0) die(path);
  struct stat st;
  if (fstat(fd, &st) < 0) die("fstat");
  f.n = (size_t)st.st_size;
  size_t pg = 4096;
  size_t span = ((f.n + pg - 1) / pg) * pg + PAD;
  char *r = mmap(NULL, span, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (r == MAP_FAILED) die("mmap reserve");
  if (f.n) {
    void *m = mmap(r, f.n, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
    if (m == MAP_FAILED) die("mmap");
  }
  f.p = r;
  close(fd);
  return f;
}

static int list_csv(const char *dir, char ***out) {
  DIR *d = opendir(dir);
  int n = 0, cap = 64;
  char **v = xmalloc(sizeof(char *) * cap);
  if (d) {
    struct dirent *e;
    while ((e = readdir(d))) {
      size_t l = strlen(e->d_name);
      if (l < 4 || strcmp(e->d_name + l - 4, ".csv") != 0) continue;
      char *path = xmalloc(strlen(dir) + l + 2);
      sprintf(path, "%s/%s", dir, e->d_name);
      if (e->d_type != DT_REG) {
        struct stat st;
        if (stat(path, &st) < 0 || !S_ISREG(st.st_mode)) { free(path); continue; }
      }
      if (n == cap) { cap *= 2; v = xrealloc(v, sizeof(char *) * cap); }
      v[n++] = path;
    }
    closedir(d);
  }
  *out = v;
  return n;
}

// ---------- SIMD masks over 64 bytes ----------
// Bit i of nl/cm/q is set when p[i] is '\n' / ',' / '"'.
#if defined(__AVX2__)
INL uint64_t eq64(__m256i a, __m256i b, char c) {
  __m256i v = _mm256_set1_epi8(c);
  uint32_t lo = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, v));
  uint32_t hi = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, v));
  return (uint64_t)lo | ((uint64_t)hi << 32);
}
INL void blk_masks(const char *p, uint64_t *nl, uint64_t *cm, uint64_t *q) {
  __m256i a = _mm256_loadu_si256((const __m256i *)p), b = _mm256_loadu_si256((const __m256i *)(p + 32));
  *nl = eq64(a, b, '\n');
  *cm = eq64(a, b, ',');
  *q = eq64(a, b, '"');
}
#elif defined(__aarch64__)
// Compare lanes ANDed with their bit; two pairwise-add rounds fold 64 lanes into 8 mask bytes (doubled),
// a third round packs two masks into one vector's 64-bit lanes.
INL uint8x16_t fold2(const uint8x16_t *v, uint8x16_t c, uint8x16_t bits) {
  uint8x16_t t0 = vandq_u8(vceqq_u8(v[0], c), bits), t1 = vandq_u8(vceqq_u8(v[1], c), bits);
  uint8x16_t t2 = vandq_u8(vceqq_u8(v[2], c), bits), t3 = vandq_u8(vceqq_u8(v[3], c), bits);
  return vpaddq_u8(vpaddq_u8(t0, t1), vpaddq_u8(t2, t3));
}
INL void blk_masks(const char *p, uint64_t *nl, uint64_t *cm, uint64_t *q) {
  static const uint8_t bitv[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  uint8x16_t bits = vld1q_u8(bitv);
  uint8x16_t v[4];
  for (int i = 0; i < 4; i++) v[i] = vld1q_u8((const uint8_t *)p + 16 * i);
  uint8x16_t n = fold2(v, vdupq_n_u8('\n'), bits), c = fold2(v, vdupq_n_u8(','), bits), d = fold2(v, vdupq_n_u8('"'), bits);
  uint64x2_t nc = vreinterpretq_u64_u8(vpaddq_u8(n, c));
  *nl = vgetq_lane_u64(nc, 0);
  *cm = vgetq_lane_u64(nc, 1);
  *q = vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(d, d)), 0);
}
#else
INL void blk_masks(const char *p, uint64_t *nl, uint64_t *cm, uint64_t *q) {
  uint64_t a = 0, b = 0, c = 0;
  for (int i = 0; i < 64; i++) {
    a |= (uint64_t)(p[i] == '\n') << i;
    b |= (uint64_t)(p[i] == ',') << i;
    c |= (uint64_t)(p[i] == '"') << i;
  }
  *nl = a; *cm = b; *q = c;
}
#endif

// ---------- timestamps ----------
#define BADTS INT64_MIN
static inline int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = (unsigned)(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + (int64_t)doe - 719468;
}
INL int isdig(char c) { return (unsigned)(c - '0') <= 9; }

// Days since the epoch at Jan 1 of 1900..2300 (one past the range gives the last year's length).
#define TS_Y0 1900
#define TS_NY 400
static int32_t ts_ystart[TS_NY + 1];
// Days before month m (1..12) in a common / leap year; month 0 and 13..15 are padding.
static const int16_t ts_moff[2][16] = {{0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},
                                       {0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335}};
static const uint8_t ts_mdays[2][16] = {{0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
                                        {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};
__attribute__((constructor)) static void ts_init(void) {
  for (int i = 0; i <= TS_NY; i++) ts_ystart[i] = (int32_t)days_from_civil(TS_Y0 + i, 1, 1);
}

#define ONES 0x0101010101010101ULL
// Bytes of x (each 0..255) on lanes in m that are not 0..9.
INL uint64_t ts_nondig(uint64_t x, uint64_t m) { return ((x + 0x76 * ONES) | x) & m & (0x80 * ONES); }
// Per-lane 10*x[i] + x[i+1], for x holding digit values 0..9 (no lane overflows).
INL uint64_t ts_pairs(uint64_t x) { return x * 10 + (x >> 8); }
#define LANE(w, i) ((unsigned)((w) >> (8 * (i))) & 0xFF)

// "YYYY-MM-DDTHH:MM:SS" (19 bytes at p, ss already validated by the caller and passed in) -> ms, or BADTS.
// Reads only p[0..15].
INL int64_t ts_cal(const char *p, unsigned ss, uint64_t bad) {
  // Expected bytes with '0' on digit lanes: "0000-00-" and "00T00:00".
  const uint64_t e0 = 0x2D30302D30303030ULL, e1 = 0x30303A3030543030ULL;
  const uint64_t d0 = 0x00FFFF00FFFFFFFFULL, d1 = 0xFFFF00FFFF00FFFFULL;
  uint64_t x0 = ld8(p) ^ e0, x1 = ld8(p + 8) ^ e1;
  bad |= (x0 & ~d0) | (x1 & ~d1) | ts_nondig(x0, d0) | ts_nondig(x1, d1);
  if (UNLIKELY(bad)) return BADTS;
  uint64_t q0 = ts_pairs(x0), q1 = ts_pairs(x1);
  unsigned y = LANE(q0, 0) * 100 + LANE(q0, 2), mo = LANE(q0, 5);
  unsigned da = LANE(q1, 0), hh = LANE(q1, 3), mi = LANE(q1, 6);
  if (UNLIKELY((mo - 1 > 11) | (da == 0) | (hh > 23) | (mi > 59))) return BADTS;
  unsigned yi = y - TS_Y0;
  int64_t days;
  unsigned leap;
  if (LIKELY(yi < TS_NY)) {
    leap = (unsigned)(ts_ystart[yi + 1] - ts_ystart[yi]) - 365;
    days = ts_ystart[yi] + ts_moff[leap][mo] + (int64_t)da - 1;
  } else {
    leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    days = days_from_civil(y, mo, da);
  }
  if (UNLIKELY(da > ts_mdays[leap][mo])) return BADTS;
  return days * 86400000LL + (int64_t)(hh * 3600 + mi * 60 + ss) * 1000;
}

// ingest_ts is always valid F1 (rules.md), so it needs no checks: "YYYY-MM-DDTHH:MM:SSZ" -> epoch seconds.
INL int64_t ingest_s(const char *p) {
  uint64_t q0 = ts_pairs(ld8(p) ^ 0x2D30302D30303030ULL), q1 = ts_pairs(ld8(p + 8) ^ 0x30303A3030543030ULL);
  unsigned y = LANE(q0, 0) * 100 + LANE(q0, 2), mo = LANE(q0, 5);
  unsigned da = LANE(q1, 0), hh = LANE(q1, 3), mi = LANE(q1, 6);
  unsigned ss = (unsigned)(p[17] - '0') * 10 + (unsigned)(p[18] - '0');
  unsigned yi = y - TS_Y0;
  int64_t days;
  if (LIKELY(yi < TS_NY)) {
    unsigned leap = (unsigned)(ts_ystart[yi + 1] - ts_ystart[yi]) - 365;
    days = ts_ystart[yi] + ts_moff[leap][mo] + (int64_t)da - 1;
  } else days = days_from_civil(y, mo, da);
  return days * 86400 + hh * 3600 + mi * 60 + ss;
}

// Lemire: 8 ASCII digits (first digit in the low byte) -> value. Zero bytes count as '0'.
INL uint64_t ts_parse8(uint64_t w) {
  w = ((w & 0x0F0F0F0F0F0F0F0FULL) * 2561) >> 8;
  w = ((w & 0x00FF00FF00FF00FFULL) * 6553601) >> 16;
  return ((w & 0x0000FFFF0000FFFFULL) * 42949672960001ULL) >> 32;
}
// The 8 bytes ending rem bytes into p (n >= 8), with bytes before p as zeros.
INL uint64_t ts_chunk(const char *p, uint32_t rem) {
  return rem >= 8 ? ld8(p + rem - 8) : ld8(p) << (8 * (8 - rem));
}

// Reads only p[0..n-1].
static int64_t parse_ts(const char *p, uint32_t n) {
  if (n == 20) {
    // ":SSZ"
    uint32_t t;
    memcpy(&t, p + 16, 4);
    uint32_t x = t ^ 0x5A30303AU;
    unsigned ss = ((x >> 8) & 0xFF) * 10 + ((x >> 16) & 0xFF);
    uint64_t bad = (x & 0xFF0000FFU) | ts_nondig(x, 0x00FFFF00U) | (ss > 59);
    return ts_cal(p, ss, bad);
  }
  if (n == 25) {
    // "SS+HH:MM" at 17; the sign lane is accepted as '+' (x = 0) or '-' (x = 6).
    uint64_t x = ld8(p + 17) ^ 0x30303A30302B3030ULL;
    const uint64_t dm = 0xFFFF00FFFF00FFFFULL;
    unsigned sx = LANE(x, 2);
    uint64_t bad = (x & 0x0000FF0000000000ULL) | ts_nondig(x, dm) | (sx != 0 && sx != 6) | (p[16] != ':');
    uint64_t q = ts_pairs(x & dm);
    unsigned ss = LANE(q, 0), oh = LANE(q, 3), om = LANE(q, 6);
    bad |= (ss > 59) | (oh > 23) | (om > 59);
    int64_t c = ts_cal(p, ss, bad);
    if (c == BADTS) return BADTS;
    int64_t off = (int64_t)(oh * 60 + om) * 60000;
    return sx ? c + off : c - off;
  }
  if (n == 0) return BADTS;
  if (n < 8) {
    int64_t v = 0;
    for (uint32_t i = 0; i < n; i++) {
      if (!isdig(p[i])) return BADTS;
      v = v * 10 + (p[i] - '0');
    }
    return v;
  }
  uint64_t bad = 0;
  for (uint32_t i = 0; i + 8 < n; i += 8) bad |= ts_nondig(ld8(p + i) ^ 0x30 * ONES, ~0ULL);
  bad |= ts_nondig(ld8(p + n - 8) ^ 0x30 * ONES, ~0ULL);
  if (bad) return BADTS;
  uint32_t m = n;
  if (UNLIKELY(n > 18)) {
    uint32_t j = 0;
    while (j < n && p[j] == '0') j++;
    m = n - j;
    if (m > 18) return INT64_MAX; // valid form, far outside any window
  }
  uint64_t v = ts_parse8(ld8(p + n - 8));
  if (m > 8) v += ts_parse8(ts_chunk(p, n - 8)) * 100000000ULL;
  if (m > 16) v += ts_parse8(ts_chunk(p, n - 16)) * 10000000000000000ULL;
  return (int64_t)v;
}

// price: digits only, 1..10,000,000; returns 0 if bad
INL uint32_t parse_price(const char *p, uint32_t n) {
  if (n == 0) return 0;
  uint32_t j = 0;
  for (uint32_t i = 0; i < n; i++) if (!isdig(p[i])) return 0;
  while (j < n && p[j] == '0') j++;
  if (n - j > 8) return 0;
  uint32_t v = 0;
  for (; j < n; j++) v = v * 10 + (uint32_t)(p[j] - '0');
  if (v < 1 || v > 10000000) return 0;
  return v;
}

// ---------- small string maps (products, campaigns) ----------
typedef struct { uint64_t h; const char *p; uint32_t n; int32_t v; } SSlot;
typedef struct { SSlot *s; uint64_t mask; } SMap;
static void smap_init(SMap *m, size_t want) {
  size_t cap = 16;
  while (cap < want * 2) cap <<= 1;
  m->s = xcalloc(cap, sizeof(SSlot));
  m->mask = cap - 1;
}
static SSlot *smap_put(SMap *m, const char *p, uint32_t n, int32_t v, int *created) {
  uint64_t h = hstr(p, n);
  uint64_t i = h & m->mask;
  for (;;) {
    SSlot *s = &m->s[i];
    if (!s->p) { s->h = h; s->p = p; s->n = n; s->v = v; *created = 1; return s; }
    if (s->h == h && str_eq(s->p, s->n, p, n)) { *created = 0; return s; }
    i = (i + 1) & m->mask;
  }
}
INL int32_t smap_get(const SMap *m, const char *p, uint32_t n) {
  uint64_t h = hstr(p, n);
  uint64_t i = h & m->mask;
  for (;;) {
    const SSlot *s = &m->s[i];
    if (!s->p) return -1;
    if (s->h == h && str_eq(s->p, s->n, p, n)) return s->v;
    i = (i + 1) & m->mask;
  }
}

// ---------- thread pool ----------
static int NT = 8;
typedef void (*task_fn)(int tid, long task, void *ctx);
typedef struct { task_fn fn; void *ctx; long ntask; const long *order; atomic_long next; } Pool;
typedef struct { Pool *p; int tid; } PArg;
static void *pool_main(void *a) {
  PArg *pa = a;
  Pool *p = pa->p;
  for (;;) {
    long t = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed);
    if (t >= p->ntask) break;
    p->fn(pa->tid, p->order ? p->order[t] : t, p->ctx);
  }
  return NULL;
}
static void run_par_o(long ntask, const long *order, task_fn fn, void *ctx) {
  Pool p; p.fn = fn; p.ctx = ctx; p.ntask = ntask; p.order = order; atomic_init(&p.next, 0);
  pthread_t th[64]; PArg args[64];
  for (int i = 0; i < NT; i++) { args[i].p = &p; args[i].tid = i; }
  for (int i = 1; i < NT; i++) pthread_create(&th[i], NULL, pool_main, &args[i]);
  pool_main(&args[0]);
  for (int i = 1; i < NT; i++) pthread_join(th[i], NULL);
}
static void run_par(long ntask, task_fn fn, void *ctx) { run_par_o(ntask, NULL, fn, ctx); }

// ---------- per-thread arenas ----------
#define ARENA_SZ (2ull << 30) // payload locs are 28-bit units of 8 bytes
typedef struct { char *base, *cur, *end; char pad[40]; } Arena;
static Arena *arenas; // [NT] record storage (addressable by loc)
static Arena *scratch; // [NT] per-task scratch, reset per task
static void arena_init(Arena *a, size_t sz) {
  char *m = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (m == MAP_FAILED) die("arena");
#ifdef MADV_HUGEPAGE
  madvise(m, sz, MADV_HUGEPAGE);
#endif
  a->base = a->cur = m; a->end = m + sz;
}
// Zeroed anonymous memory on huge pages (random-access tables: avoids a TLB miss per probe).
static void *hcalloc(size_t n) {
  n = (n + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
  char *m = mmap(NULL, n + (2u << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (m == MAP_FAILED) die("hcalloc");
  m = (char *)(((uintptr_t)m + (2u << 20) - 1) & ~(uintptr_t)((2u << 20) - 1));
#ifdef MADV_HUGEPAGE
  madvise(m, n, MADV_HUGEPAGE);
#endif
  return m;
}
INL void *aalloc(Arena *a, size_t n) {
  n = (n + 63) & ~(size_t)63;
  if (UNLIKELY(a->cur + n > a->end)) die("arena full");
  void *r = a->cur; a->cur += n; return r;
}

// Bucket: chain of blocks of fixed-size records.
typedef struct Blk { struct Blk *next; uint32_t n, cap; } Blk; // records follow (16-byte header)
typedef struct { char *cur, *end; Blk *head, *tail; } Bkt;
static __attribute__((noinline)) void bkt_grow(Arena *a, Bkt *b, uint32_t rs, uint32_t cap) {
  if (b->tail) b->tail->n = (uint32_t)((b->cur - (char *)(b->tail + 1)) / rs);
  Blk *k = aalloc(a, sizeof(Blk) + (size_t)rs * cap);
  k->next = NULL; k->n = 0; k->cap = cap;
  if (b->tail) b->tail->next = k; else b->head = k;
  b->tail = k;
  b->cur = (char *)(k + 1); b->end = b->cur + (size_t)rs * cap;
}
static void bkt_close(Bkt *b, uint32_t rs) {
  if (b->tail) b->tail->n = (uint32_t)((b->cur - (char *)(b->tail + 1)) / rs);
}

// ---------- globals ----------
static int64_t WS, WE, CWS; // event window [WS,WE), click window [CWS,WE)
static int64_t DAY0; static int32_t NDAYS;
static SMap prodmap; // product -> category index
static int unknown_cat;
static int ncat; static const char **catp; static uint32_t *catn;
static SMap campmap; // campaign -> idx
static int ncamp; static const char **camp_id; static uint32_t *camp_idn; static const char **chan; static uint32_t *chann;
static uint32_t PB, QB; static uint32_t NPART, NQART; // eid partitions 2^PB, person partitions 2^QB

// ---------- linked IDs ----------
// build table: key hash, string ref, union-find parent, best (name) slot
static _Atomic uint32_t *lpar; static _Atomic uint32_t *lbest;
static uint64_t lcap, lmask_b;
// lookup table: id hash -> person key
// Link ID table, open addressing on the ID hash: val is the ID's string ref while links load, its person key after.
typedef struct { _Atomic uint64_t key; uint64_t val; } LEnt;
static LEnt *ltab; static uint64_t *lname; // lname[root slot] = name ref
static int have_links;

#define LINKED (1ull << 63)
INL uint32_t qpart(uint64_t pkey) { return (uint32_t)((pkey >> 40) & (NQART - 1)); }

INL uint64_t person_of(uint64_t uh) {
  if (have_links) {
    uint64_t i = uh & lmask_b;
    for (;;) {
      const LEnt *e = &ltab[i];
      uint64_t k = atomic_load_explicit(&e->key, memory_order_relaxed);
      if (k == uh) return e->val;
      if (!k) break;
      i = (i + 1) & lmask_b;
    }
  }
  return uh & ~LINKED;
}

// ---------- records ----------
typedef struct { uint64_t h; uint32_t ingest; uint32_t loc; } KRec;     // dedup key, per eid partition
typedef struct { uint64_t pkey; uint64_t uref; uint32_t ts; uint32_t x; } PRec; // event payload, per person partition
typedef struct { uint64_t pkey; uint64_t cref; uint32_t ts; uint32_t camp; } CRec; // click, per person partition
// PRec.x: price (24 bits) | cat (5 bits) << 24 | purchase << 29 | dead << 31
#define X_PURCH (1u << 29)
#define X_DEAD (1u << 31)
#define LOC_NONE 0x0FFFFFFFu

static Bkt *kb; // [NT][NPART]
static Bkt *pb; // [NT][NQART]
static Bkt *cb; // [NT][NQART]
static uint32_t kcap0, pcap0, ccap0;

// ---------- chunks ----------
typedef struct { const char *base; size_t start, end, fsize; int kind; } Chunk;
static Chunk *chunks; static long nchunks, chcap;
static void add_chunks(MFile f, int kind, size_t csz) {
  if (!f.n) return;
  for (size_t s = 0; s < f.n; s += csz) {
    if (nchunks == chcap) { chcap = chcap ? chcap * 2 : 256; chunks = xrealloc(chunks, sizeof(Chunk) * chcap); }
    Chunk c = {f.p, s, s + csz < f.n ? s + csz : f.n, f.n, kind};
    chunks[nchunks++] = c;
  }
}
static const char *chunk_begin(const Chunk *c) {
  const char *b = c->base;
  const char *fend = b + c->fsize;
  const char *p;
  if (c->start == 0) { p = memchr(b, '\n', c->fsize); return p ? p + 1 : fend; } // skip header
  p = memchr(b + c->start - 1, '\n', c->fsize - (c->start - 1));
  return p ? p + 1 : fend;
}

// Tokenizer: calls LINE(ls, sep, nsep, nq) for each line starting in [begin, cend).
// sep[k] = position of the k-th separator (',' or the line end), nq = quotes in the line.
// Bytes past the file end are zero (map_file's PAD), so blocks need no end mask.
// Commas go to sep_ SEPW at a time, unconditionally: slots past min(ns, MAXF) are scratch, so no per-field branch.
// SEPW >= MAXF, so commas past the first SEPW of a block can only land past MAXF and need no store.
#define MAXF 8
#define SEPW 8
INL int tk_commas(const char **sep, int ns, const char *blk, uint64_t m) {
  int n = __builtin_popcountll(m);
  const char **d = sep + (ns < MAXF ? ns : MAXF);
  for (int j = 0; j < SEPW; j++) {
    d[j] = blk + __builtin_ctzll(m | (1ull << 63));
    m &= m - 1;
  }
  return ns + n;
}
#define TOKENIZE(c, LINE)                                                                         \
  do {                                                                                            \
    const char *fend_ = (c)->base + (c)->fsize;                                                   \
    const char *cend_ = (c)->base + (c)->end;                                                     \
    const char *ls_ = chunk_begin(c);                                                             \
    if (ls_ >= cend_) break;                                                                      \
    const char *sep_[MAXF + SEPW]; int ns_ = 0; uint32_t qacc_ = 0;                               \
    const char *blk_ = ls_;                                                                       \
    int done_ = 0;                                                                                \
    while (!done_) {                                                                              \
      uint64_t nl_, cm_, q_;                                                                      \
      blk_masks(blk_, &nl_, &cm_, &q_);                                                           \
      while (nl_) {                                                                               \
        int i_ = __builtin_ctzll(nl_);                                                            \
        uint64_t below_ = nl_ ^ (nl_ - 1);                                                        \
        nl_ &= nl_ - 1;                                                                           \
        ns_ = tk_commas(sep_, ns_, blk_, cm_ & below_);                                           \
        sep_[ns_ < MAXF ? ns_ : MAXF] = blk_ + i_;                                                \
        ns_++;                                                                                    \
        uint32_t nq_ = qacc_ + (uint32_t)__builtin_popcountll(q_ & below_);                       \
        cm_ &= ~below_; q_ &= ~below_; qacc_ = 0;                                                 \
        LINE(ls_, sep_, ns_, nq_);                                                                \
        ls_ = blk_ + i_ + 1; ns_ = 0;                                                             \
        if (ls_ >= cend_) { done_ = 1; break; }                                                   \
      }                                                                                           \
      if (done_) break;                                                                           \
      if (cm_) ns_ = tk_commas(sep_, ns_, blk_, cm_);                                             \
      qacc_ += (uint32_t)__builtin_popcountll(q_);                                                \
      blk_ += 64;                                                                                 \
      if (blk_ >= fend_) {                                                                        \
        if (ls_ < fend_) { sep_[ns_ < MAXF ? ns_ : MAXF] = fend_; ns_++; LINE(ls_, sep_, ns_, qacc_); } \
        break;                                                                                    \
      }                                                                                           \
    }                                                                                             \
  } while (0)

// Field k of a line: [fs, fe). Strips quotes; returns 0 if a quote is misplaced. nqf counts quotes used.
INL int field(const char *ls, const char *const *sep, int k, const char **fp, uint32_t *fl, uint32_t *nqf, uint32_t nq) {
  const char *s = k ? sep[k - 1] + 1 : ls;
  const char *e = sep[k];
  if (nq && e > s && *s == '"') {
    if (e - s < 2 || e[-1] != '"') return 0;
    s++; e--; *nqf += 2;
  }
  *fp = s; *fl = (uint32_t)(e - s);
  return 1;
}

// Delay line: person lookups miss cache, so each is resolved RING lines after its prefetch.
#define RING 32
typedef struct { uint64_t uh, uref; uint32_t ts, x; KRec *k; } EPend;
typedef struct { uint64_t uh, cref; uint32_t ts, camp; } CPend;
typedef struct { EPend e[RING]; CPend c[RING]; uint32_t ne, nc; char pad[56]; } Pend;
static Pend *pend; // [NT]
INL void ev_emit(int tid, const EPend *p) {
  uint64_t pk = person_of(p->uh);
  Bkt *b = &pb[(size_t)tid * NQART + qpart(pk)];
  if (UNLIKELY(b->cur == b->end)) bkt_grow(&arenas[tid], b, sizeof(PRec), b->tail ? b->tail->cap : pcap0);
  PRec *r = (PRec *)b->cur; b->cur += sizeof(PRec);
  r->pkey = pk; r->uref = p->uref; r->ts = p->ts; r->x = p->x;
  p->k->loc = (uint32_t)(((char *)r - arenas[tid].base) >> 3);
}
INL void cl_emit(int tid, const CPend *p) {
  uint64_t pk = person_of(p->uh);
  Bkt *b = &cb[(size_t)tid * NQART + qpart(pk)];
  if (UNLIKELY(b->cur == b->end)) bkt_grow(&arenas[tid], b, sizeof(CRec), b->tail ? b->tail->cap : ccap0);
  CRec *r = (CRec *)b->cur; b->cur += sizeof(CRec);
  r->pkey = pk; r->cref = p->cref; r->ts = p->ts; r->camp = p->camp;
}
static void pend_flush(int tid) {
  Pend *P = &pend[tid];
  uint32_t lo = P->ne > RING ? P->ne - RING : 0;
  for (uint32_t i = lo; i < P->ne; i++) ev_emit(tid, &P->e[i % RING]);
  lo = P->nc > RING ? P->nc - RING : 0;
  for (uint32_t i = lo; i < P->nc; i++) cl_emit(tid, &P->c[i % RING]);
  P->ne = P->nc = 0;
}

static inline void ev_line(int tid, const char *ls, const char *const *sep, int ns, uint32_t nq) {
  if (ns != 7) return;
  const char *f[7]; uint32_t fl[7]; uint32_t nqf = 0;
  for (int k = 0; k < 7; k++) if (!field(ls, sep, k, &f[k], &fl[k], &nqf, nq)) return;
  if (nqf != nq) return;
  if (fl[0] == 0) return;
  uint64_t eh = hstr(f[0], fl[0]);
  uint64_t uh = 0;
  if (fl[1]) {
    uh = hstr(f[1], fl[1]);
    if (have_links) __builtin_prefetch(&ltab[uh & lmask_b]);
  }
  int64_t ing = LIKELY(fl[3] == 20) ? ingest_s(f[3]) * 1000 : parse_ts(f[3], fl[3]);
  int64_t ts = parse_ts(f[2], fl[2]);
  uint32_t x = 0;
  int ok = fl[1] > 0 && ts != BADTS && ts >= WS && ts < WE && fl[5] > 0;
  if (ok) {
    if (fl[4] == 4 && memcmp(f[4], "view", 4) == 0) {}
    else if (fl[4] == 4 && memcmp(f[4], "cart", 4) == 0) {}
    else if (fl[4] == 8 && memcmp(f[4], "purchase", 8) == 0) {
      uint32_t pr = parse_price(f[6], fl[6]);
      if (!pr) ok = 0;
      else {
        int32_t c = smap_get(&prodmap, f[5], fl[5]);
        x = pr | ((uint32_t)(c < 0 ? unknown_cat : c) << 24) | X_PURCH;
      }
    } else ok = 0;
  }
  if (ok && ing != BADTS && ing - ts > 86400000LL) ok = 0;
  int64_t is = ing == BADTS ? 0xFFFFFFFFll : ing / 1000;
  if (is < 0) is = 0;
  if (is > 0xFFFFFFFFll) is = 0xFFFFFFFFll;
  Bkt *b = &kb[(size_t)tid * NPART + (eh >> (64 - PB))];
  if (UNLIKELY(b->cur == b->end)) bkt_grow(&arenas[tid], b, sizeof(KRec), b->tail ? b->tail->cap : kcap0);
  KRec *k = (KRec *)b->cur; b->cur += sizeof(KRec);
  k->h = eh; k->ingest = (uint32_t)is; k->loc = LOC_NONE;
  if (ok) {
    Pend *P = &pend[tid];
    EPend *e = &P->e[P->ne % RING];
    if (P->ne >= RING) ev_emit(tid, e);
    e->uh = uh; e->uref = mkref(f[1], fl[1]); e->ts = (uint32_t)(ts - CWS); e->x = x; e->k = k;
    P->ne++;
  }
}

static inline void cl_line(int tid, const char *ls, const char *const *sep, int ns, uint32_t nq) {
  if (ns != 4) return;
  const char *f[4]; uint32_t fl[4]; uint32_t nqf = 0;
  for (int k = 0; k < 4; k++) if (!field(ls, sep, k, &f[k], &fl[k], &nqf, nq)) return;
  if (nqf != nq) return;
  if (fl[0] == 0 || fl[1] == 0) return;
  uint64_t uh = hstr(f[1], fl[1]);
  if (have_links) __builtin_prefetch(&ltab[uh & lmask_b]);
  int64_t ts = parse_ts(f[3], fl[3]);
  if (ts == BADTS || ts < CWS || ts >= WE) return;
  int32_t c = smap_get(&campmap, f[2], fl[2]);
  if (c < 0) return;
  Pend *P = &pend[tid];
  CPend *e = &P->c[P->nc % RING];
  if (P->nc >= RING) cl_emit(tid, e);
  e->uh = uh; e->cref = mkref(f[0], fl[0]); e->ts = (uint32_t)(ts - CWS); e->camp = (uint32_t)c;
  P->nc++;
}

#define EVL(ls, sep, ns, nq) ev_line(tid, ls, sep, ns, nq)
#define CLL(ls, sep, ns, nq) cl_line(tid, ls, sep, ns, nq)
static void parse_task(int tid, long t, void *ctx) {
  (void)ctx;
  const Chunk *c = &chunks[t];
  if (c->kind == 0) { TOKENIZE(c, EVL); }
  else { TOKENIZE(c, CLL); }
  pend_flush(tid);
}

// ---------- links ----------
typedef struct { uint32_t a, b; } Edge;
typedef struct { Edge *e; size_t n, cap; } EVec;
static EVec *ledges; // [NT]
static Chunk *lchunks; static long nlchunks;

INL uint32_t lt_insert(uint64_t h, const char *p, uint32_t n) {
  uint64_t i = h & lmask_b;
  for (;;) {
    uint64_t k = atomic_load_explicit(&ltab[i].key, memory_order_acquire);
    if (k == h) return (uint32_t)i;
    if (k == 0) {
      uint64_t z = 0;
      if (atomic_compare_exchange_strong(&ltab[i].key, &z, h)) { ltab[i].val = mkref(p, n); return (uint32_t)i; }
      if (z == h) return (uint32_t)i;
    }
    i = (i + 1) & lmask_b;
  }
}
// Link lines go through a delay line too: prefetch both slots, insert LRING lines later.
#define LRING 16
typedef struct { uint64_t ha, hb; const char *pa, *pb; uint32_t na, nb; } LPend;
typedef struct { LPend e[LRING]; uint32_t n; char pad[60]; } LRing;
static LRing *lring; // [NT]
INL void lk_emit(int tid, const LPend *p) {
  uint32_t a = lt_insert(p->ha, p->pa, p->na);
  uint32_t b = lt_insert(p->hb, p->pb, p->nb);
  if (a == b) return;
  EVec *v = &ledges[tid];
  if (v->n == v->cap) { v->cap = v->cap ? v->cap * 2 : 4096; v->e = xrealloc(v->e, v->cap * sizeof(Edge)); }
  v->e[v->n].a = a; v->e[v->n].b = b; v->n++;
}
static inline void lk_line(int tid, const char *ls, const char *const *sep, int ns, uint32_t nq) {
  if (ns != 2) return;
  const char *f[2]; uint32_t fl[2]; uint32_t nqf = 0;
  for (int k = 0; k < 2; k++) if (!field(ls, sep, k, &f[k], &fl[k], &nqf, nq)) return;
  if (nqf != nq) return;
  if (fl[0] == 0 || fl[1] == 0) return;
  uint64_t ha = hstr(f[0], fl[0]), hb = hstr(f[1], fl[1]);
  __builtin_prefetch(&ltab[ha & lmask_b], 1); __builtin_prefetch(&ltab[hb & lmask_b], 1);
  LRing *R = &lring[tid];
  LPend *e = &R->e[R->n % LRING];
  if (R->n >= LRING) lk_emit(tid, e);
  e->ha = ha; e->hb = hb; e->pa = f[0]; e->pb = f[1]; e->na = fl[0]; e->nb = fl[1];
  R->n++;
}
static void lring_flush(int tid) {
  LRing *R = &lring[tid];
  for (uint32_t i = R->n > LRING ? R->n - LRING : 0; i < R->n; i++) lk_emit(tid, &R->e[i % LRING]);
  R->n = 0;
}
#define LKL(ls, sep, ns, nq) lk_line(tid, ls, sep, ns, nq)
static void link_parse_task(int tid, long t, void *ctx) { (void)ctx; const Chunk *c = &lchunks[t]; TOKENIZE(c, LKL); lring_flush(tid); }

INL uint32_t uf_find(uint32_t x) {
  for (;;) {
    uint32_t p = atomic_load_explicit(&lpar[x], memory_order_relaxed);
    if (p == x) return x;
    uint32_t g = atomic_load_explicit(&lpar[p], memory_order_relaxed);
    if (g != p) atomic_compare_exchange_weak_explicit(&lpar[x], &p, g, memory_order_relaxed, memory_order_relaxed);
    x = g;
  }
}
static void uf_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  EVec *v = &ledges[t];
  for (size_t i = 0; i < v->n; i++) {
    uint32_t a = v->e[i].a, b = v->e[i].b;
    for (;;) {
      a = uf_find(a); b = uf_find(b);
      if (a == b) break;
      if (a < b) { uint32_t s = a; a = b; b = s; } // link larger root under smaller
      uint32_t exp = a;
      if (atomic_compare_exchange_strong(&lpar[a], &exp, b)) break;
    }
  }
}
#define LSPAN 65536
static void init_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  uint64_t lo = (uint64_t)t * LSPAN, hi = lo + LSPAN; if (hi > lcap) hi = lcap;
  for (uint64_t i = lo; i < hi; i++) { atomic_store_explicit(&lpar[i], (uint32_t)i, memory_order_relaxed); atomic_store_explicit(&lbest[i], UINT32_MAX, memory_order_relaxed); }
}
static void best_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  uint64_t lo = (uint64_t)t * LSPAN, hi = lo + LSPAN; if (hi > lcap) hi = lcap;
  for (uint64_t i = lo; i < hi; i++) {
    if (!atomic_load_explicit(&ltab[i].key, memory_order_relaxed)) continue;
    uint32_t r = uf_find((uint32_t)i);
    uint64_t me = ltab[i].val;
    uint32_t cur = atomic_load(&lbest[r]);
    for (;;) {
      if (cur != UINT32_MAX) {
        uint64_t o = ltab[cur].val;
        if (str_cmp(refp(me), refn(me), refp(o), refn(o)) >= 0) break;
      }
      if (atomic_compare_exchange_weak(&lbest[r], &cur, (uint32_t)i)) break;
    }
  }
}
INL uint64_t mix64(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return x; }
static void name_task(int tid, long t, void *ctx) { // lname[root] = smallest ID's ref
  (void)tid; (void)ctx;
  uint64_t lo = (uint64_t)t * LSPAN, hi = lo + LSPAN; if (hi > lcap) hi = lcap;
  for (uint64_t i = lo; i < hi; i++)
    if (atomic_load_explicit(&ltab[i].key, memory_order_relaxed) && atomic_load_explicit(&lpar[i], memory_order_relaxed) == (uint32_t)i)
      lname[i] = ltab[atomic_load(&lbest[i])].val;
}
static void pkey_task(int tid, long t, void *ctx) { // val: string ref -> person key
  (void)tid; (void)ctx;
  uint64_t lo = (uint64_t)t * LSPAN, hi = lo + LSPAN; if (hi > lcap) hi = lcap;
  for (uint64_t i = lo; i < hi; i++) {
    if (!atomic_load_explicit(&ltab[i].key, memory_order_relaxed)) continue;
    uint32_t r = uf_find((uint32_t)i);
    ltab[i].val = LINKED | (mix64(r) & 0x7FFFFFFF00000000ull) | r;
  }
}
static _Atomic uint64_t nl_count;
static void nl_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  const Chunk *c = &lchunks[t]; const char *p = c->base + c->start, *e = c->base + c->end; uint64_t n = 0;
  while ((p = memchr(p, '\n', (size_t)(e - p)))) { n++; p++; }
  atomic_fetch_add(&nl_count, n);
}

static void load_links(const char *dd) {
  char dir[4096]; snprintf(dir, sizeof dir, "%s/links", dd);
  char **paths; int nf = list_csv(dir, &paths);
  size_t bytes = 0;
  MFile *fs = xmalloc(sizeof(MFile) * (nf ? nf : 1));
  for (int i = 0; i < nf; i++) { fs[i] = map_file(paths[i]); bytes += fs[i].n; }
  if (bytes == 0) return;
  // chunks
  Chunk *save = chunks; long sn = nchunks, sc = chcap;
  chunks = NULL; nchunks = 0; chcap = 0;
  for (int i = 0; i < nf; i++) add_chunks(fs[i], 2, 1 << 18);
  lchunks = chunks; nlchunks = nchunks;
  chunks = save; nchunks = sn; chcap = sc;
  run_par(nlchunks, nl_task, NULL);
  PH(" l.count");
  uint64_t lines = nf + nl_count;
  uint64_t est = 2 * lines + lines / 2 + 1024; // distinct IDs <= 2 * lines
  lcap = 1024; while (lcap < est) lcap <<= 1;
  lmask_b = lcap - 1;
  ltab = hcalloc(lcap * sizeof(LEnt));
  lpar = hcalloc(lcap * 4); lbest = hcalloc(lcap * 4);
  ledges = xcalloc(NT, sizeof(EVec));
  lring = xcalloc(NT, sizeof(LRing));
  long nspan = (long)((lcap + LSPAN - 1) / LSPAN);
  run_par(nspan, init_task, NULL);
  PH(" l.init");
  run_par(nlchunks, link_parse_task, NULL);
  PH(" l.parse");
  run_par(NT, uf_task, NULL);
  PH(" l.uf");
  run_par(nspan, best_task, NULL);
  PH(" l.best");
  lname = hcalloc(lcap * 8);
  run_par(nspan, name_task, NULL);
  PH(" l.name");
  run_par(nspan, pkey_task, NULL);
  have_links = 1;
}

// ---------- dedup ----------
typedef struct { uint64_t h; uint32_t ingest; uint32_t loc; } DSlot; // loc: thread << 28 | loc
INL void kill_loc(uint32_t tl) {
  uint32_t l = tl & 0x0FFFFFFFu;
  if (l == LOC_NONE) return;
  PRec *r = (PRec *)(arenas[tl >> 28].base + ((size_t)l << 3));
  r->x |= X_DEAD;
}
static void dedup_task(int tid, long q, void *ctx) {
  (void)ctx;
  size_t cnt = 0;
  for (int t = 0; t < NT; t++) for (Blk *k = kb[(size_t)t * NPART + q].head; k; k = k->next) cnt += k->n;
  if (!cnt) return;
  size_t cap = 64; while (cap < cnt * 2) cap <<= 1;
  Arena *s = &scratch[tid]; s->cur = s->base;
  DSlot *tab = aalloc(s, cap * sizeof(DSlot));
  memset(tab, 0, cap * sizeof(DSlot));
  uint64_t mask = cap - 1;
  for (int t = 0; t < NT; t++) {
    for (Blk *k = kb[(size_t)t * NPART + q].head; k; k = k->next) {
      KRec *r = (KRec *)(k + 1);
      for (uint32_t j = 0; j < k->n; j++) {
        uint64_t h = r[j].h;
        uint32_t tl = ((uint32_t)t << 28) | r[j].loc;
        uint64_t i = h & mask;
        for (;;) {
          DSlot *d = &tab[i];
          if (!d->h) { d->h = h; d->ingest = r[j].ingest; d->loc = tl; break; }
          if (d->h == h) {
            if (r[j].ingest < d->ingest) { kill_loc(d->loc); d->ingest = r[j].ingest; d->loc = tl; }
            else kill_loc(tl);
            break;
          }
          i = (i + 1) & mask;
        }
      }
    }
  }
}

// ---------- sessionize ----------
typedef struct { uint64_t name; int32_t day; uint32_t sessions, events; int32_t top; int64_t revenue; } URow;
typedef struct { URow *a; size_t n, cap; } UVec;
static UVec *urows; // [NT]
typedef struct { int64_t sessions, purch, revenue, assisted; } CAgg;
static CAgg *cagg; // [NT][(ncamp+1)*NDAYS]
static uint32_t **stamp; static uint32_t *stampv;

static void sort_u64(uint64_t *a, uint64_t *tmp, uint32_t n) {
  if (n < 2) return;
  if (n <= 48) {
    for (uint32_t i = 1; i < n; i++) { uint64_t x = a[i]; uint32_t j = i; while (j && a[j - 1] > x) { a[j] = a[j - 1]; j--; } a[j] = x; }
    return;
  }
  if (n <= 8192) { // 4 passes of 8 bits over ts
    uint32_t c[4][256];
    memset(c, 0, sizeof c);
    for (uint32_t i = 0; i < n; i++) { uint32_t k = (uint32_t)(a[i] >> 32); c[0][k & 255]++; c[1][(k >> 8) & 255]++; c[2][(k >> 16) & 255]++; c[3][k >> 24]++; }
    uint64_t *src = a, *dst = tmp;
    for (int p = 0; p < 4; p++) {
      uint32_t s = 0;
      for (int b = 0; b < 256; b++) { uint32_t v = c[p][b]; c[p][b] = s; s += v; }
      int sh = 32 + 8 * p;
      for (uint32_t i = 0; i < n; i++) { uint32_t k = (uint32_t)(src[i] >> sh) & 255; dst[c[p][k]++] = src[i]; }
      uint64_t *t = src; src = dst; dst = t;
    }
    if (src != a) memcpy(a, src, (size_t)n * 8);
    return;
  }
  // LSD radix on bits 32..63 (ts), 3 passes of 11 bits
  uint32_t c[3][2048];
  memset(c, 0, sizeof c);
  for (uint32_t i = 0; i < n; i++) { uint32_t k = (uint32_t)(a[i] >> 32); c[0][k & 2047]++; c[1][(k >> 11) & 2047]++; c[2][k >> 22]++; }
  uint64_t *src = a, *dst = tmp;
  for (int p = 0; p < 3; p++) {
    uint32_t s = 0;
    for (int b = 0; b < 2048; b++) { uint32_t v = c[p][b]; c[p][b] = s; s += v; }
    int sh = 32 + 11 * p;
    for (uint32_t i = 0; i < n; i++) { uint32_t k = (uint32_t)(src[i] >> sh) & 2047; dst[c[p][k]++] = src[i]; }
    uint64_t *t = src; src = dst; dst = t;
  }
  if (src != a) memcpy(a, src, (size_t)n * 8);
}
static void flush_row(UVec *ur, URow *row, int64_t *catsum, uint32_t touched) {
  int best = -1; int64_t bv = 0;
  for (; touched; touched &= touched - 1) { // ascending category index, so a tie keeps the smallest
    int z = __builtin_ctz(touched);
    if (catsum[z] > bv) { bv = catsum[z]; best = z; }
    catsum[z] = 0;
  }
  row->top = best;
  if (ur->n == ur->cap) { ur->cap = ur->cap ? ur->cap * 2 : 65536; ur->a = xrealloc(ur->a, ur->cap * sizeof(URow)); }
  ur->a[ur->n++] = *row;
}

// person table slot: person index + 1, 0 = empty
INL uint32_t *pslot(uint32_t *tab, uint32_t pmask, const uint64_t *pkeys, uint64_t pk) {
  uint32_t i = (uint32_t)pk & pmask;
  while (tab[i] && pkeys[tab[i] - 1] != pk) i = (i + 1) & pmask;
  return &tab[i];
}

static void sess_task(int tid, long q, void *ctx) {
  (void)ctx;
  Arena *S = &scratch[tid]; S->cur = S->base;
  size_t ne = 0, nc = 0;
  for (int t = 0; t < NT; t++) {
    for (Blk *k = pb[(size_t)t * NQART + q].head; k; k = k->next) ne += k->n;
    for (Blk *k = cb[(size_t)t * NQART + q].head; k; k = k->next) nc += k->n;
  }
  if (!ne) return;
  // one pass over the records: per person counts, (ts << 32 | slot) and (person << 32 | x) per slot
  uint32_t pcap = 256; while (pcap < ne) pcap <<= 1;
  uint32_t *tab = aalloc(S, (size_t)pcap * 4); memset(tab, 0, (size_t)pcap * 4);
  uint32_t pmask = pcap - 1, np = 0;
  uint64_t *pkeys = aalloc(S, ne * 8);
  uint64_t *names = aalloc(S, ne * 8);
  uint32_t *eoff = aalloc(S, (ne + 1) * 4);
  eoff[0] = 0;
  uint64_t *px = aalloc(S, ne * 8);
  uint64_t *tx = aalloc(S, ne * 8); // ts << 32 | record slot
  size_t k = 0;
  for (int t = 0; t < NT; t++) {
    for (Blk *bk = pb[(size_t)t * NQART + q].head; bk; bk = bk->next) {
      PRec *r = (PRec *)(bk + 1);
      for (uint32_t j = 0; j < bk->n; j++) {
        uint32_t x = r[j].x;
        if (x & X_DEAD) continue;
        uint64_t pk = r[j].pkey;
        uint32_t *s = pslot(tab, pmask, pkeys, pk);
        if (!*s) {
          if (UNLIKELY((np + 1) * 2 > pcap)) { // grow
            pcap *= 2; pmask = pcap - 1;
            tab = aalloc(S, (size_t)pcap * 4); memset(tab, 0, (size_t)pcap * 4);
            for (uint32_t z = 0; z < np; z++) *pslot(tab, pmask, pkeys, pkeys[z]) = z + 1;
            s = pslot(tab, pmask, pkeys, pk);
          }
          pkeys[np] = pk;
          names[np] = pk & LINKED ? lname[(uint32_t)pk] : r[j].uref;
          *s = ++np;
          eoff[np] = 0;
        }
        eoff[*s]++;
        px[k] = ((uint64_t)(*s - 1) << 32) | x;
        tx[k] = ((uint64_t)r[j].ts << 32) | k;
        k++;
      }
    }
  }
  size_t nalive = k;
  for (uint32_t i = 0; i < np; i++) eoff[i + 1] += eoff[i];
  uint32_t *pos = aalloc(S, ((size_t)np + 1) * 4); memcpy(pos, eoff, ((size_t)np + 1) * 4);
  uint64_t *ev = aalloc(S, nalive * 8 + 8);
  uint64_t *tmp = aalloc(S, (nalive > nc ? nalive : nc) * 8 + 8);
  // sort the partition by ts once; the stable scatter by person leaves each run sorted
  sort_u64(tx, tmp, (uint32_t)nalive);
  for (size_t i = 0; i < nalive; i++) {
    uint64_t v = px[(uint32_t)tx[i]];
    ev[pos[v >> 32]++] = (tx[i] & 0xFFFFFFFF00000000ull) | (uint32_t)v;
  }
  // clicks: per person runs of (ts << 32 | click slot); ts ties are settled by click_id at credit time
  uint32_t *cpid = aalloc(S, (nc + 1) * 4);
  uint64_t *cref = aalloc(S, (nc + 1) * 8);
  uint32_t *ccamp = aalloc(S, (nc + 1) * 4);
  uint64_t *ctx_ = aalloc(S, (nc + 1) * 8);
  uint32_t *coff = aalloc(S, ((size_t)np + 1) * 4); memset(coff, 0, ((size_t)np + 1) * 4);
  k = 0;
  for (int t = 0; t < NT; t++) {
    for (Blk *bk = cb[(size_t)t * NQART + q].head; bk; bk = bk->next) {
      CRec *r = (CRec *)(bk + 1);
      for (uint32_t j = 0; j < bk->n; j++) {
        uint32_t id = *pslot(tab, pmask, pkeys, r[j].pkey);
        if (!id) continue;
        cpid[k] = id - 1;
        cref[k] = r[j].cref;
        ccamp[k] = r[j].camp;
        ctx_[k] = ((uint64_t)r[j].ts << 32) | k;
        coff[id]++;
        k++;
      }
    }
  }
  size_t nlive = k;
  for (uint32_t i = 0; i < np; i++) coff[i + 1] += coff[i];
  memcpy(pos, coff, ((size_t)np + 1) * 4);
  uint64_t *cl = aalloc(S, (nlive + 1) * 8);
  sort_u64(ctx_, tmp, (uint32_t)nlive);
  for (size_t i = 0; i < nlive; i++) cl[pos[cpid[(uint32_t)ctx_[i]]]++] = ctx_[i];
  UVec *ur = &urows[tid];
  CAgg *ca = &cagg[(size_t)tid * (ncamp + 1) * NDAYS];
  uint32_t *st = stamp[tid];
  int64_t catsum[32] = {0};
  uint32_t *sst = aalloc(S, nalive * 4); // per person: session start ts, day index, end of its day run
  uint32_t *sdi = aalloc(S, nalive * 4);
  uint32_t *drun = aalloc(S, nalive * 4);
  uint32_t *jprev = aalloc(S, ((size_t)ncamp + 1) * 4); // valid where st[] carries this person's stamp
  for (uint32_t p = 0; p < np; p++) {
    uint64_t *e = ev + eoff[p];
    uint32_t n = eoff[p + 1] - eoff[p];
    uint64_t *c = cl + coff[p];
    uint32_t m = coff[p + 1] - coff[p];
    uint32_t hi = 0, ns = 0;
    int64_t curday = INT64_MIN;
    URow row; memset(&row, 0, sizeof row);
    uint32_t touched = 0;
    uint32_t i = 0;
    while (i < n) {
      uint32_t s0 = (uint32_t)(e[i] >> 32);
      uint32_t last = s0;
      int64_t rev = 0; uint32_t cnt = 0; int purch = 0;
      int64_t day = (CWS + (int64_t)s0) / 86400000LL;
      if (day != curday) {
        if (curday != INT64_MIN) flush_row(ur, &row, catsum, touched);
        touched = 0;
        curday = day;
        row.name = names[p]; row.day = (int32_t)day; row.sessions = 0; row.events = 0; row.revenue = 0;
      }
      while (i < n) {
        uint32_t t = (uint32_t)(e[i] >> 32);
        if (t - last > 1800000u) break;
        last = t;
        uint32_t x = (uint32_t)e[i];
        cnt++;
        if (x & X_PURCH) {
          uint32_t pr = x & 0xFFFFFF;
          uint32_t cat = (x >> 24) & 31;
          rev += pr; purch = 1; catsum[cat] += pr; touched |= 1u << cat;
        }
        i++;
      }
      row.sessions++; row.events += cnt; row.revenue += rev;
      while (hi < m && (uint32_t)(c[hi] >> 32) <= s0) hi++;
      int di = (int)(day - DAY0);
      sst[ns] = s0;
      sdi[ns] = (uint32_t)di;
      ns++;
      uint32_t credited = (uint32_t)ncamp;
      if (hi && (int64_t)(c[hi - 1] >> 32) >= (int64_t)s0 - 604800000LL) {
        // latest ts wins; among equal ts the smallest click_id
        uint32_t ct = (uint32_t)(c[hi - 1] >> 32), best = (uint32_t)c[hi - 1];
        for (uint32_t z = hi - 1; z > 0 && (uint32_t)(c[z - 1] >> 32) == ct; z--) {
          uint32_t o = (uint32_t)c[z - 1];
          if (str_cmp(refp(cref[o]), refn(cref[o]), refp(cref[best]), refn(cref[best])) < 0) best = o;
        }
        credited = ccamp[best];
      }
      CAgg *a = &ca[(size_t)credited * NDAYS + di];
      a->sessions++; a->purch += purch; a->revenue += rev;
    }
    if (curday != INT64_MIN) flush_row(ur, &row, catsum, touched);
    if (!m) continue;
    // assists: a click at t assists the sessions starting in [t, t + 7 days]; per campaign only the
    // sessions past the previous click's range count, so each (session, campaign) pair counts once
    drun[ns - 1] = ns;
    for (uint32_t z = ns - 1; z-- > 0;) drun[z] = sdi[z] == sdi[z + 1] ? drun[z + 1] : z + 1;
    uint32_t sv = ++stampv[tid];
    uint32_t ip = 0, jp = 0;
    for (uint32_t z = 0; z < m; z++) {
      int64_t t = (int64_t)(c[z] >> 32);
      uint32_t cc = ccamp[(uint32_t)c[z]];
      while (ip < ns && sst[ip] < t) ip++;
      while (jp < ns && sst[jp] <= t + 604800000LL) jp++;
      uint32_t from = ip;
      if (st[cc] != sv) st[cc] = sv;
      else if (jprev[cc] > from) from = jprev[cc];
      jprev[cc] = jp;
      while (from < jp) {
        uint32_t to = drun[from] < jp ? drun[from] : jp;
        ca[(size_t)cc * NDAYS + sdi[from]].assisted += to - from;
        from = to;
      }
    }
  }
}

// ---------- parquet writer ----------
typedef struct { uint8_t *b; size_t n, cap; } Buf;
static inline void bres(Buf *b, size_t add) {
  if (b->n + add > b->cap) { size_t c = b->cap ? b->cap : 4096; while (c < b->n + add) c *= 2; b->b = xrealloc(b->b, c); b->cap = c; }
}
static inline void bput(Buf *b, const void *p, size_t n) { bres(b, n); memcpy(b->b + b->n, p, n); b->n += n; }
static inline void bbyte(Buf *b, uint8_t x) { bres(b, 1); b->b[b->n++] = x; }
static inline void bvar(Buf *b, uint64_t v) { bres(b, 10); while (v >= 0x80) { b->b[b->n++] = (uint8_t)(v | 0x80); v >>= 7; } b->b[b->n++] = (uint8_t)v; }
static inline void bu32(Buf *b, uint32_t v) { bput(b, &v, 4); }
static inline void bi64(Buf *b, int64_t v) { bput(b, &v, 8); }
static inline uint64_t zz(int64_t v) { return ((uint64_t)v << 1) ^ (uint64_t)(v >> 63); }

typedef struct { Buf *b; int16_t last[32]; int d; } TW;
static void tw_fh(TW *w, int id, int type) {
  int delta = id - w->last[w->d];
  if (delta > 0 && delta <= 15) bbyte(w->b, (uint8_t)((delta << 4) | type));
  else { bbyte(w->b, (uint8_t)type); bvar(w->b, zz(id)); }
  w->last[w->d] = (int16_t)id;
}
static void tw_i32(TW *w, int id, int32_t v) { tw_fh(w, id, 5); bvar(w->b, zz(v)); }
static void tw_i64(TW *w, int id, int64_t v) { tw_fh(w, id, 6); bvar(w->b, zz(v)); }
static void tw_bin(TW *w, int id, const char *s, size_t n) { tw_fh(w, id, 8); bvar(w->b, n); bput(w->b, s, n); }
static void tw_sbeg(TW *w, int id) { tw_fh(w, id, 12); w->d++; w->last[w->d] = 0; }
static void tw_send(TW *w) { bbyte(w->b, 0); w->d--; }
static void tw_lbeg(TW *w, int id, int et, int n) { tw_fh(w, id, 9); if (n < 15) bbyte(w->b, (uint8_t)((n << 4) | et)); else { bbyte(w->b, (uint8_t)(0xF0 | et)); bvar(w->b, (uint64_t)n); } }
static void tw_ebeg(TW *w) { w->d++; w->last[w->d] = 0; }
static void tw_eend(TW *w) { bbyte(w->b, 0); w->d--; }

enum { PT_INT32 = 1, PT_INT64 = 2, PT_BYTES = 6 };
enum { LT_NONE = 0, LT_STRING = 1, LT_DATE = 6 };
typedef struct { const char *name; int type; int logical; int optional; } ColDef;
typedef struct { Buf data; int64_t nvals; int64_t off; int64_t size; } ColOut;

static void write_page(Buf *file, ColOut *co, Buf *levels, int64_t nvals) {
  Buf hdr = {0};
  TW w = {&hdr, {0}, 0};
  size_t body = co->data.n + (levels ? levels->n : 0);
  tw_i32(&w, 1, 0);
  tw_i32(&w, 2, (int32_t)body);
  tw_i32(&w, 3, (int32_t)body);
  tw_sbeg(&w, 5);
  tw_i32(&w, 1, (int32_t)nvals);
  tw_i32(&w, 2, 0);
  tw_i32(&w, 3, 3);
  tw_i32(&w, 4, 3);
  tw_send(&w);
  bbyte(&hdr, 0);
  co->off = (int64_t)file->n;
  bput(file, hdr.b, hdr.n);
  if (levels) bput(file, levels->b, levels->n);
  bput(file, co->data.b, co->data.n);
  co->size = (int64_t)(file->n) - co->off;
  co->nvals = nvals;
  free(hdr.b);
}

static void write_footer(Buf *file, const ColDef *cols, int nc, ColOut *co, int64_t nrows) {
  Buf meta = {0};
  TW w = {&meta, {0}, 0};
  tw_i32(&w, 1, 1);
  tw_lbeg(&w, 2, 12, nc + 1);
  tw_ebeg(&w); tw_bin(&w, 4, "schema", 6); tw_i32(&w, 5, nc); tw_eend(&w);
  for (int i = 0; i < nc; i++) {
    tw_ebeg(&w);
    tw_i32(&w, 1, cols[i].type);
    tw_i32(&w, 3, cols[i].optional ? 1 : 0);
    tw_bin(&w, 4, cols[i].name, strlen(cols[i].name));
    if (cols[i].logical == LT_STRING) tw_i32(&w, 6, 0);
    if (cols[i].logical == LT_DATE) tw_i32(&w, 6, 6);
    if (cols[i].logical) { tw_sbeg(&w, 10); tw_sbeg(&w, cols[i].logical); tw_send(&w); tw_send(&w); }
    tw_eend(&w);
  }
  tw_i64(&w, 3, nrows);
  if (nrows > 0) {
    tw_lbeg(&w, 4, 12, 1);
    tw_ebeg(&w);
    tw_lbeg(&w, 1, 12, nc);
    int64_t total = 0;
    for (int i = 0; i < nc; i++) {
      tw_ebeg(&w);
      tw_i64(&w, 2, co[i].off);
      tw_sbeg(&w, 3);
      tw_i32(&w, 1, cols[i].type);
      tw_lbeg(&w, 2, 5, 2); bvar(&meta, zz(0)); bvar(&meta, zz(3));
      tw_lbeg(&w, 3, 8, 1); bvar(&meta, strlen(cols[i].name)); bput(&meta, cols[i].name, strlen(cols[i].name));
      tw_i32(&w, 4, 0);
      tw_i64(&w, 5, co[i].nvals);
      tw_i64(&w, 6, co[i].size);
      tw_i64(&w, 7, co[i].size);
      tw_i64(&w, 9, co[i].off);
      tw_send(&w);
      tw_eend(&w);
      total += co[i].size;
    }
    tw_i64(&w, 2, total);
    tw_i64(&w, 3, nrows);
    tw_eend(&w);
  } else {
    tw_lbeg(&w, 4, 12, 0);
  }
  tw_bin(&w, 6, "sess", 4);
  bbyte(&meta, 0);
  bput(file, meta.b, meta.n);
  bu32(file, (uint32_t)meta.n);
  bput(file, "PAR1", 4);
  free(meta.b);
}

static void write_file(const char *path, Buf *file) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) die(path);
  size_t off = 0;
  while (off < file->n) {
    ssize_t w = write(fd, file->b + off, file->n - off);
    if (w < 0) { if (errno == EINTR) continue; die("write"); }
    off += (size_t)w;
  }
  close(fd);
}

static const ColDef UCOLS[6] = {
  {"user_id", PT_BYTES, LT_STRING, 0}, {"day", PT_INT32, LT_DATE, 0}, {"sessions", PT_INT32, 0, 0},
  {"events", PT_INT32, 0, 0}, {"revenue_cents", PT_INT64, 0, 0}, {"top_category", PT_BYTES, LT_STRING, 1}};
static const ColDef CCOLS[7] = {
  {"campaign_id", PT_BYTES, LT_STRING, 0}, {"channel", PT_BYTES, LT_STRING, 0}, {"day", PT_INT32, LT_DATE, 0},
  {"sessions", PT_INT64, 0, 0}, {"purchasing_sessions", PT_INT64, 0, 0}, {"revenue_cents", PT_INT64, 0, 0},
  {"assisted_sessions", PT_INT64, 0, 0}};

static char *OUTDIR;

static void write_user_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  UVec *v = &urows[t];
  if (!v->n && t != 0) return;
  size_t n = v->n;
  ColOut co[6]; memset(co, 0, sizeof co);
  size_t sbytes = 0;
  for (size_t i = 0; i < n; i++) sbytes += refn(v->a[i].name) + 4;
  bres(&co[0].data, sbytes); bres(&co[1].data, n * 4); bres(&co[2].data, n * 4); bres(&co[3].data, n * 4);
  bres(&co[4].data, n * 8); bres(&co[5].data, n * 16 + 16);
  for (size_t i = 0; i < n; i++) {
    URow *r = &v->a[i];
    uint32_t nn = refn(r->name);
    memcpy(co[0].data.b + co[0].data.n, &nn, 4); memcpy(co[0].data.b + co[0].data.n + 4, refp(r->name), nn); co[0].data.n += 4 + nn;
    int32_t d = r->day; memcpy(co[1].data.b + co[1].data.n, &d, 4); co[1].data.n += 4;
    memcpy(co[2].data.b + co[2].data.n, &r->sessions, 4); co[2].data.n += 4;
    memcpy(co[3].data.b + co[3].data.n, &r->events, 4); co[3].data.n += 4;
    memcpy(co[4].data.b + co[4].data.n, &r->revenue, 8); co[4].data.n += 8;
    if (r->top >= 0) { bu32(&co[5].data, catn[r->top]); bput(&co[5].data, catp[r->top], catn[r->top]); }
  }
  Buf lev = {0};
  size_t groups = (n + 7) / 8;
  Buf enc = {0};
  bvar(&enc, (groups << 1) | 1);
  bres(&enc, groups);
  memset(enc.b + enc.n, 0, groups);
  for (size_t i = 0; i < n; i++) if (v->a[i].top >= 0) enc.b[enc.n + i / 8] |= (uint8_t)(1u << (i % 8));
  enc.n += groups;
  bu32(&lev, (uint32_t)enc.n); bput(&lev, enc.b, enc.n);
  Buf file = {0};
  size_t tot = 64 + lev.n; for (int c = 0; c < 6; c++) tot += co[c].data.n + 64;
  bres(&file, tot + 4096);
  bput(&file, "PAR1", 4);
  if (n) for (int c = 0; c < 6; c++) write_page(&file, &co[c], c == 5 ? &lev : NULL, (int64_t)n);
  write_footer(&file, UCOLS, 6, co, (int64_t)n);
  char path[4096];
  snprintf(path, sizeof path, "%s/user_daily/part-%03ld.parquet", OUTDIR, t);
  write_file(path, &file);
}

static void write_campaign(void) {
  ColOut co[7]; memset(co, 0, sizeof co);
  int64_t n = 0;
  size_t stride = (size_t)(ncamp + 1) * NDAYS;
  for (int c = 0; c <= ncamp; c++) {
    for (int d = 0; d < NDAYS; d++) {
      CAgg a = {0, 0, 0, 0};
      for (int t = 0; t < NT; t++) {
        CAgg *x = &cagg[(size_t)t * stride + (size_t)c * NDAYS + d];
        a.sessions += x->sessions; a.purch += x->purch; a.revenue += x->revenue; a.assisted += x->assisted;
      }
      if (!a.sessions && !a.assisted) continue;
      const char *id = c == ncamp ? "organic" : camp_id[c];
      uint32_t idn = c == ncamp ? 7 : camp_idn[c];
      const char *ch = c == ncamp ? "organic" : chan[c];
      uint32_t chn = c == ncamp ? 7 : chann[c];
      bu32(&co[0].data, idn); bput(&co[0].data, id, idn);
      bu32(&co[1].data, chn); bput(&co[1].data, ch, chn);
      int32_t day = (int32_t)(DAY0 + d); bput(&co[2].data, &day, 4);
      bi64(&co[3].data, a.sessions);
      bi64(&co[4].data, a.purch);
      bi64(&co[5].data, a.revenue);
      bi64(&co[6].data, c == ncamp ? 0 : a.assisted);
      n++;
    }
  }
  Buf file = {0};
  bput(&file, "PAR1", 4);
  if (n) for (int c = 0; c < 7; c++) write_page(&file, &co[c], NULL, n);
  write_footer(&file, CCOLS, 7, co, n);
  char path[4096];
  snprintf(path, sizeof path, "%s/campaign_daily/part-000.parquet", OUTDIR);
  write_file(path, &file);
}

// ---------- setup ----------
static int64_t manifest_ts(const char *s, size_t n, const char *key) {
  const char *k = NULL;
  size_t kl = strlen(key);
  for (size_t i = 0; i + kl <= n; i++) if (memcmp(s + i, key, kl) == 0) { k = s + i + kl; break; }
  if (!k) die("manifest key");
  const char *e = s + n;
  while (k < e && *k != ':') k++;
  k++;
  while (k < e && *k != '"') k++;
  k++;
  const char *q = k;
  while (q < e && *q != '"') q++;
  int64_t v = parse_ts(k, (uint32_t)(q - k));
  if (v == BADTS) die("manifest ts");
  return v;
}

typedef struct { const char *p; uint32_t n; } Str;
static int cmp_str(const void *a, const void *b) { const Str *x = a, *y = b; return str_cmp(x->p, x->n, y->p, y->n); }

// simple scalar CSV split for the small clean files
static int split2(const char *p, size_t n, const char **fp, uint32_t *fl) {
  const char *end = p + n; const char *s = p; int k = 0;
  for (;;) {
    const char *c = memchr(s, ',', (size_t)(end - s));
    const char *fe = c ? c : end;
    if (k >= 2) return 0;
    size_t l = (size_t)(fe - s);
    if (l && s[0] == '"') { if (l < 2 || s[l - 1] != '"') return 0; if (l > 2 && memchr(s + 1, '"', l - 2)) return 0; fp[k] = s + 1; fl[k] = (uint32_t)(l - 2); }
    else { if (l && memchr(s, '"', l)) return 0; fp[k] = s; fl[k] = (uint32_t)l; }
    k++;
    if (!c) break;
    s = c + 1;
  }
  return k == 2;
}

static void load_products(const char *dd) {
  char path[4096]; snprintf(path, sizeof path, "%s/products.csv", dd);
  MFile f = map_file(path);
  Str *cats = xmalloc(sizeof(Str) * 64); int nc = 0, ccap = 64;
  SMap cm; smap_init(&cm, 1024);
  size_t lines = f.n / 8 + 16;
  smap_init(&prodmap, lines);
  const char *p = f.p, *end = f.p + f.n;
  const char *nl = memchr(p, '\n', f.n); p = nl ? nl + 1 : end;
  Str *prods = xmalloc(sizeof(Str) * lines), *pcat = xmalloc(sizeof(Str) * lines); size_t npd = 0;
  while (p < end) {
    nl = memchr(p, '\n', (size_t)(end - p));
    const char *le = nl ? nl : end;
    const char *fp[2]; uint32_t fl[2];
    if (le > p && split2(p, (size_t)(le - p), fp, fl)) {
      int cr;
      smap_put(&cm, fp[1], fl[1], 0, &cr);
      if (cr) { if (nc == ccap) { ccap *= 2; cats = xrealloc(cats, sizeof(Str) * ccap); } cats[nc].p = fp[1]; cats[nc].n = fl[1]; nc++; }
      prods[npd].p = fp[0]; prods[npd].n = fl[0]; pcat[npd].p = fp[1]; pcat[npd].n = fl[1]; npd++;
    }
    p = le + 1;
  }
  int cr;
  smap_put(&cm, "unknown", 7, 0, &cr);
  if (cr) { if (nc == ccap) { ccap *= 2; cats = xrealloc(cats, sizeof(Str) * ccap); } cats[nc].p = "unknown"; cats[nc].n = 7; nc++; }
  qsort(cats, nc, sizeof(Str), cmp_str);
  ncat = nc;
  if (ncat > 31) die("too many categories");
  catp = xmalloc(sizeof(char *) * ncat); catn = xmalloc(sizeof(uint32_t) * ncat);
  for (int i = 0; i < ncat; i++) {
    catp[i] = cats[i].p; catn[i] = cats[i].n;
    SSlot *s = smap_put(&cm, cats[i].p, cats[i].n, 0, &cr); s->v = i;
    if (str_eq(cats[i].p, cats[i].n, "unknown", 7)) unknown_cat = i;
  }
  for (size_t i = 0; i < npd; i++) {
    int32_t ci = smap_get(&cm, pcat[i].p, pcat[i].n);
    smap_put(&prodmap, prods[i].p, prods[i].n, ci, &cr);
  }
}

static void load_campaigns(const char *dd) {
  char path[4096]; snprintf(path, sizeof path, "%s/campaigns.csv", dd);
  MFile f = map_file(path);
  size_t lines = f.n / 4 + 16;
  smap_init(&campmap, lines);
  camp_id = xmalloc(sizeof(char *) * lines); camp_idn = xmalloc(4 * lines);
  chan = xmalloc(sizeof(char *) * lines); chann = xmalloc(4 * lines);
  const char *p = f.p, *end = f.p + f.n;
  const char *nl = memchr(p, '\n', f.n); p = nl ? nl + 1 : end;
  while (p < end) {
    nl = memchr(p, '\n', (size_t)(end - p));
    const char *le = nl ? nl : end;
    const char *fp[2]; uint32_t fl[2];
    if (le > p && split2(p, (size_t)(le - p), fp, fl) && fl[0]) {
      int cr;
      smap_put(&campmap, fp[0], fl[0], ncamp, &cr);
      if (cr) { camp_id[ncamp] = fp[0]; camp_idn[ncamp] = fl[0]; chan[ncamp] = fp[1]; chann[ncamp] = fl[1]; ncamp++; }
    }
    p = le + 1;
  }
}

static uint64_t *qsize;
static int cmp_q(const void *a, const void *b) {
  uint64_t x = qsize[*(const long *)a], y = qsize[*(const long *)b];
  return x > y ? -1 : (x < y ? 1 : 0);
}

static void close_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  for (uint32_t i = 0; i < NPART; i++) bkt_close(&kb[(size_t)t * NPART + i], sizeof(KRec));
  for (uint32_t i = 0; i < NQART; i++) { bkt_close(&pb[(size_t)t * NQART + i], sizeof(PRec)); bkt_close(&cb[(size_t)t * NQART + i], sizeof(CRec)); }
}

// Drop pages in parallel (madvise takes the mmap lock shared) so exit has little left to tear down serially.
static void release_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  if (t < nchunks) { const Chunk *c = &chunks[t]; madvise((void *)(c->base + c->start), c->end - c->start, MADV_DONTNEED); return; }
  t -= nchunks;
  madvise(arenas[t].base, (size_t)(arenas[t].cur - arenas[t].base), MADV_DONTNEED);
  madvise(scratch[t].base, 1ull << 30, MADV_DONTNEED);
}

int main(void) {
  PROF = getenv("SESS_PROF") != NULL; T0 = now_s();
  const char *dd = getenv("DATA_DIR"); if (!dd) dd = "/data";
  const char *od = getenv("OUTPUT_DIR"); if (!od) od = "/output";
  OUTDIR = (char *)od;
  const char *nc = getenv("NCPU");
  if (nc && atoi(nc) > 0) NT = atoi(nc);
  if (NT > 16) NT = 16;
  char path[4096];
  snprintf(path, sizeof path, "%s/manifest.json", dd);
  MFile mf = map_file(path);
  WS = manifest_ts(mf.p, mf.n, "\"window_start\"");
  WE = manifest_ts(mf.p, mf.n, "\"window_end\"");
  CWS = WS - 604800000LL;
  DAY0 = WS / 86400000LL; if (WS < 0 && WS % 86400000LL) DAY0--;
  NDAYS = (int32_t)((WE - 1) / 86400000LL - DAY0 + 1);
  load_products(dd);
  load_campaigns(dd);
  PH("small");
  load_links(dd);
  PH("links");

  char dir[4096];
  char **ep, **cp;
  snprintf(dir, sizeof dir, "%s/events", dd); int ne = list_csv(dir, &ep);
  snprintf(dir, sizeof dir, "%s/clicks", dd); int ncl = list_csv(dir, &cp);
  size_t ebytes = 0, cbytes = 0;
  MFile *ef = xmalloc(sizeof(MFile) * (ne + 1)), *cf = xmalloc(sizeof(MFile) * (ncl + 1));
  for (int i = 0; i < ne; i++) { ef[i] = map_file(ep[i]); ebytes += ef[i].n; }
  for (int i = 0; i < ncl; i++) { cf[i] = map_file(cp[i]); cbytes += cf[i].n; }
  size_t csz = 1u << 20;
  for (int i = 0; i < ne; i++) add_chunks(ef[i], 0, csz);
  for (int i = 0; i < ncl; i++) add_chunks(cf[i], 1, csz);
  // partition sizing
  uint64_t eest = ebytes / 60 + 1024, cest = cbytes / 45 + 1024;
  // ~30K events per partition: fewer write streams at parse (2^10 at 20M), tables still in L2
  PB = 6; while (PB < 14 && (eest >> PB) > 30000) PB++;
  QB = 6; while (QB < 13 && (eest >> QB) > 30000) QB++;
  NPART = 1u << PB; NQART = 1u << QB;
  uint64_t per_t = eest / NT;
  kcap0 = (uint32_t)(per_t / NPART); kcap0 = kcap0 + kcap0 / 4 + 32;
  pcap0 = (uint32_t)(per_t / NQART); pcap0 = pcap0 + pcap0 / 4 + 32;
  ccap0 = (uint32_t)(cest / NT / NQART); ccap0 = ccap0 / 2 + 32;
  arenas = xcalloc(NT, sizeof(Arena)); scratch = xcalloc(NT, sizeof(Arena));
  for (int t = 0; t < NT; t++) { arena_init(&arenas[t], ARENA_SZ); arena_init(&scratch[t], 1ull << 30); }
  pend = xcalloc(NT, sizeof(Pend));
  kb = xcalloc((size_t)NT * NPART, sizeof(Bkt));
  pb = xcalloc((size_t)NT * NQART, sizeof(Bkt));
  cb = xcalloc((size_t)NT * NQART, sizeof(Bkt));
  PH("map");
  run_par(nchunks, parse_task, NULL);
  run_par(NT, close_task, NULL);
  PH("parse");
  run_par(NPART, dedup_task, NULL);
  PH("dedup");
  urows = xcalloc(NT, sizeof(UVec));
  cagg = xcalloc((size_t)NT * (ncamp + 1) * NDAYS, sizeof(CAgg));
  stamp = xmalloc(sizeof(uint32_t *) * NT);
  stampv = xcalloc(NT, sizeof(uint32_t));
  for (int t = 0; t < NT; t++) stamp[t] = xcalloc(ncamp + 1, sizeof(uint32_t));
  // largest person partitions first
  qsize = xcalloc(NQART, 8);
  for (int t = 0; t < NT; t++) for (uint32_t q = 0; q < NQART; q++) for (Blk *k = pb[(size_t)t * NQART + q].head; k; k = k->next) qsize[q] += k->n;
  long *order = xmalloc(sizeof(long) * NQART);
  for (uint32_t q = 0; q < NQART; q++) order[q] = q;
  qsort(order, NQART, sizeof(long), cmp_q);
  run_par_o(NQART, order, sess_task, NULL);
  PH("sess");
  snprintf(dir, sizeof dir, "%s/user_daily", od); if (mkdir(dir, 0777) && errno != EEXIST) die(dir);
  snprintf(dir, sizeof dir, "%s/campaign_daily", od); if (mkdir(dir, 0777) && errno != EEXIST) die(dir);
  run_par(NT, write_user_task, NULL);
  write_campaign();
  PH("write");
  run_par(nchunks + NT, release_task, NULL);
  PH("release");
  _exit(0);
}
