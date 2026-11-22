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
#include <sys/uio.h>
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
// Sub-phase ticks per thread (SESS_PROF): a cheap counter, scaled to ms against now_s over the phase.
INL uint64_t ticks(void) {
#if defined(__x86_64__)
  return __rdtsc();
#elif defined(__aarch64__)
  uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
#else
  return (uint64_t)(now_s() * 1e9);
#endif
}
enum { SP_DSPLIT, SP_DPROBE, SP_GATHER, SP_SORT, SP_CLICKS, SP_WALK, SP_N };
static uint64_t sprof[16][8]; // [thread][SP_*], one cache line per thread
#define SP_ADD(tid, k, t0) do { if (PROF) { uint64_t t1_ = ticks(); sprof[tid][k] += t1_ - (t0); (t0) = t1_; } } while (0)
// Prints sub-phases k0..k1-1 as PH-style lines: max over threads, then the sum.
static void sp_print(int k0, int k1, const char *const *names, double ms_per_tick) {
  if (!PROF) return;
  for (int k = k0; k < k1; k++) {
    uint64_t mx = 0, sum = 0;
    for (int t = 0; t < 16; t++) { sum += sprof[t][k]; if (sprof[t][k] > mx) mx = sprof[t][k]; }
    fprintf(stderr, "%-10s %7.1f ms max/thread, %.1f ms sum\n", names[k - k0], mx * ms_per_tick, sum * ms_per_tick);
  }
}

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

static size_t file_size(const char *path) {
  struct stat st;
  if (stat(path, &st) < 0) die(path);
  return (size_t)st.st_size;
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
#if !defined(__AVX2__)
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
#endif

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

// Every form, exact. Reads only p[0..n-1].
static int64_t parse_ts_any(const char *p, uint32_t n) {
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

#if defined(__AVX2__)
// Per form k (0: 13-digit F3, 1: F1, 2: F2), with the 32 bytes at p as x = p - t8 and the bytes at p + 1 as y = p + 1 - '0':
// lane i is good when x[i] <= l8[i]; u16 pairs u = maddubs(x, wa) + maddubs(y, wb) are good when u - lo <= rg;
// madd(u, m) gives F3 as 4+4+4+1 digit groups, or F1/F2 as year, month, day * 24 + hour, minute * 60, second, offset minutes.
typedef struct { uint8_t t8[32], l8[32]; int8_t wa[32], wb[32]; uint16_t lo[16], rg[16]; int16_t m[16]; } TsForm;
static TsForm ts_form[3] __attribute__((aligned(32)));
// Form by length: 0 F3 13 digits, 1 F1, 2 F2, 3 any other length.
static const uint8_t ts_kind[26] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 2};
__attribute__((constructor)) static void ts_form_init(void) {
  static const char *pat[3] = {"DDDDDDDDDDDDD", "DDDD-DD-DDTDD:DD:DDZ", "DDDD-DD-DDTDD:DD:DD+DD:DD"};
  for (int k = 0; k < 3; k++) {
    TsForm *f = &ts_form[k];
    memset(f->l8, 0xFF, 32);
    for (int i = 0; pat[k][i]; i++) {
      char c = pat[k][i];
      f->t8[i] = c == 'D' ? '0' : (uint8_t)c;
      f->l8[i] = c == 'D' ? 9 : c == '+' ? 2 : 0; // '+' covers '+', ',' and '-'
    }
    for (int i = 0; i < 16; i++) f->rg[i] = 0xFFFF;
    if (k == 0) {
      for (int i = 0; i < 12; i += 2) { f->wa[i] = 10; f->wa[i + 1] = 1; }
      f->wa[12] = 1;
      for (int i = 0; i < 6; i += 2) { f->m[i] = 100; f->m[i + 1] = 1; }
      f->m[6] = 1;
      continue;
    }
    // u16 lanes from x: 0, 1 year halves, 4 day, 7 minute, 10 offset hour; from y: 2 month, 5 hour, 8 second, 11 offset minute.
    static const int fa[5] = {0, 1, 4, 7, 10}, fb[4] = {2, 5, 8, 11};
    for (int j = 0; j < 5; j++) { f->wa[2 * fa[j]] = 10; f->wa[2 * fa[j] + 1] = 1; }
    for (int j = 0; j < 4; j++) { f->wb[2 * fb[j]] = 10; f->wb[2 * fb[j] + 1] = 1; }
    f->lo[2] = 1; f->rg[2] = 11;
    f->lo[4] = 1; f->rg[4] = 30;
    f->rg[5] = 23; f->rg[7] = 59; f->rg[8] = 59;
    f->m[0] = 100; f->m[1] = 1; f->m[2] = 1; f->m[4] = 24; f->m[5] = 1; f->m[7] = 60; f->m[8] = 1;
    if (k == 2) { f->rg[10] = 23; f->rg[11] = 59; f->m[10] = 60; f->m[11] = 1; }
  }
}

// Branch-free for the three common lengths (the form mix makes a length branch unpredictable); the rest go to
// parse_ts_any. Reads p[0..32].
INL int64_t parse_ts(const char *p, uint32_t n) {
  // Selects below use masks: gcc turns ternaries on k into branches.
  unsigned k = n < 26 ? ts_kind[n] : 3;
  if (UNLIKELY(k == 3)) return parse_ts_any(p, n);
  const TsForm *f = &ts_form[k];
  __m256i x = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)p), _mm256_load_si256((const __m256i *)f->t8));
  __m256i y = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(p + 1)), _mm256_set1_epi8('0'));
  __m256i l8 = _mm256_load_si256((const __m256i *)f->l8);
  __m256i u = _mm256_add_epi16(_mm256_maddubs_epi16(x, _mm256_load_si256((const __m256i *)f->wa)),
                               _mm256_maddubs_epi16(y, _mm256_load_si256((const __m256i *)f->wb)));
  __m256i rg = _mm256_load_si256((const __m256i *)f->rg);
  __m256i ur = _mm256_sub_epi16(u, _mm256_load_si256((const __m256i *)f->lo));
  __m256i good = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(x, l8), l8), _mm256_cmpeq_epi16(_mm256_max_epu16(ur, rg), rg));
  unsigned bad = (unsigned)_mm256_movemask_epi8(good) != 0xFFFFFFFFu;
  bad |= (k == 2) & (p[19] == ',');
  __m256i g = _mm256_madd_epi16(u, _mm256_load_si256((const __m256i *)f->m));
  __m128i g0 = _mm256_castsi256_si128(g);
  uint64_t a = (uint64_t)_mm_cvtsi128_si64(g0), b = (uint64_t)_mm_extract_epi64(g0, 1);
  uint64_t c = (uint64_t)_mm_cvtsi128_si64(_mm256_extracti128_si256(g, 1));
  unsigned g1 = (uint32_t)a, g2 = (uint32_t)(a >> 32), g3 = (uint32_t)b, g4 = (uint32_t)(b >> 32);
  int64_t f3 = (int64_t)g1 * 1000000000 + (int64_t)g2 * 100000 + (int64_t)g3 * 10 + g4;
  // F3 takes year 1900, so only an F1/F2 year can go out of the table.
  unsigned cm = -(unsigned)(k != 0);
  unsigned yi = (g1 - TS_Y0) & cm;
  unsigned out = yi >= TS_NY;
  if (UNLIKELY(out & !bad)) return parse_ts_any(p, n);
  yi &= out - 1;
  unsigned mo = g2 & 15, da = (unsigned)_mm256_extract_epi16(u, 4);
  unsigned leap = (unsigned)(ts_ystart[yi + 1] - ts_ystart[yi]) - 365;
  bad |= cm & (da > ts_mdays[leap][mo]);
  int64_t hours = (int64_t)(ts_ystart[yi] + ts_moff[leap][mo] - 1) * 24 + g3;
  // Offset sign at p[19]: '+' (bit 2 clear) is subtracted, '-' added; F1 has offset 0.
  int64_t off = (int64_t)(uint32_t)(c >> 32) * 60 * (1 - ((p[19] >> 1) & 2));
  int64_t cal = (hours * 3600 + g4 + (uint32_t)c - off) * 1000;
  int64_t km = (int32_t)cm, bm = -(int64_t)bad;
  int64_t r = (cal & km) | (f3 & ~km);
  return (r & ~bm) | (BADTS & bm);
}

// parse_ts's F1 lanes without its checks. Reads p[0..32].
INL int64_t ingest_s(const char *p) {
  const TsForm *f = &ts_form[1];
  __m256i z = _mm256_set1_epi8('0');
  __m256i x = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)p), z);
  __m256i y = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(p + 1)), z);
  __m256i u = _mm256_add_epi16(_mm256_maddubs_epi16(x, _mm256_load_si256((const __m256i *)f->wa)),
                               _mm256_maddubs_epi16(y, _mm256_load_si256((const __m256i *)f->wb)));
  __m256i g = _mm256_madd_epi16(u, _mm256_load_si256((const __m256i *)f->m));
  __m128i g0 = _mm256_castsi256_si128(g);
  uint64_t a = (uint64_t)_mm_cvtsi128_si64(g0), b = (uint64_t)_mm_extract_epi64(g0, 1);
  unsigned ss = (unsigned)_mm_cvtsi128_si32(_mm256_extracti128_si256(g, 1));
  unsigned yr = (uint32_t)a, mo = (uint32_t)(a >> 32), yi = yr - TS_Y0;
  int64_t days;
  if (LIKELY(yi < TS_NY)) {
    unsigned leap = (unsigned)(ts_ystart[yi + 1] - ts_ystart[yi]) - 365;
    days = ts_ystart[yi] + ts_moff[leap][mo];
  } else days = days_from_civil(yr, mo, 1);
  return ((days - 1) * 24 + (uint32_t)b) * 3600 + (b >> 32) + ss;
}
#else
#define parse_ts parse_ts_any
#endif

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
// 8 B slots: the hash's top 32 bits << 32 | entry + 1 (0 = empty), confirmed on the entry's string. Sized by the
// entries, not the file: the 2,000 campaigns probe 32 KB instead of 512 KB, the 50,000 products 1 MB instead of 8 MB.
typedef struct { const char *p; uint32_t n; int32_t v; } SEnt;
typedef struct { uint64_t *s; SEnt *e; uint32_t n, ecap; uint64_t mask; } SMap;
#define SFP 0xFFFFFFFF00000000ull
static void smap_init(SMap *m) {
  m->mask = 63;
  m->s = xcalloc(64, 8);
  m->n = 0;
  m->ecap = 32;
  m->e = xmalloc(m->ecap * sizeof(SEnt));
}
// Both strings have 16 readable bytes from their start (padded buffers), so a short key is two masked word compares.
INL int key_eq(const char *a, const char *b, uint32_t n) {
  if (n > 16) return memcmp(a, b, n) == 0;
  uint64_t m0 = n >= 8 ? ~0ull : (1ull << (8 * n)) - 1;
  uint64_t m1 = n >= 16 ? ~0ull : n <= 8 ? 0 : (1ull << (8 * (n - 8))) - 1;
  return !(((ld8(a) ^ ld8(b)) & m0) | ((ld8(a + 8) ^ ld8(b + 8)) & m1));
}
// The slot holding the key, or the empty slot ending its probe run.
INL uint64_t smap_slot(const SMap *m, uint64_t h, const char *p, uint32_t n) {
  uint64_t i = (h >> 1) & m->mask; // hstr sets bit 0
  for (;;) {
    uint64_t v = m->s[i];
    if (!v) return i;
    if ((v & SFP) == (h & SFP)) {
      const SEnt *e = &m->e[(uint32_t)v - 1];
      if (e->n == n && key_eq(e->p, p, n)) return i;
    }
    i = (i + 1) & m->mask;
  }
}
// The key's entry, added with value v when new (*created says which).
static uint32_t smap_put(SMap *m, const char *p, uint32_t n, int32_t v, int *created) {
  if ((m->n + 1) * 2 > m->mask + 1) {
    free(m->s);
    m->mask = m->mask * 2 + 1;
    m->s = xcalloc(m->mask + 1, 8);
    for (uint32_t j = 0; j < m->n; j++) {
      uint64_t h = hstr(m->e[j].p, m->e[j].n);
      m->s[smap_slot(m, h, m->e[j].p, m->e[j].n)] = (h & SFP) | (j + 1);
    }
  }
  uint64_t h = hstr(p, n);
  uint64_t i = smap_slot(m, h, p, n);
  *created = !m->s[i];
  if (!*created) return (uint32_t)m->s[i] - 1;
  if (m->n == m->ecap) { m->ecap *= 2; m->e = xrealloc(m->e, m->ecap * sizeof(SEnt)); }
  m->e[m->n].p = p; m->e[m->n].n = n; m->e[m->n].v = v;
  m->s[i] = (h & SFP) | (m->n + 1);
  return m->n++;
}
INL int32_t smap_get(const SMap *m, const char *p, uint32_t n) {
  uint64_t v = m->s[smap_slot(m, hstr(p, n), p, n)];
  return v ? m->e[(uint32_t)v - 1].v : -1;
}

// ---------- thread pool ----------
static int NT = 8;
typedef void (*task_fn)(int tid, long task, void *ctx);
typedef struct { task_fn fn, fin; void *ctx; long ntask; const long *order; atomic_long next; } Pool;
typedef struct { Pool *p; int tid; } PArg;
static void *pool_main(void *a) {
  PArg *pa = a;
  Pool *p = pa->p;
  for (;;) {
    long t = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed);
    if (t >= p->ntask) break;
    p->fn(pa->tid, p->order ? p->order[t] : t, p->ctx);
  }
  if (p->fin) p->fin(pa->tid, pa->tid, p->ctx);
  return NULL;
}
// fin, when set, runs once per thread (task = tid) as soon as that thread finds the queue empty.
static void run_par_f(long ntask, const long *order, task_fn fn, task_fn fin, void *ctx) {
  Pool p; p.fn = fn; p.fin = fin; p.ctx = ctx; p.ntask = ntask; p.order = order; atomic_init(&p.next, 0);
  pthread_t th[64]; PArg args[64];
  for (int i = 0; i < NT; i++) { args[i].p = &p; args[i].tid = i; }
  for (int i = 1; i < NT; i++) pthread_create(&th[i], NULL, pool_main, &args[i]);
  pool_main(&args[0]);
  for (int i = 1; i < NT; i++) pthread_join(th[i], NULL);
}
static void run_par_o(long ntask, const long *order, task_fn fn, void *ctx) { run_par_f(ntask, order, fn, NULL, ctx); }
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
// Records follow the 64 B header, so a block's records start on a cache line (stage runs are whole lines).
typedef struct Blk { struct Blk *next; uint32_t n, cap; char pad[48]; } Blk;
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
// x86: parse puts PRecs and CRecs in 2^PQ times fewer coarse partitions (pcb, ccb: [NT][NCQ]), so their stage runs
// and heads stay in L1/L2 beside the read span, and split_task copies the live ones to pb/cb by origin thread after
// dedup. With PQ 0 (arm64, where stores scatter cheaply), pcb and ccb are pb and cb and nothing is split.
#ifndef PQ
#define PQ (STAGE ? 3 : 0)
#endif
static uint32_t CQB, NCQ;
static Bkt *pcb, *ccb;
static uint32_t kcap0, pcap0, ccap0, pcq0, ccq0;

// x86: records reach their bucket through a staging run per (thread, bucket), so the scattered stores land in a
// small hot buffer and the bucket gets whole runs as streaming stores (measured faster on the Xeon, slower on arm64).
// A run starts at a block start or right after a full run, so it never spans blocks, and a record's stage slot is
// its offset in the block mod the run size. Blocks start at a 64 B-aligned aalloc plus the 64 B Blk header, and a
// run is a multiple of 64 B, so every streamed run fills whole cache lines: a partly written line costs the Xeon a
// read-modify-write of the line in memory.
#ifndef STAGE // -DSTAGE=1 tests the x86 path elsewhere
#if defined(__AVX2__)
#define STAGE 1
#else
#define STAGE 0
#endif
#endif
// Run bytes for rs-byte records: the fewest whole cache lines that hold whole records (KRec 64, PRec and CRec 192),
// so a thread's stage runs stay small enough to share the L2 with its hyperthread sibling.
#define RUN(rs) ((rs) * (64 / ((rs) & -(rs))))
_Static_assert(sizeof(Blk) == 64 && (sizeof(PRec) & -sizeof(PRec)) <= 64 && (sizeof(KRec) & -sizeof(KRec)) <= 64 && (sizeof(CRec) & -sizeof(CRec)) <= 64, "stage runs must be whole cache lines");
static char *pst; // [NT][NQART] PRec runs
static char *kst; // [NT][NPART] KRec runs
static char *cst; // [NT][NQART] CRec runs
static char *pcs, *ccs; // [NT][NCQ] parse stage runs (pst and cst under PQ 0)
static char *sst; // [NT][2^PQ] split stage runs
INL void run_stream(char *d, const char *s, size_t n) {
#if defined(__AVX2__)
  for (size_t i = 0; i < n; i += 16) _mm_stream_si128((__m128i *)(d + i), _mm_loadu_si128((const __m128i *)(s + i)));
#else
  memcpy(d, s, n);
#endif
}
// Copies the partial run staged at st to the slots it was assigned: the records since the last run boundary.
static void stage_flush(Bkt *b, const char *st, uint32_t rs) {
  if (!b->tail) return;
  size_t k = (size_t)(b->cur - (char *)(b->tail + 1)) % (RUN(rs));
  memcpy(b->cur - k, st, k);
}
// Appends the rs-byte record at src to bucket b (stage run st) and returns its final address.
INL char *rec_put(Arena *a, Bkt *b, char *st, const void *src, uint32_t rs, uint32_t cap0) {
  if (UNLIKELY(b->cur == b->end)) {
    if (STAGE) stage_flush(b, st, rs);
    bkt_grow(a, b, rs, b->tail ? b->tail->cap : cap0);
  }
  char *d = b->cur;
  b->cur += rs;
  if (!STAGE) {
    memcpy(d, src, rs);
    return d;
  }
  size_t off = (size_t)(d - (char *)(b->tail + 1)) % (RUN(rs));
  memcpy(st + off, src, rs);
  if (off + rs == RUN(rs)) run_stream(b->cur - RUN(rs), st, RUN(rs));
  return d;
}

// ---------- chunks ----------
// base is the mapped file (links) or NULL with path set (events, clicks: parse_task reads the span into a buffer).
typedef struct { const char *base; size_t start, end, fsize; int kind; const char *path; } Chunk;
static Chunk *chunks; static long nchunks, chcap;
static void add_chunks(const char *base, const char *path, size_t n, int kind, size_t csz) {
  for (size_t s = 0; s < n; s += csz) {
    if (nchunks == chcap) { chcap = chcap ? chcap * 2 : 256; chunks = xrealloc(chunks, sizeof(Chunk) * chcap); }
    Chunk c = {base, s, s + csz < n ? s + csz : n, n, kind, path};
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

// Field k of a line: [fs, fe), quotes stripped; nqf counts the quotes used, and the result is nonzero when a quote
// is misplaced. Branch-free: a fifth of fields are quoted at random, so a quote branch mispredicts on most lines.
// An empty field's first byte is its separator, and e[-1] is readable since a line never starts a buffer.
INL uint32_t field(const char *ls, const char *const *sep, int k, const char **fp, uint32_t *fl, uint32_t *nqf) {
  const char *s = k ? sep[k - 1] + 1 : ls;
  const char *e = sep[k];
  uint32_t q = *s == '"';
  uint32_t bad = q & ((e - s < 2) | (e[-1] != '"'));
  s += q;
  e -= q;
  *nqf += 2 * q;
  *fp = s; *fl = (uint32_t)(e - s);
  return bad;
}

// Delay line: person lookups miss cache, so each is resolved RING lines after its prefetch.
#define RING 32
typedef struct { uint64_t uh, uref, eh; uint32_t ts, x, ingest; } EPend;
typedef struct { uint64_t uh, cref; uint32_t ts, camp; } CPend;
typedef struct { EPend e[RING]; CPend c[RING]; uint32_t ne, nc; char pad[56]; } Pend;
static Pend *pend; // [NT]
static Arena *strs; // [NT] ID strings the records refer to, copied out of the read buffer
// Copies the string ref r points at (in the read buffer, which has PAD readable bytes past every line) to strs.
INL uint64_t keep_str(int tid, uint64_t r) {
  Arena *a = &strs[tid];
  uint32_t n = refn(r);
  char *d = a->cur;
  if (UNLIKELY(d + n + 48 > a->end)) die("strings full");
  // IDs are 6-9 or 33 bytes at random, so one fixed 48 B copy spares a length branch that mispredicts on one in eight
  if (LIKELY(n <= 48)) memcpy(d, refp(r), 48);
  else memcpy(d, refp(r), n);
  a->cur = d + n;
  return mkref(d, n);
}
INL void ev_emit(int tid, const EPend *p) {
  uint64_t pk = person_of(p->uh);
  size_t bi = (size_t)tid * NCQ + (qpart(pk) >> (QB - CQB));
  PRec r = {pk, pk & LINKED ? 0 : keep_str(tid, p->uref), p->ts, p->x}; // a linked person's name is lname's
  char *d = rec_put(&arenas[tid], &pcb[bi], pcs + bi * RUN(sizeof(PRec)), &r, sizeof(PRec), pcq0);
  KRec k = {p->eh, p->ingest, (uint32_t)((d - arenas[tid].base) >> 3)};
  size_t ki = (size_t)tid * NPART + (p->eh >> (64 - PB));
  rec_put(&arenas[tid], &kb[ki], kst + ki * RUN(sizeof(KRec)), &k, sizeof(KRec), kcap0);
}
INL void cl_emit(int tid, const CPend *p) {
  uint64_t pk = person_of(p->uh);
  size_t bi = (size_t)tid * NCQ + (qpart(pk) >> (QB - CQB));
  CRec r = {pk, keep_str(tid, p->cref), p->ts, p->camp};
  rec_put(&arenas[tid], &ccb[bi], ccs + bi * RUN(sizeof(CRec)), &r, sizeof(CRec), ccq0);
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
  const char *f[7]; uint32_t fl[7]; uint32_t nqf = 0, bad = 0;
  for (int k = 0; k < 7; k++) bad |= field(ls, sep, k, &f[k], &fl[k], &nqf);
  if (bad | (nqf != nq)) return;
  if (fl[0] == 0) return;
  uint64_t eh = hstr(f[0], fl[0]);
  uint64_t uh = 0;
  if (fl[1]) {
    uh = hstr(f[1], fl[1]);
    if (have_links) __builtin_prefetch(&ltab[uh & lmask_b]);
  }
  int64_t is, ing;
  if (LIKELY(fl[3] == 20)) {
    is = ingest_s(f[3]);
    ing = is * 1000;
  } else {
    ing = parse_ts(f[3], fl[3]);
    is = ing == BADTS ? 0xFFFFFFFFll : ing / 1000;
  }
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
  if (is < 0) is = 0;
  if (is > 0xFFFFFFFFll) is = 0xFFFFFFFFll;
  // A valid event's KRec is written by ev_emit once its payload loc is known; dedup does not depend on KRec order
  // (it keeps the smallest ingest, and copies with equal ingest_ts are identical).
  if (!ok) {
    size_t ki = (size_t)tid * NPART + (eh >> (64 - PB));
    KRec k = {eh, (uint32_t)is, LOC_NONE};
    rec_put(&arenas[tid], &kb[ki], kst + ki * RUN(sizeof(KRec)), &k, sizeof(KRec), kcap0);
    return;
  }
  Pend *P = &pend[tid];
  EPend *e = &P->e[P->ne % RING];
  if (P->ne >= RING) ev_emit(tid, e);
  e->uh = uh; e->uref = mkref(f[1], fl[1]); e->eh = eh; e->ts = (uint32_t)(ts - CWS); e->x = x; e->ingest = (uint32_t)is;
  P->ne++;
}

static inline void cl_line(int tid, const char *ls, const char *const *sep, int ns, uint32_t nq) {
  if (ns != 4) return;
  const char *f[4]; uint32_t fl[4]; uint32_t nqf = 0, bad = 0;
  for (int k = 0; k < 4; k++) bad |= field(ls, sep, k, &f[k], &fl[k], &nqf);
  if (bad | (nqf != nq)) return;
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
// Input is read, not mapped: no file page faults or teardown, and each task's span stays in the thread's L2.
// 64 KB: the Xeon's 1 MB L2 serves two hyperthreads, each also streaming ~512 KB of stage runs.
#ifndef IN_CHUNK
#define IN_CHUNK (64u << 10)
#endif
#ifndef IN_MARGIN
#define IN_MARGIN 1024u // read past the span for the line crossing its end (a longer line is read again, further)
#endif
#define IN_ZERO 256u // zeroed past the read end: the tokenizer's last block and the field readers stay inside it
typedef struct { char *buf; int fd; const char *path; char pad[40]; } RBuf;
static RBuf *rbufs; // [NT]
static void read_at(int fd, char *d, size_t n, size_t off) {
  while (n) {
    ssize_t r = pread(fd, d, n, (off_t)off);
    if (r <= 0) die("pread");
    d += r; n -= (size_t)r; off += (size_t)r;
  }
}
static void parse_task(int tid, long t, void *ctx) {
  (void)ctx;
  const Chunk *f = &chunks[t];
  RBuf *rb = &rbufs[tid];
  if (rb->path != f->path) {
    if (rb->path) close(rb->fd);
    rb->fd = open(f->path, O_RDONLY);
    if (rb->fd < 0) die(f->path);
    rb->path = f->path;
  }
  // The span starts one byte early (chunk_begin looks for the newline before start) and the chunk is rebased on it.
  size_t off = f->start ? f->start - 1 : 0;
  size_t n = f->end + IN_MARGIN - off;
  if (off + n > f->fsize) n = f->fsize - off;
  char *b = rb->buf;
  read_at(rb->fd, b, n, off);
  Chunk c = {b, f->start - off, f->end - off, n, f->kind, f->path};
  char *big = NULL;
  size_t rest = f->fsize - off;
  if (n < rest && !memchr(b + c.end - 1, '\n', n - (c.end - 1)) && chunk_begin(&c) < b + c.end) {
    // a line starting in the span ends past the margin: read twice as far until it ends
    do {
      n = 2 * n < rest ? 2 * n : rest;
      big = xrealloc(big, n + PAD);
      read_at(rb->fd, big, n, off);
    } while (n < rest && !memchr(big + c.end - 1, '\n', n - (c.end - 1)));
    c.base = b = big;
    c.fsize = n;
  }
  memset(b + c.fsize, 0, IN_ZERO); // past the read end: zeros like map_file's PAD, so the end of file ends the last line
  if (c.kind == 0) { TOKENIZE((&c), EVL); }
  else { TOKENIZE((&c), CLL); }
  pend_flush(tid);
  free(big);
#if defined(__AVX2__)
  _mm_sfence(); // streamed runs are visible once the tasks are joined
#endif
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
  const char *f[2]; uint32_t fl[2]; uint32_t nqf = 0, bad = 0;
  for (int k = 0; k < 2; k++) bad |= field(ls, sep, k, &f[k], &fl[k], &nqf);
  if (bad | (nqf != nq)) return;
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
  // A span of ltab is 1 MB, so this writes every 2 MB page first: lt_insert's first touch is a load, which maps the
  // huge zero page, and the store after it splits that into 4 KB pages (a fault per page, a TLB miss per lookup).
  atomic_store_explicit(&ltab[lo].key, 0, memory_order_relaxed);
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
  for (int i = 0; i < nf; i++) add_chunks(fs[i].p, paths[i], fs[i].n, 2, 1 << 18);
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
#define DSUB 1024 // most entries per dedup sub-table, so its 2^11-2^12 slots (16-32 KB) stay in L1 (Mac: 1024 beat 512, 2048, 4096)
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
  Arena *s = &scratch[tid]; s->cur = s->base;
  uint64_t t0 = PROF ? ticks() : 0;
  // 8-byte entry: 32 hash bits as a fingerprint << 32 | the record's ref (thread << 28 | arena offset / 16, never 0
  // since a block header comes first); a fingerprint match is confirmed on the records.
  // The entries go to sub-tables by low hash bits first, each small enough to probe in L1.
  uint32_t nsub = 1;
  while (cnt / nsub > DSUB) nsub <<= 1;
  // sub-table b fills ent[start[b], start[b] + fill[b]), counted first
  uint64_t *ent = aalloc(s, cnt * 8);
  uint32_t *fill = aalloc(s, (size_t)nsub * 4); memset(fill, 0, (size_t)nsub * 4);
  uint32_t *start = aalloc(s, (size_t)nsub * 4);
  for (int t = 0; t < NT; t++) {
    for (Blk *k = kb[(size_t)t * NPART + q].head; k; k = k->next) {
      const KRec *r = (const KRec *)(k + 1);
      for (uint32_t j = 0; j < k->n; j++) fill[(uint32_t)(r[j].h >> 1) & (nsub - 1)]++; // hstr sets bit 0
    }
  }
  uint32_t at = 0;
  for (uint32_t b = 0; b < nsub; b++) {
    start[b] = at;
    at += fill[b];
  }
  uint32_t *pos = aalloc(s, (size_t)nsub * 4); memcpy(pos, start, (size_t)nsub * 4);
  for (int t = 0; t < NT; t++) {
    const char *ab = arenas[t].base;
    for (Blk *k = kb[(size_t)t * NPART + q].head; k; k = k->next) {
      KRec *r = (KRec *)(k + 1);
      uint64_t ref0 = ((uint64_t)t << 28) | (uint64_t)(((const char *)r - ab) >> 4);
      for (uint32_t j = 0; j < k->n; j++) {
        uint64_t h = r[j].h;
        uint32_t b = (uint32_t)(h >> 1) & (nsub - 1);
        ent[pos[b]++] = ((h >> 24) << 32) | (ref0 + j);
      }
    }
  }
  uint32_t most = 0;
  for (uint32_t b = 0; b < nsub; b++) most = fill[b] > most ? fill[b] : most;
  uint32_t cap = 64; while (cap < most * 2) cap <<= 1;
  uint64_t *tab = aalloc(s, (size_t)cap * 8);
  SP_ADD(tid, SP_DSPLIT, t0);
  for (uint32_t b = 0; b < nsub; b++) {
    const uint64_t *e = ent + start[b];
    uint32_t n = fill[b];
    cap = 64; while (cap < n * 2) cap <<= 1;
    uint32_t mask = cap - 1;
    memset(tab, 0, (size_t)cap * 8);
    for (uint32_t j = 0; j < n; j++) {
      uint64_t v = e[j];
      uint64_t fp = v & 0xFFFFFFFF00000000ull;
      uint32_t i = (uint32_t)(v >> 32) & mask;
      for (;;) {
        uint64_t d = tab[i];
        if (!d) { tab[i] = v; break; }
        if ((d & 0xFFFFFFFF00000000ull) == fp) {
          uint32_t dr = (uint32_t)d, vr = (uint32_t)v;
          KRec *o = (KRec *)(arenas[dr >> 28].base + ((size_t)(dr & 0x0FFFFFFFu) << 4));
          KRec *r = (KRec *)(arenas[vr >> 28].base + ((size_t)(vr & 0x0FFFFFFFu) << 4));
          if (o->h == r->h) {
            if (r->ingest < o->ingest) { kill_loc((dr & 0xF0000000u) | o->loc); tab[i] = v; }
            else kill_loc((vr & 0xF0000000u) | r->loc);
            break;
          }
        }
        i = (i + 1) & mask;
      }
    }
  }
  SP_ADD(tid, SP_DPROBE, t0);
}

// ---------- sessionize ----------
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

// LSB-first bit-packed values, as a Parquet bit-packed run stores them
typedef struct { Buf b; uint64_t acc; uint32_t nb, n; } Bits;
INL void bits_put(Bits *w, uint32_t v, uint32_t bw) {
  w->acc |= (uint64_t)v << w->nb;
  w->nb += bw;
  w->n++;
  if (w->nb >= 32) {
    bres(&w->b, 4);
    memcpy(w->b.b + w->b.n, &w->acc, 4);
    w->b.n += 4;
    w->acc >>= 32;
    w->nb -= 32;
  }
}
// Flushes the partial word and pads to whole groups of 8 values, the unit of a bit-packed run.
static void bits_end(Bits *w, uint32_t bw) {
  size_t want = (size_t)(w->n + 7) / 8 * bw;
  bres(&w->b, want - w->b.n + 8);
  memcpy(w->b.b + w->b.n, &w->acc, 8);
  memset(w->b.b + w->b.n + (w->nb + 7) / 8, 0, 8);
  w->b.n = want;
  w->acc = 0;
  w->nb = 0;
}

// user_daily rows, emitted by each sess thread as it walks: user_id, day, sessions and top_category are dictionary
// encoded (a person's rows are adjacent, so user_id is one RLE run per person over a dictionary of names), events and
// revenue_cents PLAIN.
typedef struct {
  Buf names, uidx, ev, rev; // dictionary of person names, RLE runs of its index, events, revenue_cents
  Bits day, ses, tdef, top; // day index, sessions, top_category definition bits, top_category index
  uint32_t np;
  char pad[60];
} UCol;
static UCol *ucol; // [NT]
static uint32_t DBW, CBW; // bit widths of the day and category indexes
// Sessions starting on one day are over 30 minutes apart, so a day holds at most 48: 6 bits.
#define SBW 6
// Per thread and (campaign, day), randomly hit per session and per click, so the cells hit by every session and
// click are 8 B and the purchase totals (~15% of sessions) live apart, keeping the hot table in L2: a count of
// sessions or (session, campaign) pairs on one day is at most that day's sessions, so 32 bits hold it.
typedef struct { uint32_t sessions, assisted; } CHot;
typedef struct { uint32_t purch; int64_t revenue; } CBuy;
static CHot *chot; // [NT][(ncamp+1)*NDAYS]
static CBuy *cbuy; // [NT][(ncamp+1)*NDAYS]
static uint32_t **stamp; static uint32_t *stampv;

typedef uint32_t Hist[3][2048]; // 11-bit digit counts of the high 32 bits
INL void hist_add(Hist c, uint32_t k) { c[0][k & 2047]++; c[1][(k >> 11) & 2047]++; c[2][k >> 22]++; }
// LSD radix on bits 32..63, 3 passes of 11 bits, from the digit counts of the n keys (consumed).
static uint64_t *radix3(uint64_t *a, uint64_t *tmp, uint32_t n, Hist c) {
  uint64_t *src = a, *dst = tmp;
  for (int p = 0; p < 3; p++) {
    uint32_t s = 0;
    for (int b = 0; b < 2048; b++) { uint32_t v = c[p][b]; c[p][b] = s; s += v; }
    int sh = 32 + 11 * p;
    for (uint32_t i = 0; i < n; i++) { uint32_t k = (uint32_t)(src[i] >> sh) & 2047; dst[c[p][k]++] = src[i]; }
    uint64_t *t = src; src = dst; dst = t;
  }
  return src;
}
typedef uint32_t Hist4[4][256]; // 8-bit digit counts of the high 32 bits
INL void hist4_add(Hist4 c, uint32_t k) { c[0][k & 255]++; c[1][(k >> 8) & 255]++; c[2][(k >> 16) & 255]++; c[3][k >> 24]++; }
// LSD radix on bits 32..63, 4 passes of 8 bits, from the digit counts of the n keys (consumed).
static uint64_t *radix4(uint64_t *a, uint64_t *tmp, uint32_t n, Hist4 c) {
  uint64_t *src = a, *dst = tmp;
  for (int p = 0; p < 4; p++) {
    uint32_t s = 0;
    for (int b = 0; b < 256; b++) { uint32_t v = c[p][b]; c[p][b] = s; s += v; }
    int sh = 32 + 8 * p;
    for (uint32_t i = 0; i < n; i++) { uint32_t k = (uint32_t)(src[i] >> sh) & 255; dst[c[p][k]++] = src[i]; }
    uint64_t *t = src; src = dst; dst = t;
  }
  return src;
}
// Sorts n keys by their high 32 bits, using tmp as the other buffer; returns whichever of a and tmp holds the result.
static uint64_t *sort_u64(uint64_t *a, uint64_t *tmp, uint32_t n) {
  if (n < 2) return a;
  if (n <= 48) {
    for (uint32_t i = 1; i < n; i++) { uint64_t x = a[i]; uint32_t j = i; while (j && a[j - 1] > x) { a[j] = a[j - 1]; j--; } a[j] = x; }
    return a;
  }
  if (n <= 8192) {
    Hist4 c;
    memset(c, 0, sizeof c);
    for (uint32_t i = 0; i < n; i++) hist4_add(c, (uint32_t)(a[i] >> 32));
    return radix4(a, tmp, n, c);
  }
  Hist c;
  memset(c, 0, sizeof c);
  for (uint32_t i = 0; i < n; i++) hist_add(c, (uint32_t)(a[i] >> 32));
  return radix3(a, tmp, n, c);
}
static void emit_row(UCol *U, uint32_t di, uint32_t sessions, uint32_t events, int64_t revenue, int64_t *catsum, uint32_t touched) {
  int best = -1; int64_t bv = 0;
  for (; touched; touched &= touched - 1) { // ascending category index, so a tie keeps the smallest
    int z = __builtin_ctz(touched);
    if (catsum[z] > bv) { bv = catsum[z]; best = z; }
    catsum[z] = 0;
  }
  bits_put(&U->day, di, DBW);
  bits_put(&U->ses, sessions, SBW);
  bu32(&U->ev, events);
  bi64(&U->rev, revenue);
  bits_put(&U->tdef, best >= 0, 1);
  if (best >= 0) bits_put(&U->top, (uint32_t)best, CBW);
}

// Person table: open addressing on the person key, id at the same index, so a probe is one load chain.
typedef struct { uint64_t *key; uint32_t *id; uint32_t mask; } PTab;
INL uint32_t pt_find(const PTab *t, uint64_t pk) {
  uint32_t i = (uint32_t)pk & t->mask;
  while (t->key[i] && t->key[i] != pk) i = (i + 1) & t->mask;
  return i;
}

// A person's events go to buckets of 2^BSH ms, shorter than the 30 min gap, so a session never breaks inside a
// bucket, and between consecutive non-empty buckets it breaks when the later's first ts is over 30 min past the
// earlier's last: sessions come from per-bucket min/max with no sort by ts.
#define BSH 20
typedef struct { uint32_t mn, mx, cnt, sess; } TBkt; // sess: index of the person's session, set by the walk

static void sess_task(int tid, long q, void *ctx) {
  (void)ctx;
  Arena *S = &scratch[tid]; S->cur = S->base;
  size_t ne = 0, nc = 0;
  for (int t = 0; t < NT; t++) {
    for (Blk *k = pb[(size_t)t * NQART + q].head; k; k = k->next) ne += k->n;
    for (Blk *k = cb[(size_t)t * NQART + q].head; k; k = k->next) nc += k->n;
  }
  if (!ne) return;
  uint64_t t0 = PROF ? ticks() : 0;
  // one pass over the records: dense person id, per person counts, (id << 32 | ts) per event, and the purchases
  // apart (~3% of events), so the per-event arrays hold no price
  // sized for ~16 events per person (the mean is ~30); it doubles past half full
  PTab pt;
  uint32_t pcap = 256; while (pcap < ne / 8) pcap <<= 1;
  pt.key = aalloc(S, (size_t)pcap * 8); memset(pt.key, 0, (size_t)pcap * 8);
  pt.id = aalloc(S, (size_t)pcap * 4);
  pt.mask = pcap - 1;
  uint32_t np = 0;
  uint64_t *pkeys = aalloc(S, ne * 8);
  uint64_t *names = aalloc(S, ne * 8);
  uint32_t *eoff = aalloc(S, (ne + 1) * 4);
  eoff[0] = 0;
  uint64_t *g = aalloc(S, ne * 8); // id << 32 | ts
  uint64_t *ptx = aalloc(S, ne * 8); // purchases: ts << 32 | x
  uint32_t *ppid = aalloc(S, ne * 4);
  size_t k = 0, npu = 0;
  for (int t = 0; t < NT; t++) {
    for (Blk *bk = pb[(size_t)t * NQART + q].head; bk; bk = bk->next) {
      PRec *r = (PRec *)(bk + 1);
      for (uint32_t j = 0; j < bk->n; j++) {
        uint32_t x = r[j].x;
        if (x & X_DEAD) continue;
        uint64_t pk = r[j].pkey;
        uint32_t i = pt_find(&pt, pk);
        if (!pt.key[i]) {
          if (UNLIKELY((np + 1) * 2 > pcap)) { // grow
            pcap *= 2;
            pt.key = aalloc(S, (size_t)pcap * 8); memset(pt.key, 0, (size_t)pcap * 8);
            pt.id = aalloc(S, (size_t)pcap * 4);
            pt.mask = pcap - 1;
            for (uint32_t z = 0; z < np; z++) {
              uint32_t gi = pt_find(&pt, pkeys[z]);
              pt.key[gi] = pkeys[z]; pt.id[gi] = z;
            }
            i = pt_find(&pt, pk);
          }
          pt.key[i] = pk; pt.id[i] = np;
          pkeys[np] = pk;
          names[np] = pk & LINKED ? lname[(uint32_t)pk] : r[j].uref;
          np++;
          eoff[np] = 0;
        }
        uint32_t id = pt.id[i];
        eoff[id + 1]++;
        g[k] = ((uint64_t)id << 32) | r[j].ts;
        k++;
        if (x & X_PURCH) {
          ptx[npu] = ((uint64_t)r[j].ts << 32) | x;
          ppid[npu] = id;
          npu++;
        }
      }
    }
  }
  size_t nalive = k;
  SP_ADD(tid, SP_GATHER, t0);
  for (uint32_t i = 0; i < np; i++) eoff[i + 1] += eoff[i];
  uint32_t *pos = aalloc(S, ((size_t)np + 1) * 4); memcpy(pos, eoff, ((size_t)np + 1) * 4);
  uint32_t *ev = aalloc(S, nalive * 4);
  for (size_t i = 0; i < nalive; i++) ev[pos[g[i] >> 32]++] = (uint32_t)g[i];
  uint32_t *poff = aalloc(S, ((size_t)np + 1) * 4); memset(poff, 0, ((size_t)np + 1) * 4);
  for (size_t i = 0; i < npu; i++) poff[ppid[i] + 1]++;
  for (uint32_t i = 0; i < np; i++) poff[i + 1] += poff[i];
  memcpy(pos, poff, ((size_t)np + 1) * 4);
  uint64_t *pur = aalloc(S, npu * 8 + 8); // per person runs of ts << 32 | x
  for (size_t i = 0; i < npu; i++) pur[pos[ppid[i]]++] = ptx[i];
  SP_ADD(tid, SP_SORT, t0);
  // clicks: per person runs of (ts << 32 | click slot); ts ties are settled by click_id at credit time
  uint32_t *cpid = aalloc(S, (nc + 1) * 4);
  uint64_t *cref = aalloc(S, (nc + 1) * 8);
  uint32_t *ccamp = aalloc(S, (nc + 1) * 4);
  uint64_t *ctx_ = aalloc(S, (nc + 1) * 8);
  uint64_t *tmp = aalloc(S, (nc + 1) * 8);
  uint32_t *coff = aalloc(S, ((size_t)np + 1) * 4); memset(coff, 0, ((size_t)np + 1) * 4);
  // the click sort digits are counted here too, for the radix sort_u64 would pick for nc keys
  int cbig = nc > 8192, cmid = nc > 48 && !cbig;
  Hist *ch = aalloc(S, sizeof(Hist));
  Hist4 *ch4 = aalloc(S, sizeof(Hist4));
  if (cbig) memset(ch, 0, sizeof(Hist));
  if (cmid) memset(ch4, 0, sizeof(Hist4));
  k = 0;
  for (int t = 0; t < NT; t++) {
    for (Blk *bk = cb[(size_t)t * NQART + q].head; bk; bk = bk->next) {
      CRec *r = (CRec *)(bk + 1);
      for (uint32_t j = 0; j < bk->n; j++) {
        uint32_t i = pt_find(&pt, r[j].pkey);
        if (!pt.key[i]) continue;
        uint32_t id = pt.id[i];
        cpid[k] = id;
        cref[k] = r[j].cref;
        ccamp[k] = r[j].camp;
        ctx_[k] = ((uint64_t)r[j].ts << 32) | k;
        if (cbig) hist_add(*ch, r[j].ts);
        if (cmid) hist4_add(*ch4, r[j].ts);
        coff[id + 1]++;
        k++;
      }
    }
  }
  size_t nlive = k;
  for (uint32_t i = 0; i < np; i++) coff[i + 1] += coff[i];
  memcpy(pos, coff, ((size_t)np + 1) * 4);
  uint64_t *cl = aalloc(S, (nlive + 1) * 8);
  cl[nlive] = 0;
  const uint64_t *csorted = cbig ? radix3(ctx_, tmp, (uint32_t)nlive, *ch)
                          : cmid ? radix4(ctx_, tmp, (uint32_t)nlive, *ch4) : sort_u64(ctx_, tmp, (uint32_t)nlive);
  for (size_t i = 0; i < nlive; i++) cl[pos[cpid[(uint32_t)csorted[i]]]++] = csorted[i];
  SP_ADD(tid, SP_CLICKS, t0);
  UCol *U = &ucol[tid];
  CHot *hot = &chot[(size_t)tid * (ncamp + 1) * NDAYS];
  CBuy *cy = &cbuy[(size_t)tid * (ncamp + 1) * NDAYS];
  uint32_t *st = stamp[tid];
  // event ts are offsets from CWS; buckets count from the event window start
  uint32_t ebase = (uint32_t)(WS - CWS);
  uint32_t nb = (uint32_t)((WE - 1 - WS) >> BSH) + 1;
  TBkt *tb = aalloc(S, (size_t)nb * sizeof(TBkt));
  for (uint32_t b = 0; b < nb; b++) { tb[b].mn = UINT32_MAX; tb[b].mx = 0; tb[b].cnt = 0; }
  uint64_t *bits = aalloc(S, ((size_t)nb / 64 + 1) * 8); memset(bits, 0, ((size_t)nb / 64 + 1) * 8);
  int64_t *catd = aalloc(S, (size_t)NDAYS * 32 * 8); memset(catd, 0, (size_t)NDAYS * 32 * 8);
  uint32_t *dmask = aalloc(S, (size_t)NDAYS * 4); memset(dmask, 0, (size_t)NDAYS * 4);
  // the person's sessions: start ts, day index, events, revenue, and the end of its day's run
  uint32_t *sst = aalloc(S, nalive * 4);
  uint32_t *sdi = aalloc(S, nalive * 4);
  uint32_t *scnt = aalloc(S, nalive * 4);
  int64_t *srev = aalloc(S, nalive * 8);
  uint32_t *drun = aalloc(S, nalive * 4);
  uint32_t *scr = aalloc(S, nalive * 4); // the campaign each session is credited to, ncamp for organic
  uint32_t *jprev = aalloc(S, ((size_t)ncamp + 1) * 4); // valid where st[] carries this person's stamp
  for (uint32_t p = 0; p < np; p++) {
    const uint32_t *e = ev + eoff[p];
    uint32_t n = eoff[p + 1] - eoff[p];
    const uint64_t *pu = pur + poff[p];
    uint32_t npur = poff[p + 1] - poff[p];
    uint64_t *c = cl + coff[p];
    uint32_t m = coff[p + 1] - coff[p];
    uint32_t rows0 = U->day.n;
    // a name is one cache miss into the strings parse wrote (a third of the walk on arm64): fetch it 16 persons ahead
    __builtin_prefetch(refp(names[p + 16 < np ? p + 16 : p]));
    uint32_t nn = refn(names[p]);
    bres(&U->names, 4 + nn);
    memcpy(U->names.b + U->names.n, &nn, 4);
    memcpy(U->names.b + U->names.n + 4, refp(names[p]), nn);
    U->names.n += 4 + nn;
    uint32_t blo = nb, bhi = 0;
    for (uint32_t i = 0; i < n; i++) {
      uint32_t t = e[i];
      uint32_t b = (t - ebase) >> BSH;
      TBkt *B = &tb[b];
      B->mn = t < B->mn ? t : B->mn;
      B->mx = t > B->mx ? t : B->mx;
      B->cnt++;
      bits[b >> 6] |= 1ull << (b & 63);
      blo = b < blo ? b : blo;
      bhi = b > bhi ? b : bhi;
    }
    // buckets in ts order -> sessions
    uint32_t ns = 0, last = 0;
    for (uint32_t w = blo >> 6; w <= bhi >> 6; w++) {
      uint64_t bw = bits[w];
      bits[w] = 0;
      for (; bw; bw &= bw - 1) {
        TBkt *B = &tb[w * 64 + (uint32_t)__builtin_ctzll(bw)];
        if (!ns || B->mn - last > 1800000u) {
          sst[ns] = B->mn;
          sdi[ns] = (uint32_t)((CWS + (int64_t)B->mn) / 86400000LL - DAY0);
          scnt[ns] = 0;
          srev[ns] = 0;
          ns++;
        }
        last = B->mx;
        scnt[ns - 1] += B->cnt;
        B->sess = ns - 1;
        B->mn = UINT32_MAX; B->mx = 0; B->cnt = 0;
      }
    }
    // purchases count for their session's revenue and the category of its day
    for (uint32_t i = 0; i < npur; i++) {
      uint32_t t = (uint32_t)(pu[i] >> 32);
      uint32_t x = (uint32_t)pu[i];
      uint32_t z = tb[(t - ebase) >> BSH].sess;
      uint32_t d = sdi[z];
      uint32_t cat = (x >> 24) & 31;
      srev[z] += x & 0xFFFFFF;
      catd[(size_t)d * 32 + cat] += x & 0xFFFFFF;
      dmask[d] |= 1u << cat;
    }
    // One merge walk over the clicks in ts order. Credit: when a click comes past a session's start, the clicks
    // before it are all at or before that start, so the latest of them is the session's click if it is within
    // 7 days (among equal ts the smallest click_id, settled once per run of equal ts when a session asks).
    // Assists: a click at t assists the sessions starting in [t, t + 7 days]; per campaign only the sessions past
    // the previous click's range count, so each (session, campaign) pair counts once.
    uint32_t ip = 0;
    if (m) {
      drun[ns - 1] = ns;
      for (uint32_t z = ns - 1; z-- > 0;) drun[z] = sdi[z] == sdi[z + 1] ? drun[z + 1] : z + 1;
      uint32_t sv = ++stampv[tid];
      uint32_t jp = 0, g0 = 0, gbest = UINT32_MAX; // g0: first click of the equal-ts run ending at the last click
      for (uint32_t z = 0; z <= m; z++) {
        uint32_t t = z < m ? (uint32_t)(c[z] >> 32) : UINT32_MAX;
        for (; ip < ns && (z == m || sst[ip] < t); ip++) {
          if (!z) { scr[ip] = (uint32_t)ncamp; continue; }
          uint32_t ct = (uint32_t)(c[z - 1] >> 32);
          uint32_t best = (uint32_t)c[z - 1];
          if (UNLIKELY(g0 != z - 1)) {
            if (gbest == UINT32_MAX) {
              gbest = best;
              for (uint32_t y = g0; y < z - 1; y++) {
                uint32_t o = (uint32_t)c[y];
                if (str_cmp(refp(cref[o]), refn(cref[o]), refp(cref[gbest]), refn(cref[gbest])) < 0) gbest = o;
              }
            }
            best = gbest;
          }
          scr[ip] = (int64_t)ct >= (int64_t)sst[ip] - 604800000LL ? ccamp[best] : (uint32_t)ncamp;
        }
        if (z == m) break;
        if (z && t != (uint32_t)(c[z - 1] >> 32)) { g0 = z; gbest = UINT32_MAX; }
        uint32_t cc = ccamp[(uint32_t)c[z]];
        while (jp < ns && sst[jp] <= (int64_t)t + 604800000LL) jp++;
        uint32_t from = ip;
        if (st[cc] != sv) st[cc] = sv;
        else if (jprev[cc] > from) from = jprev[cc];
        jprev[cc] = jp;
        while (from < jp) {
          uint32_t to = drun[from] < jp ? drun[from] : jp;
          hot[(size_t)cc * NDAYS + sdi[from]].assisted += to - from;
          from = to;
        }
      }
    }
    for (; ip < ns; ip++) scr[ip] = (uint32_t)ncamp;
    // sessions -> campaign aggregates, one row per day
    uint32_t rses = 0, revs = 0;
    int64_t rrev = 0;
    for (uint32_t z = 0; z < ns; z++) {
      uint32_t di = sdi[z];
      if (z && di != sdi[z - 1]) {
        uint32_t pd = sdi[z - 1];
        emit_row(U, pd, rses, revs, rrev, catd + (size_t)pd * 32, dmask[pd]);
        dmask[pd] = 0;
        rses = 0;
        revs = 0;
        rrev = 0;
      }
      rses++;
      revs += scnt[z];
      rrev += srev[z];
      size_t ci = (size_t)scr[z] * NDAYS + di;
      hot[ci].sessions++;
      if (srev[z]) { cy[ci].purch++; cy[ci].revenue += srev[z]; }
    }
    uint32_t ld = sdi[ns - 1];
    emit_row(U, ld, rses, revs, rrev, catd + (size_t)ld * 32, dmask[ld]);
    dmask[ld] = 0;
    bvar(&U->uidx, (uint64_t)(U->day.n - rows0) << 1);
    bu32(&U->uidx, U->np++);
  }
  SP_ADD(tid, SP_WALK, t0);
}

// ---------- parquet writer ----------

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
// off: the data page; start: the chunk, at the dictionary page when dict
typedef struct { Buf data; int64_t nvals; int64_t off, start; int64_t size; int dict; } ColOut;

// Appends the header of a data page with body bytes (levels + values) holding nvals values.
static void page_hdr(Buf *file, size_t body, int64_t nvals, int enc) {
  TW w = {file, {0}, 0};
  tw_i32(&w, 1, 0);
  tw_i32(&w, 2, (int32_t)body);
  tw_i32(&w, 3, (int32_t)body);
  tw_sbeg(&w, 5);
  tw_i32(&w, 1, (int32_t)nvals);
  tw_i32(&w, 2, enc);
  tw_i32(&w, 3, 3);
  tw_i32(&w, 4, 3);
  tw_send(&w);
  bbyte(file, 0);
}
// Appends the header of a PLAIN dictionary page of nvals values in body bytes.
static void dict_hdr(Buf *file, size_t body, int64_t nvals) {
  TW w = {file, {0}, 0};
  tw_i32(&w, 1, 2);
  tw_i32(&w, 2, (int32_t)body);
  tw_i32(&w, 3, (int32_t)body);
  tw_sbeg(&w, 7);
  tw_i32(&w, 1, (int32_t)nvals);
  tw_i32(&w, 2, 0);
  tw_send(&w);
  bbyte(file, 0);
}
static void write_page(Buf *file, ColOut *co, int64_t nvals) {
  co->off = (int64_t)file->n;
  co->start = co->off;
  page_hdr(file, co->data.n, nvals, 0);
  bput(file, co->data.b, co->data.n);
  co->size = (int64_t)(file->n) - co->off;
  co->nvals = nvals;
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
      tw_i64(&w, 2, co[i].start);
      tw_sbeg(&w, 3);
      tw_i32(&w, 1, cols[i].type);
      tw_lbeg(&w, 2, 5, co[i].dict ? 3 : 2); bvar(&meta, zz(0)); bvar(&meta, zz(3));
      if (co[i].dict) bvar(&meta, zz(8));
      tw_lbeg(&w, 3, 8, 1); bvar(&meta, strlen(cols[i].name)); bput(&meta, cols[i].name, strlen(cols[i].name));
      tw_i32(&w, 4, 0);
      tw_i64(&w, 5, co[i].nvals);
      tw_i64(&w, 6, co[i].size);
      tw_i64(&w, 7, co[i].size);
      tw_i64(&w, 9, co[i].off);
      if (co[i].dict) tw_i64(&w, 11, co[i].start);
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

static const ColDef UCOLS[6] = {
  {"user_id", PT_BYTES, LT_STRING, 0}, {"day", PT_INT32, LT_DATE, 0}, {"sessions", PT_INT32, 0, 0},
  {"events", PT_INT32, 0, 0}, {"revenue_cents", PT_INT64, 0, 0}, {"top_category", PT_BYTES, LT_STRING, 1}};
static const ColDef CCOLS[7] = {
  {"campaign_id", PT_BYTES, LT_STRING, 0}, {"channel", PT_BYTES, LT_STRING, 0}, {"day", PT_INT32, LT_DATE, 0},
  {"sessions", PT_INT64, 0, 0}, {"purchasing_sessions", PT_INT64, 0, 0}, {"revenue_cents", PT_INT64, 0, 0},
  {"assisted_sessions", PT_INT64, 0, 0}};

static char *OUTDIR;

// The file goes out in one writev: page headers and footer gathered in one buffer, page bodies straight from the
// thread's column buffers.
typedef struct { Buf h; struct iovec v[64]; size_t hoff[64]; int nv; size_t pos, mark; } FileW;
static void fw_hdr(FileW *f) { // the header bytes since the last call
  f->hoff[f->nv] = f->mark;
  f->v[f->nv].iov_base = NULL;
  f->v[f->nv].iov_len = f->h.n - f->mark;
  f->nv++;
  f->pos += f->h.n - f->mark;
  f->mark = f->h.n;
}
static void fw_ext(FileW *f, const void *p, size_t n) {
  f->v[f->nv].iov_base = (void *)p;
  f->v[f->nv].iov_len = n;
  f->nv++;
  f->pos += n;
}
static size_t vlen(uint64_t v) { size_t n = 1; while (v >= 0x80) { v >>= 7; n++; } return n; }

// A dictionary column chunk: the dictionary page (nd values in dict), then one data page of n values whose body is
// pre (header bytes, already sized into body) followed by the bit-packed indexes in idx.
static void fw_dict(FileW *f, ColOut *co, const Buf *dict, int64_t nd, size_t n) {
  co->dict = 1;
  co->start = (int64_t)f->pos;
  dict_hdr(&f->h, dict->n, nd);
  fw_hdr(f);
  fw_ext(f, dict->b, dict->n);
  co->off = (int64_t)f->pos;
  co->nvals = (int64_t)n;
}
// Data page of a required dictionary column whose indexes are one bit-packed run.
static void fw_packed(FileW *f, ColOut *co, const Buf *dict, int64_t nd, const Bits *idx, uint32_t bw) {
  size_t n = idx->n;
  fw_dict(f, co, dict, nd, n);
  uint64_t run = ((uint64_t)(n + 7) / 8) << 1 | 1;
  page_hdr(&f->h, 1 + vlen(run) + idx->b.n, (int64_t)n, 8);
  bbyte(&f->h, (uint8_t)bw);
  bvar(&f->h, run);
  fw_hdr(f);
  fw_ext(f, idx->b.b, idx->b.n);
  co->size = (int64_t)f->pos - co->start;
}

static void write_file(const char *path, FileW *f) {
  for (int i = 0; i < f->nv; i++) if (!f->v[i].iov_base) f->v[i].iov_base = f->h.b + f->hoff[i];
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) die(path);
  struct iovec *v = f->v;
  int nv = f->nv;
  while (nv) {
    ssize_t w = writev(fd, v, nv);
    if (w < 0) { if (errno == EINTR) continue; die("write"); }
    while (nv && (size_t)w >= v->iov_len) { w -= (ssize_t)v->iov_len; v++; nv--; }
    if (nv) { v->iov_base = (char *)v->iov_base + w; v->iov_len -= (size_t)w; }
  }
  close(fd);
}

static void write_user_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  UCol *U = &ucol[t];
  size_t n = U->day.n;
  if (!n && t != 0) return;
  bits_end(&U->day, DBW);
  bits_end(&U->ses, SBW);
  bits_end(&U->tdef, 1);
  bits_end(&U->top, CBW);
  Buf dday = {0}, dses = {0}, dtop = {0};
  for (int32_t d = 0; d < NDAYS; d++) bu32(&dday, (uint32_t)(DAY0 + d));
  for (uint32_t v = 0; v < 1u << SBW; v++) bu32(&dses, v);
  for (int c = 0; c < ncat; c++) { bu32(&dtop, catn[c]); bput(&dtop, catp[c], catn[c]); }
  FileW f;
  memset(&f, 0, sizeof f);
  ColOut co[6];
  memset(co, 0, sizeof co);
  bput(&f.h, "PAR1", 4);
  fw_hdr(&f);
  if (n) {
    fw_dict(&f, &co[0], &U->names, U->np, n);
    page_hdr(&f.h, 1 + U->uidx.n, (int64_t)n, 8);
    bbyte(&f.h, 32);
    fw_hdr(&f);
    fw_ext(&f, U->uidx.b, U->uidx.n);
    co[0].size = (int64_t)f.pos - co[0].start;
    fw_packed(&f, &co[1], &dday, NDAYS, &U->day, DBW);
    fw_packed(&f, &co[2], &dses, 1u << SBW, &U->ses, SBW);
    co[3].start = co[3].off = (int64_t)f.pos;
    page_hdr(&f.h, U->ev.n, (int64_t)n, 0);
    fw_hdr(&f);
    fw_ext(&f, U->ev.b, U->ev.n);
    co[3].size = (int64_t)f.pos - co[3].start;
    co[3].nvals = (int64_t)n;
    co[4].start = co[4].off = (int64_t)f.pos;
    page_hdr(&f.h, U->rev.n, (int64_t)n, 0);
    fw_hdr(&f);
    fw_ext(&f, U->rev.b, U->rev.n);
    co[4].size = (int64_t)f.pos - co[4].start;
    co[4].nvals = (int64_t)n;
    // top_category: definition levels as one bit-packed run (4-byte length first), then the non-null indexes
    fw_dict(&f, &co[5], &dtop, ncat, n);
    uint64_t lrun = ((uint64_t)(n + 7) / 8) << 1 | 1;
    uint64_t vrun = ((uint64_t)(U->top.n + 7) / 8) << 1 | 1;
    size_t lev = vlen(lrun) + U->tdef.b.n;
    size_t vals = 1 + (U->top.n ? vlen(vrun) + U->top.b.n : 0);
    page_hdr(&f.h, 4 + lev + vals, (int64_t)n, 8);
    bu32(&f.h, (uint32_t)lev);
    bvar(&f.h, lrun);
    fw_hdr(&f);
    fw_ext(&f, U->tdef.b.b, U->tdef.b.n);
    bbyte(&f.h, (uint8_t)CBW);
    if (U->top.n) bvar(&f.h, vrun);
    fw_hdr(&f);
    fw_ext(&f, U->top.b.b, U->top.b.n);
    co[5].size = (int64_t)f.pos - co[5].start;
  }
  write_footer(&f.h, UCOLS, 6, co, (int64_t)n);
  fw_hdr(&f);
  char path[4096];
  snprintf(path, sizeof path, "%s/user_daily/part-%03ld.parquet", OUTDIR, t);
  write_file(path, &f);
}

static void write_campaign(void) {
  ColOut co[7]; memset(co, 0, sizeof co);
  int64_t n = 0;
  size_t stride = (size_t)(ncamp + 1) * NDAYS;
  for (int c = 0; c <= ncamp; c++) {
    for (int d = 0; d < NDAYS; d++) {
      int64_t sessions = 0, purch = 0, revenue = 0, assisted = 0;
      for (int t = 0; t < NT; t++) {
        size_t i = (size_t)t * stride + (size_t)c * NDAYS + d;
        sessions += chot[i].sessions;
        assisted += chot[i].assisted;
        purch += cbuy[i].purch;
        revenue += cbuy[i].revenue;
      }
      if (!sessions && !assisted) continue;
      const char *id = c == ncamp ? "organic" : camp_id[c];
      uint32_t idn = c == ncamp ? 7 : camp_idn[c];
      const char *ch = c == ncamp ? "organic" : chan[c];
      uint32_t chn = c == ncamp ? 7 : chann[c];
      bu32(&co[0].data, idn); bput(&co[0].data, id, idn);
      bu32(&co[1].data, chn); bput(&co[1].data, ch, chn);
      int32_t day = (int32_t)(DAY0 + d); bput(&co[2].data, &day, 4);
      bi64(&co[3].data, sessions);
      bi64(&co[4].data, purch);
      bi64(&co[5].data, revenue);
      bi64(&co[6].data, c == ncamp ? 0 : assisted);
      n++;
    }
  }
  Buf file = {0};
  bput(&file, "PAR1", 4);
  if (n) for (int c = 0; c < 7; c++) write_page(&file, &co[c], n);
  write_footer(&file, CCOLS, 7, co, n);
  char path[4096];
  snprintf(path, sizeof path, "%s/campaign_daily/part-000.parquet", OUTDIR);
  FileW f;
  memset(&f, 0, sizeof f);
  fw_ext(&f, file.b, file.n);
  write_file(path, &f);
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
  SMap cm; smap_init(&cm);
  size_t lines = f.n / 8 + 16;
  smap_init(&prodmap);
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
  static const char unk[16] = "unknown"; // 16 readable bytes for key_eq
  smap_put(&cm, unk, 7, 0, &cr);
  if (cr) { if (nc == ccap) { ccap *= 2; cats = xrealloc(cats, sizeof(Str) * ccap); } cats[nc].p = unk; cats[nc].n = 7; nc++; }
  qsort(cats, nc, sizeof(Str), cmp_str);
  ncat = nc;
  if (ncat > 31) die("too many categories");
  catp = xmalloc(sizeof(char *) * ncat); catn = xmalloc(sizeof(uint32_t) * ncat);
  for (int i = 0; i < ncat; i++) {
    catp[i] = cats[i].p; catn[i] = cats[i].n;
    cm.e[smap_put(&cm, cats[i].p, cats[i].n, 0, &cr)].v = i;
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
  smap_init(&campmap);
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
  for (uint32_t i = 0; i < NPART; i++) {
    size_t bi = (size_t)t * NPART + i;
    if (STAGE) stage_flush(&kb[bi], kst + bi * RUN(sizeof(KRec)), sizeof(KRec));
    bkt_close(&kb[bi], sizeof(KRec));
  }
  for (uint32_t i = 0; i < NCQ; i++) {
    size_t bi = (size_t)t * NCQ + i;
    if (STAGE) stage_flush(&pcb[bi], pcs + bi * RUN(sizeof(PRec)), sizeof(PRec));
    if (STAGE) stage_flush(&ccb[bi], ccs + bi * RUN(sizeof(CRec)), sizeof(CRec));
    bkt_close(&pcb[bi], sizeof(PRec));
    bkt_close(&ccb[bi], sizeof(CRec));
  }
}

// Coarse partition c, after dedup: each origin thread's live records go to its buckets of c's 2^PQ partitions.
static void split_task(int tid, long c, void *ctx) {
  (void)ctx;
  uint32_t nf = 1u << PQ;
  for (int t = 0; t < NT; t++) {
    size_t b0 = (size_t)t * NQART + ((size_t)c << PQ);
    for (Blk *k = pcb[(size_t)t * NCQ + c].head; k; k = k->next) {
      const PRec *r = (const PRec *)(k + 1);
      for (uint32_t j = 0; j < k->n; j++) {
        if (r[j].x & X_DEAD) continue;
        uint32_t f = qpart(r[j].pkey) & (nf - 1);
        rec_put(&arenas[tid], &pb[b0 + f], sst + ((size_t)tid * nf + f) * RUN(sizeof(PRec)), &r[j], sizeof(PRec), pcap0);
      }
    }
    for (uint32_t f = 0; f < nf; f++) {
      if (STAGE) stage_flush(&pb[b0 + f], sst + ((size_t)tid * nf + f) * RUN(sizeof(PRec)), sizeof(PRec));
      bkt_close(&pb[b0 + f], sizeof(PRec));
    }
    for (Blk *k = ccb[(size_t)t * NCQ + c].head; k; k = k->next) {
      const CRec *r = (const CRec *)(k + 1);
      for (uint32_t j = 0; j < k->n; j++) {
        uint32_t f = qpart(r[j].pkey) & (nf - 1);
        rec_put(&arenas[tid], &cb[b0 + f], sst + ((size_t)tid * nf + f) * RUN(sizeof(CRec)), &r[j], sizeof(CRec), ccap0);
      }
    }
    for (uint32_t f = 0; f < nf; f++) {
      if (STAGE) stage_flush(&cb[b0 + f], sst + ((size_t)tid * nf + f) * RUN(sizeof(CRec)), sizeof(CRec));
      bkt_close(&cb[b0 + f], sizeof(CRec));
    }
  }
#if defined(__AVX2__)
  _mm_sfence();
#endif
}

// Drop pages in parallel (madvise takes the mmap lock shared) so exit has little left to tear down serially.
// -DRELEASE=0 leaves all of it to exit.
#ifndef RELEASE
#define RELEASE 1
#endif
static void release_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  madvise(arenas[t].base, (size_t)(arenas[t].cur - arenas[t].base), MADV_DONTNEED);
  madvise(strs[t].base, (size_t)(strs[t].cur - strs[t].base), MADV_DONTNEED);
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
  size_t *ef = xmalloc(sizeof(size_t) * (ne + 1)), *cf = xmalloc(sizeof(size_t) * (ncl + 1));
  for (int i = 0; i < ne; i++) { ef[i] = file_size(ep[i]); ebytes += ef[i]; }
  for (int i = 0; i < ncl; i++) { cf[i] = file_size(cp[i]); cbytes += cf[i]; }
  for (int i = 0; i < ne; i++) add_chunks(NULL, ep[i], ef[i], 0, IN_CHUNK);
  for (int i = 0; i < ncl; i++) add_chunks(NULL, cp[i], cf[i], 1, IN_CHUNK);
  // partition sizing
  uint64_t eest = ebytes / 60 + 1024, cest = cbytes / 45 + 1024;
  // eest runs ~1.4x the real event count, so 48000 here is ~30-35K events per partition: 2^10 at 20M, 2^11 at
  // 40-64M. Board, 64M-event replica: PB=QB 11 2975 ms, 10 3020, 12 3172, 13 3490 (diag a2e7094).
  QB = 6; while (QB < 13 && (eest >> QB) > 48000) QB++;
  // Dedup keys go to 8x fewer partitions (256 at QB 11): their stage runs and bucket heads, 24 KB per thread, stay
  // in L1 while parse scatters, and dedup splits each partition into L1 sub-tables anyway.
#ifndef PBQ
#define PBQ 3
#endif
  PB = QB >= 6 + PBQ ? QB - PBQ : 6;
  NPART = 1u << PB; NQART = 1u << QB;
  uint64_t per_t = eest / NT;
  // Blocks of an eighth of a bucket's expected share: a bucket's last block is the only unused memory (and the
  // kernel zeroes all of it), so this wastes ~6% where one block of the whole share wasted ~75% (eest runs high).
  kcap0 = (uint32_t)(per_t / NPART / 8 + 32);
  pcap0 = (uint32_t)(per_t / NQART / 8 + 32);
  ccap0 = (uint32_t)(cest / NT / NQART / 16 + 32);
  CQB = QB > PQ + 2 ? QB - PQ : QB;
  NCQ = 1u << CQB;
  pcq0 = (uint32_t)(per_t / NCQ / 8 + 32);
  ccq0 = (uint32_t)(cest / NT / NCQ / 16 + 32);
  arenas = xcalloc(NT, sizeof(Arena)); scratch = xcalloc(NT, sizeof(Arena));
  for (int t = 0; t < NT; t++) { arena_init(&arenas[t], ARENA_SZ); arena_init(&scratch[t], 1ull << 30); }
  strs = xcalloc(NT, sizeof(Arena)); rbufs = xcalloc(NT, sizeof(RBuf));
  for (int t = 0; t < NT; t++) {
    arena_init(&strs[t], ebytes + cbytes + 64); // a copied ID is shorter than its line
    rbufs[t].buf = aligned_alloc(64, (IN_CHUNK + IN_MARGIN + PAD + 127) & ~63u); // the span starts a byte early
    if (!rbufs[t].buf) die("oom");
  }
  pend = xcalloc(NT, sizeof(Pend));
  kb = xcalloc((size_t)NT * NPART, sizeof(Bkt));
  pb = xcalloc((size_t)NT * NQART, sizeof(Bkt));
  cb = xcalloc((size_t)NT * NQART, sizeof(Bkt));
  if (STAGE) pst = hcalloc((size_t)NT * NQART * RUN(sizeof(PRec)));
  if (STAGE) kst = hcalloc((size_t)NT * NPART * RUN(sizeof(KRec)));
  if (STAGE) cst = hcalloc((size_t)NT * NQART * RUN(sizeof(CRec)));
  pcb = pb; ccb = cb; pcs = pst; ccs = cst;
  if (CQB < QB) {
    pcb = xcalloc((size_t)NT * NCQ, sizeof(Bkt));
    ccb = xcalloc((size_t)NT * NCQ, sizeof(Bkt));
    if (STAGE) pcs = hcalloc((size_t)NT * NCQ * RUN(sizeof(PRec)));
    if (STAGE) ccs = hcalloc((size_t)NT * NCQ * RUN(sizeof(CRec)));
    if (STAGE) sst = hcalloc((size_t)NT * (1u << PQ) * RUN(sizeof(PRec)));
  }
  PH("map");
  run_par(nchunks, parse_task, NULL);
  run_par(NT, close_task, NULL);
  PH("parse");
  double pt0 = now_s(); uint64_t pk0 = ticks();
  run_par(NPART, dedup_task, NULL);
  if (CQB < QB) run_par(NCQ, split_task, NULL);
  PH("dedup");
  ucol = xcalloc(NT, sizeof(UCol));
  DBW = 1; while ((1u << DBW) < (uint32_t)NDAYS) DBW++;
  CBW = 1; while ((1u << CBW) < (uint32_t)ncat) CBW++;
  chot = xcalloc((size_t)NT * (ncamp + 1) * NDAYS, sizeof(CHot));
  cbuy = xcalloc((size_t)NT * (ncamp + 1) * NDAYS, sizeof(CBuy));
  stamp = xmalloc(sizeof(uint32_t *) * NT);
  stampv = xcalloc(NT, sizeof(uint32_t));
  for (int t = 0; t < NT; t++) stamp[t] = xcalloc(ncamp + 1, sizeof(uint32_t));
  // largest person partitions first
  qsize = xcalloc(NQART, 8);
  for (int t = 0; t < NT; t++) for (uint32_t q = 0; q < NQART; q++) for (Blk *k = pb[(size_t)t * NQART + q].head; k; k = k->next) qsize[q] += k->n;
  long *order = xmalloc(sizeof(long) * NQART);
  for (uint32_t q = 0; q < NQART; q++) order[q] = q;
  qsort(order, NQART, sizeof(long), cmp_q);
  PH(" s.setup");
  snprintf(dir, sizeof dir, "%s/user_daily", od); if (mkdir(dir, 0777) && errno != EEXIST) die(dir);
  snprintf(dir, sizeof dir, "%s/campaign_daily", od); if (mkdir(dir, 0777) && errno != EEXIST) die(dir);
  // each thread writes its user_daily file the moment the queue runs dry, while the others finish their partitions
  run_par_f(NQART, order, sess_task, write_user_task, NULL);
  PH("sess");
  {
    static const char *const sn[] = {" d.split", " d.probe", " s.gather", " s.sort", " s.clicks", " s.walk"};
    sp_print(SP_DSPLIT, SP_N, sn, PROF ? (now_s() - pt0) * 1e3 / (double)(ticks() - pk0) : 0);
  }
  write_campaign();
  PH("write");
  if (RELEASE) run_par(NT, release_task, NULL);
  PH("release");
  _exit(0);
}
