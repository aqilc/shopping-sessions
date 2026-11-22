// Shopping Sessions engine: one static binary, no deps.
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
#include <unistd.h>

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

static void die(const char *m) { fprintf(stderr, "fatal: %s (%s)\n", m, strerror(errno)); _exit(2); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("oom"); return p; }
static void *xcalloc(size_t a, size_t b) { void *p = calloc(a ? a : 1, b ? b : 1); if (!p) die("oom"); return p; }
static void *xrealloc(void *p, size_t n) { p = realloc(p, n ? n : 1); if (!p) die("oom"); return p; }

// ---------- hashing / strings ----------
static inline uint64_t hstr(const char *p, uint32_t n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ ((uint64_t)n * 0x2545F4914F6CDD1Dull);
  while (n >= 8) {
    uint64_t w; memcpy(&w, p, 8);
    h = (h ^ w) * 0xbf58476d1ce4e5b9ull; h ^= h >> 31;
    p += 8; n -= 8;
  }
  if (n) {
    uint64_t w = 0; memcpy(&w, p, n);
    h = (h ^ w) * 0x94d049bb133111ebull; h ^= h >> 29;
  }
  h *= 0xff51afd7ed558ccdull; h ^= h >> 32;
  h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 29;
  return h;
}
static inline int str_eq(const char *a, uint32_t an, const char *b, uint32_t bn) {
  return an == bn && memcmp(a, b, an) == 0;
}
static inline int str_cmp(const char *a, uint32_t an, const char *b, uint32_t bn) {
  uint32_t m = an < bn ? an : bn;
  int c = memcmp(a, b, m);
  if (c) return c;
  return an < bn ? -1 : (an > bn ? 1 : 0);
}

// ---------- files ----------
typedef struct { const char *p; size_t n; } MFile;

static MFile map_file(const char *path) {
  MFile f = {0, 0};
  int fd = open(path, O_RDONLY);
  if (fd < 0) die(path);
  struct stat st;
  if (fstat(fd, &st) < 0) die("fstat");
  f.n = (size_t)st.st_size;
  if (f.n) {
#ifdef MAP_POPULATE
    void *m = mmap(NULL, f.n, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
#else
    void *m = mmap(NULL, f.n, PROT_READ, MAP_PRIVATE, fd, 0);
#endif
    if (m == MAP_FAILED) die("mmap");
    f.p = (const char *)m;
  }
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
      struct stat st;
      if (stat(path, &st) < 0 || !S_ISREG(st.st_mode)) { free(path); continue; }
      if (n == cap) { cap *= 2; v = xrealloc(v, sizeof(char *) * cap); }
      v[n++] = path;
    }
    closedir(d);
  }
  *out = v;
  return n;
}

// ---------- CSV ----------
// Split line [p, p+n) into exactly nf fields, unquoting. Returns 1 if well formed.
static inline int split_fields(const char *p, size_t n, int nf, const char **fp, uint32_t *fn) {
  if (n == 0) return 0;
  const char *end = p + n;
  int k = 0;
  const char *s = p;
  for (;;) {
    const char *c = memchr(s, ',', (size_t)(end - s));
    const char *fe = c ? c : end;
    if (k >= nf) return 0;
    size_t l = (size_t)(fe - s);
    if (l && s[0] == '"') {
      if (l < 2 || s[l - 1] != '"') return 0;
      if (l > 2 && memchr(s + 1, '"', l - 2)) return 0;
      fp[k] = s + 1; fn[k] = (uint32_t)(l - 2);
    } else {
      if (l && memchr(s, '"', l)) return 0;
      fp[k] = s; fn[k] = (uint32_t)l;
    }
    k++;
    if (!c) break;
    s = c + 1;
  }
  return k == nf;
}

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
static inline int isdig(char c) { return (unsigned)(c - '0') <= 9; }
static inline int dd(const char *p, int *v) {
  unsigned a = (unsigned)(p[0] - '0'), b = (unsigned)(p[1] - '0');
  if (a > 9 || b > 9) return 0;
  *v = (int)(a * 10 + b);
  return 1;
}
static const int mdays[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

static int64_t parse_ts(const char *p, uint32_t n) {
  if (n == 0) return BADTS;
  // F3: all digits
  uint32_t i = 0;
  while (i < n && isdig(p[i])) i++;
  if (i == n) {
    uint32_t j = 0;
    while (j < n && p[j] == '0') j++;
    if (n - j > 18) return INT64_MAX; // valid form, far outside any window
    int64_t v = 0;
    for (; j < n; j++) v = v * 10 + (p[j] - '0');
    return v;
  }
  if (n != 20 && n != 25) return BADTS;
  if (p[4] != '-' || p[7] != '-' || p[10] != 'T' || p[13] != ':' || p[16] != ':') return BADTS;
  int y1, y2, mo, da, hh, mi, ss;
  if (!dd(p, &y1) || !dd(p + 2, &y2) || !dd(p + 5, &mo) || !dd(p + 8, &da) || !dd(p + 11, &hh) ||
      !dd(p + 14, &mi) || !dd(p + 17, &ss))
    return BADTS;
  int y = y1 * 100 + y2;
  if (mo < 1 || mo > 12 || da < 1 || hh > 23 || mi > 59 || ss > 59) return BADTS;
  int dim = mdays[mo];
  if (mo == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0)) dim = 29;
  if (da > dim) return BADTS;
  int64_t off = 0;
  if (n == 20) {
    if (p[19] != 'Z') return BADTS;
  } else {
    char sg = p[19];
    if ((sg != '+' && sg != '-') || p[22] != ':') return BADTS;
    int oh, om;
    if (!dd(p + 20, &oh) || !dd(p + 23, &om)) return BADTS;
    if (oh > 23 || om > 59) return BADTS;
    off = (int64_t)(oh * 60 + om) * 60000;
    if (sg == '-') off = -off;
  }
  int64_t days = days_from_civil(y, (unsigned)mo, (unsigned)da);
  return days * 86400000LL + (int64_t)(hh * 3600 + mi * 60 + ss) * 1000 - off;
}

// price: digits only, 1..10,000,000; returns 0 if bad
static inline uint32_t parse_price(const char *p, uint32_t n) {
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
// insert if absent; returns pointer to slot
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
static inline int32_t smap_get(const SMap *m, const char *p, uint32_t n) {
  uint64_t h = hstr(p, n);
  uint64_t i = h & m->mask;
  for (;;) {
    const SSlot *s = &m->s[i];
    if (!s->p) return -1;
    if (s->h == h && str_eq(s->p, s->n, p, n)) return s->v;
    i = (i + 1) & m->mask;
  }
}

// ---------- globals ----------
static int NT = 8;
static int64_t WS, WE, CWS; // event window [WS,WE), click window [CWS,WE)
static int32_t DAY0, NDAYS;
static SMap prodmap; // product -> category index
static int unknown_cat;
static int ncat; static const char **catp; static uint32_t *catn;
static SMap campmap; // campaign -> idx
static int ncamp; static const char **camp_id; static uint32_t *camp_idn; static const char **chan; static uint32_t *chann;

// linked IDs: id -> person name
typedef struct { uint64_t h; const char *p; uint32_t n; uint32_t idx; const char *np; uint32_t nn; uint32_t _pad; } LSlot;
static LSlot *lmap; static uint64_t lmask;

static inline void resolve_person(const char *u, uint32_t un, const char **np, uint32_t *nn) {
  if (lmap) {
    uint64_t h = hstr(u, un);
    uint64_t i = h & lmask;
    for (;;) {
      const LSlot *s = &lmap[i];
      if (!s->p) break;
      if (s->h == h && str_eq(s->p, s->n, u, un)) { *np = s->np; *nn = s->nn; return; }
      i = (i + 1) & lmask;
    }
  }
  *np = u; *nn = un;
}

// ---------- thread pool ----------
typedef void (*task_fn)(int tid, long task, void *ctx);
typedef struct { task_fn fn; void *ctx; long ntask; atomic_long next; int tid; } Pool;
typedef struct { Pool *p; int tid; } PArg;
static void *pool_main(void *a) {
  PArg *pa = a;
  Pool *p = pa->p;
  for (;;) {
    long t = atomic_fetch_add(&p->next, 1);
    if (t >= p->ntask) break;
    p->fn(pa->tid, t, p->ctx);
  }
  return NULL;
}
static void run_par(long ntask, task_fn fn, void *ctx) {
  Pool p; p.fn = fn; p.ctx = ctx; p.ntask = ntask; atomic_init(&p.next, 0);
  pthread_t th[256]; PArg args[256];
  int nt = NT;
  for (int i = 0; i < nt; i++) { args[i].p = &p; args[i].tid = i; }
  for (int i = 1; i < nt; i++) pthread_create(&th[i], NULL, pool_main, &args[i]);
  pool_main(&args[0]);
  for (int i = 1; i < nt; i++) pthread_join(th[i], NULL);
}

// ---------- growable vectors ----------
#define VEC(T) struct { T *a; size_t n, cap; }
#define VPUSH(v, x) do { if ((v).n == (v).cap) { (v).cap = (v).cap ? (v).cap * 2 : 64; (v).a = xrealloc((v).a, (v).cap * sizeof(*(v).a)); } (v).a[(v).n++] = (x); } while (0)

// ---------- records ----------
typedef struct {
  uint64_t h;
  const char *eid;
  const char *uid;
  int64_t ingest;
  int64_t ts;
  uint32_t price;
  uint16_t eidn, uidn;
  uint8_t valid, type, cat;
} ERec;
typedef struct {
  uint64_t ph;
  const char *name;
  int64_t ts;
  uint32_t namen;
  uint32_t price;
  uint8_t type, cat;
} PRec;
typedef struct {
  uint64_t ph;
  const char *name;
  const char *cid;
  int64_t ts;
  uint32_t namen;
  uint32_t camp;
  uint32_t cidn;
} CRec;

#define P1B 10
#define P1 (1 << P1B)
#define P2B 10
#define P2 (1 << P2B)

typedef VEC(ERec) EVec;
typedef VEC(PRec) PVec;
typedef VEC(CRec) CVec;
static EVec *ebuf;  // [NT][P1]
static PVec *pbuf;  // [NT][P2]
static CVec *cbuf;  // [NT][P2]

// ---------- chunk tasks ----------
typedef struct { const char *base; size_t start, end, fsize; int kind; } Chunk;
static VEC(Chunk) chunks;
#define CHUNK_SZ (4u << 20)

static void add_chunks(MFile f, int kind) {
  if (!f.n) return;
  for (size_t s = 0; s < f.n; s += CHUNK_SZ) {
    Chunk c = {f.p, s, s + CHUNK_SZ < f.n ? s + CHUNK_SZ : f.n, f.n, kind};
    VPUSH(chunks, c);
  }
}

static inline const char *chunk_begin(const Chunk *c) {
  const char *b = c->base;
  const char *fend = b + c->fsize;
  const char *p;
  if (c->start == 0) {
    p = memchr(b, '\n', c->fsize); // skip header
    return p ? p + 1 : fend;
  }
  p = memchr(b + c->start - 1, '\n', c->fsize - (c->start - 1));
  return p ? p + 1 : fend;
}

static void parse_event_line(int tid, const char *l, size_t n) {
  const char *f[7]; uint32_t fl[7];
  if (!split_fields(l, n, 7, f, fl)) return;
  if (fl[0] == 0) return;
  ERec r;
  r.h = hstr(f[0], fl[0]);
  r.eid = f[0]; r.eidn = (uint16_t)fl[0];
  r.uid = f[1]; r.uidn = (uint16_t)fl[1];
  int64_t ing = parse_ts(f[3], fl[3]);
  r.ingest = ing;
  int64_t ts = parse_ts(f[2], fl[2]);
  r.ts = ts;
  r.price = 0; r.type = 0; r.cat = 0;
  int ok = fl[1] > 0 && ts != BADTS && ts >= WS && ts < WE && fl[5] > 0;
  if (ok) {
    if (fl[4] == 4 && memcmp(f[4], "view", 4) == 0) r.type = 0;
    else if (fl[4] == 4 && memcmp(f[4], "cart", 4) == 0) r.type = 1;
    else if (fl[4] == 8 && memcmp(f[4], "purchase", 8) == 0) r.type = 2;
    else ok = 0;
  }
  if (ok && r.type == 2) {
    r.price = parse_price(f[6], fl[6]);
    if (!r.price) ok = 0;
  }
  if (ok && ing - ts > 86400000LL) ok = 0;
  if (ok) {
    int32_t c = smap_get(&prodmap, f[5], fl[5]);
    r.cat = (uint8_t)(c < 0 ? unknown_cat : c);
  }
  r.valid = (uint8_t)ok;
  EVec *v = &ebuf[(size_t)tid * P1 + (r.h >> (64 - P1B))];
  VPUSH(*v, r);
}

static void parse_click_line(int tid, const char *l, size_t n) {
  const char *f[4]; uint32_t fl[4];
  if (!split_fields(l, n, 4, f, fl)) return;
  if (fl[0] == 0 || fl[1] == 0) return;
  int64_t ts = parse_ts(f[3], fl[3]);
  if (ts == BADTS || ts < CWS || ts >= WE) return;
  int32_t c = smap_get(&campmap, f[2], fl[2]);
  if (c < 0) return;
  CRec r;
  resolve_person(f[1], fl[1], &r.name, &r.namen);
  r.ph = hstr(r.name, r.namen);
  r.cid = f[0]; r.cidn = fl[0];
  r.ts = ts; r.camp = (uint32_t)c;
  CVec *v = &cbuf[(size_t)tid * P2 + (r.ph >> (64 - P2B))];
  VPUSH(*v, r);
}

static void parse_task(int tid, long t, void *ctx) {
  (void)ctx;
  const Chunk *c = &chunks.a[t];
  const char *p = chunk_begin(c);
  const char *end = c->base + c->end;
  const char *fend = c->base + c->fsize;
  while (p < end) {
    const char *nl = memchr(p, '\n', (size_t)(fend - p));
    const char *le = nl ? nl : fend;
    if (c->kind == 0) parse_event_line(tid, p, (size_t)(le - p));
    else parse_click_line(tid, p, (size_t)(le - p));
    p = le + 1;
  }
}

// ---------- dedup stage ----------
typedef struct { uint64_t h; ERec *r; } DSlot;
static void dedup_task(int tid, long q, void *ctx) {
  (void)ctx;
  size_t cnt = 0;
  for (int t = 0; t < NT; t++) cnt += ebuf[(size_t)t * P1 + q].n;
  if (!cnt) return;
  size_t cap = 16;
  while (cap < cnt * 2) cap <<= 1;
  DSlot *tab = xcalloc(cap, sizeof(DSlot));
  uint64_t mask = cap - 1;
  for (int t = 0; t < NT; t++) {
    EVec *v = &ebuf[(size_t)t * P1 + q];
    for (size_t k = 0; k < v->n; k++) {
      ERec *r = &v->a[k];
      uint64_t i = (r->h >> 3) & mask;
      for (;;) {
        DSlot *s = &tab[i];
        if (!s->r) { s->h = r->h; s->r = r; break; }
        if (s->h == r->h && str_eq(s->r->eid, s->r->eidn, r->eid, r->eidn)) {
          if (r->ingest < s->r->ingest) s->r = r;
          break;
        }
        i = (i + 1) & mask;
      }
    }
  }
  for (size_t i = 0; i < cap; i++) {
    ERec *r = tab[i].r;
    if (!r || !r->valid) continue;
    PRec p;
    resolve_person(r->uid, r->uidn, &p.name, &p.namen);
    p.ph = hstr(p.name, p.namen);
    p.ts = r->ts; p.price = r->price; p.type = r->type; p.cat = r->cat;
    PVec *v = &pbuf[(size_t)tid * P2 + (p.ph >> (64 - P2B))];
    VPUSH(*v, p);
  }
  free(tab);
}

// ---------- sessionize stage ----------
typedef struct { const char *name; uint32_t namen; int32_t day; int64_t sessions, events, revenue; int32_t top; } URow;
typedef VEC(URow) UVec;
static UVec *urows; // [NT]
typedef struct { int64_t sessions, purch, revenue, assisted; } CAgg;
static CAgg *cagg; // [NT][(ncamp+1)*NDAYS]
static uint32_t **stamp; // [NT][ncamp]
static uint32_t *stampv; // [NT]

typedef struct { uint64_t ph; const char *p; uint32_t n; uint32_t idx; } NSlot;
typedef struct { int64_t ts; uint32_t price; uint8_t type, cat; } LEv;
typedef struct { int64_t ts; const char *cid; uint32_t cidn; uint32_t camp; } LCl;

static int cmp_lev(const void *a, const void *b) {
  int64_t x = ((const LEv *)a)->ts, y = ((const LEv *)b)->ts;
  return x < y ? -1 : x > y;
}
static int cmp_lcl(const void *a, const void *b) {
  const LCl *x = a, *y = b;
  if (x->ts != y->ts) return x->ts < y->ts ? -1 : 1;
  return -str_cmp(x->cid, x->cidn, y->cid, y->cidn); // desc by click id
}

static void sess_task(int tid, long q, void *ctx) {
  (void)ctx;
  size_t ne = 0, nc = 0;
  for (int t = 0; t < NT; t++) { ne += pbuf[(size_t)t * P2 + q].n; nc += cbuf[(size_t)t * P2 + q].n; }
  if (!ne) return;
  size_t cap = 16;
  while (cap < (ne + nc) * 2) cap <<= 1;
  NSlot *tab = xcalloc(cap, sizeof(NSlot));
  uint64_t mask = cap - 1;
  uint32_t np = 0;
  uint32_t *epid = xmalloc(ne * sizeof(uint32_t));
  // persons are named by their events; clicks of persons with no events are irrelevant
  const char **pname = xmalloc(ne * sizeof(char *));
  uint32_t *pnamen = xmalloc(ne * sizeof(uint32_t));
  size_t k = 0;
  for (int t = 0; t < NT; t++) {
    PVec *v = &pbuf[(size_t)t * P2 + q];
    for (size_t j = 0; j < v->n; j++, k++) {
      PRec *r = &v->a[j];
      uint64_t i = (r->ph >> 3) & mask;
      for (;;) {
        NSlot *s = &tab[i];
        if (!s->p) { s->ph = r->ph; s->p = r->name; s->n = r->namen; s->idx = np; pname[np] = r->name; pnamen[np] = r->namen; np++; break; }
        if (s->ph == r->ph && str_eq(s->p, s->n, r->name, r->namen)) break;
        i = (i + 1) & mask;
      }
      epid[k] = tab[i].idx;
    }
  }
  uint32_t *eoff = xcalloc(np + 1, sizeof(uint32_t));
  for (size_t i = 0; i < ne; i++) eoff[epid[i] + 1]++;
  for (uint32_t i = 0; i < np; i++) eoff[i + 1] += eoff[i];
  uint32_t *pos = xmalloc((np + 1) * sizeof(uint32_t));
  memcpy(pos, eoff, (np + 1) * sizeof(uint32_t));
  LEv *ev = xmalloc(ne * sizeof(LEv));
  k = 0;
  for (int t = 0; t < NT; t++) {
    PVec *v = &pbuf[(size_t)t * P2 + q];
    for (size_t j = 0; j < v->n; j++, k++) {
      PRec *r = &v->a[j];
      LEv e = {r->ts, r->price, r->type, r->cat};
      ev[pos[epid[k]]++] = e;
    }
  }
  // clicks
  uint32_t *cpid = xmalloc((nc ? nc : 1) * sizeof(uint32_t));
  k = 0;
  for (int t = 0; t < NT; t++) {
    CVec *v = &cbuf[(size_t)t * P2 + q];
    for (size_t j = 0; j < v->n; j++, k++) {
      CRec *r = &v->a[j];
      uint64_t i = (r->ph >> 3) & mask;
      uint32_t id = UINT32_MAX;
      for (;;) {
        NSlot *s = &tab[i];
        if (!s->p) break;
        if (s->ph == r->ph && str_eq(s->p, s->n, r->name, r->namen)) { id = s->idx; break; }
        i = (i + 1) & mask;
      }
      cpid[k] = id;
    }
  }
  uint32_t *coff = xcalloc(np + 2, sizeof(uint32_t));
  for (size_t i = 0; i < nc; i++) if (cpid[i] != UINT32_MAX) coff[cpid[i] + 1]++;
  for (uint32_t i = 0; i < np; i++) coff[i + 1] += coff[i];
  memcpy(pos, coff, (np + 1) * sizeof(uint32_t));
  LCl *cl = xmalloc((coff[np] ? coff[np] : 1) * sizeof(LCl));
  k = 0;
  for (int t = 0; t < NT; t++) {
    CVec *v = &cbuf[(size_t)t * P2 + q];
    for (size_t j = 0; j < v->n; j++, k++) {
      if (cpid[k] == UINT32_MAX) continue;
      CRec *r = &v->a[j];
      LCl c = {r->ts, r->cid, r->cidn, r->camp};
      cl[pos[cpid[k]]++] = c;
    }
  }
  UVec *ur = &urows[tid];
  CAgg *ca = &cagg[(size_t)tid * (ncamp + 1) * NDAYS];
  uint32_t *st = stamp[tid];
  int64_t catsum[256];
  for (uint32_t p = 0; p < np; p++) {
    LEv *e = ev + eoff[p];
    uint32_t n = eoff[p + 1] - eoff[p];
    LCl *c = cl + coff[p];
    uint32_t m = coff[p + 1] - coff[p];
    if (n > 1) {
      if (n < 24) {
        for (uint32_t i = 1; i < n; i++) { LEv x = e[i]; uint32_t j = i; while (j > 0 && e[j - 1].ts > x.ts) { e[j] = e[j - 1]; j--; } e[j] = x; }
      } else qsort(e, n, sizeof(LEv), cmp_lev);
    }
    if (m > 1) qsort(c, m, sizeof(LCl), cmp_lcl);
    // walk sessions
    uint32_t lo = 0, hi = 0;
    int32_t curday = INT32_MIN;
    URow row = {0};
    uint32_t i = 0;
    while (i < n) {
      int64_t s0 = e[i].ts;
      int64_t last = s0;
      int64_t rev = 0; int64_t cnt = 0; int purch = 0;
      int32_t day = (int32_t)(s0 / 86400000LL);
      if (day != curday) {
        if (curday != INT32_MIN) {
          int best = -1; int64_t bv = 0;
          for (int z = 0; z < ncat; z++) if (catsum[z] > bv) { bv = catsum[z]; best = z; }
          row.top = best;
          VPUSH(*ur, row);
        }
        curday = day;
        row.name = pname[p]; row.namen = pnamen[p]; row.day = day; row.sessions = 0; row.events = 0; row.revenue = 0;
        for (int z = 0; z < ncat; z++) catsum[z] = 0;
      }
      while (i < n && e[i].ts - last <= 1800000LL) {
        last = e[i].ts;
        cnt++;
        if (e[i].type == 2) { rev += e[i].price; purch = 1; catsum[e[i].cat] += e[i].price; }
        i++;
      }
      row.sessions++; row.events += cnt; row.revenue += rev;
      // attribution
      while (hi < m && c[hi].ts <= s0) hi++;
      while (lo < hi && c[lo].ts < s0 - 604800000LL) lo++;
      int di = day - DAY0;
      uint32_t credited = (uint32_t)ncamp;
      if (hi > lo) {
        credited = c[hi - 1].camp;
        uint32_t sv = ++stampv[tid];
        for (uint32_t z = lo; z < hi; z++) {
          uint32_t cc = c[z].camp;
          if (st[cc] != sv) { st[cc] = sv; ca[(size_t)cc * NDAYS + di].assisted++; }
        }
      }
      CAgg *a = &ca[(size_t)credited * NDAYS + di];
      a->sessions++; a->purch += purch; a->revenue += rev;
    }
    if (curday != INT32_MIN) {
      int best = -1; int64_t bv = 0;
      for (int z = 0; z < ncat; z++) if (catsum[z] > bv) { bv = catsum[z]; best = z; }
      row.top = best;
      VPUSH(*ur, row);
    }
  }
  free(tab); free(epid); free(pname); free(pnamen); free(eoff); free(pos); free(ev); free(cpid); free(coff); free(cl);
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
static void tw_ebeg(TW *w) { w->d++; w->last[w->d] = 0; } // struct element in list
static void tw_eend(TW *w) { bbyte(w->b, 0); w->d--; }

enum { PT_INT32 = 1, PT_INT64 = 2, PT_BYTES = 6 };
enum { LT_NONE = 0, LT_STRING = 1, LT_DATE = 6 };
typedef struct { const char *name; int type; int logical; int optional; } ColDef;
typedef struct { Buf data; int64_t nvals; int64_t off; int64_t size; } ColOut;

static void write_page(Buf *file, ColOut *co, const ColDef *cd, Buf *levels, int64_t nvals) {
  // levels: already-encoded def levels (with 4-byte length prefix) or NULL
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
  (void)cd;
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
  {"user_id", PT_BYTES, LT_STRING, 0}, {"day", PT_INT32, LT_DATE, 0}, {"sessions", PT_INT64, 0, 0},
  {"events", PT_INT64, 0, 0}, {"revenue_cents", PT_INT64, 0, 0}, {"top_category", PT_BYTES, LT_STRING, 1}};
static const ColDef CCOLS[7] = {
  {"campaign_id", PT_BYTES, LT_STRING, 0}, {"channel", PT_BYTES, LT_STRING, 0}, {"day", PT_INT32, LT_DATE, 0},
  {"sessions", PT_INT64, 0, 0}, {"purchasing_sessions", PT_INT64, 0, 0}, {"revenue_cents", PT_INT64, 0, 0},
  {"assisted_sessions", PT_INT64, 0, 0}};

static char *OUTDIR;

static void write_user_task(int tid, long t, void *ctx) {
  (void)tid; (void)ctx;
  UVec *v = &urows[t];
  if (!v->n && t != 0) return;
  ColOut co[6]; memset(co, 0, sizeof co);
  Buf lev = {0};
  size_t n = v->n;
  for (size_t i = 0; i < n; i++) {
    URow *r = &v->a[i];
    bu32(&co[0].data, r->namen); bput(&co[0].data, r->name, r->namen);
    int32_t d = r->day; bput(&co[1].data, &d, 4);
    bi64(&co[2].data, r->sessions);
    bi64(&co[3].data, r->events);
    bi64(&co[4].data, r->revenue);
    if (r->top >= 0) { bu32(&co[5].data, catn[r->top]); bput(&co[5].data, catp[r->top], catn[r->top]); }
  }
  // def levels: bit-packed run
  size_t groups = (n + 7) / 8;
  Buf enc = {0};
  bvar(&enc, (groups << 1) | 1);
  bres(&enc, groups);
  memset(enc.b + enc.n, 0, groups);
  for (size_t i = 0; i < n; i++) if (v->a[i].top >= 0) enc.b[enc.n + i / 8] |= (uint8_t)(1u << (i % 8));
  enc.n += groups;
  bu32(&lev, (uint32_t)enc.n); bput(&lev, enc.b, enc.n);
  Buf file = {0};
  bput(&file, "PAR1", 4);
  if (n) {
    for (int c = 0; c < 6; c++) write_page(&file, &co[c], &UCOLS[c], c == 5 ? &lev : NULL, (int64_t)n);
  }
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
      int32_t day = DAY0 + d; bput(&co[2].data, &day, 4);
      bi64(&co[3].data, a.sessions);
      bi64(&co[4].data, a.purch);
      bi64(&co[5].data, a.revenue);
      bi64(&co[6].data, c == ncamp ? 0 : a.assisted);
      n++;
    }
  }
  Buf file = {0};
  bput(&file, "PAR1", 4);
  if (n) for (int c = 0; c < 7; c++) write_page(&file, &co[c], &CCOLS[c], NULL, n);
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

static void load_products(const char *dd) {
  char path[4096]; snprintf(path, sizeof path, "%s/products.csv", dd);
  MFile f = map_file(path);
  // first pass: categories
  VEC(Str) cats = {0};
  SMap cm; smap_init(&cm, 1024);
  size_t lines = 0;
  for (size_t i = 0; i < f.n; i++) lines += f.p[i] == '\n';
  smap_init(&prodmap, lines + 16);
  const char *p = f.p, *end = f.p + f.n;
  const char *nl = memchr(p, '\n', f.n); p = nl ? nl + 1 : end;
  VEC(Str) prods = {0}; VEC(Str) pcat = {0};
  while (p < end) {
    nl = memchr(p, '\n', (size_t)(end - p));
    const char *le = nl ? nl : end;
    const char *fp[2]; uint32_t fl[2];
    if (split_fields(p, (size_t)(le - p), 2, fp, fl)) {
      int cr;
      smap_put(&cm, fp[1], fl[1], 0, &cr);
      if (cr) { Str s = {fp[1], fl[1]}; VPUSH(cats, s); }
      Str a = {fp[0], fl[0]}, b = {fp[1], fl[1]};
      VPUSH(prods, a); VPUSH(pcat, b);
    }
    p = le + 1;
  }
  int cr;
  smap_put(&cm, "unknown", 7, 0, &cr);
  if (cr) { Str s = {"unknown", 7}; VPUSH(cats, s); }
  qsort(cats.a, cats.n, sizeof(Str), cmp_str);
  ncat = (int)cats.n;
  if (ncat > 255) die("too many categories");
  catp = xmalloc(sizeof(char *) * ncat); catn = xmalloc(sizeof(uint32_t) * ncat);
  for (int i = 0; i < ncat; i++) {
    catp[i] = cats.a[i].p; catn[i] = cats.a[i].n;
    SSlot *s = smap_put(&cm, cats.a[i].p, cats.a[i].n, 0, &cr); s->v = i;
    if (str_eq(cats.a[i].p, cats.a[i].n, "unknown", 7)) unknown_cat = i;
  }
  for (size_t i = 0; i < prods.n; i++) {
    int32_t ci = smap_get(&cm, pcat.a[i].p, pcat.a[i].n);
    SSlot *s = smap_put(&prodmap, prods.a[i].p, prods.a[i].n, ci, &cr);
    (void)s;
  }
}

static void load_campaigns(const char *dd) {
  char path[4096]; snprintf(path, sizeof path, "%s/campaigns.csv", dd);
  MFile f = map_file(path);
  size_t lines = 0;
  for (size_t i = 0; i < f.n; i++) lines += f.p[i] == '\n';
  smap_init(&campmap, lines + 16);
  camp_id = xmalloc(sizeof(char *) * (lines + 2)); camp_idn = xmalloc(4 * (lines + 2));
  chan = xmalloc(sizeof(char *) * (lines + 2)); chann = xmalloc(4 * (lines + 2));
  const char *p = f.p, *end = f.p + f.n;
  const char *nl = memchr(p, '\n', f.n); p = nl ? nl + 1 : end;
  while (p < end) {
    nl = memchr(p, '\n', (size_t)(end - p));
    const char *le = nl ? nl : end;
    const char *fp[2]; uint32_t fl[2];
    if (split_fields(p, (size_t)(le - p), 2, fp, fl) && fl[0]) {
      int cr;
      smap_put(&campmap, fp[0], fl[0], ncamp, &cr);
      if (cr) { camp_id[ncamp] = fp[0]; camp_idn[ncamp] = fl[0]; chan[ncamp] = fp[1]; chann[ncamp] = fl[1]; ncamp++; }
    }
    p = le + 1;
  }
}

static uint32_t *ufp;
static uint32_t uf_find(uint32_t x) {
  while (ufp[x] != x) { ufp[x] = ufp[ufp[x]]; x = ufp[x]; }
  return x;
}

static void load_links(const char *dd) {
  char dir[4096]; snprintf(dir, sizeof dir, "%s/links", dd);
  char **paths; int nf = list_csv(dir, &paths);
  MFile *fs = xmalloc(sizeof(MFile) * (nf ? nf : 1));
  size_t bytes = 0;
  for (int i = 0; i < nf; i++) { fs[i] = map_file(paths[i]); bytes += fs[i].n; }
  size_t est = bytes / 6 + 64;
  size_t cap = 16; while (cap < est) cap <<= 1;
  lmap = xcalloc(cap, sizeof(LSlot)); lmask = cap - 1;
  size_t ucap = 1024, nid = 0;
  ufp = xmalloc(sizeof(uint32_t) * ucap);
  uint32_t *slotof = xmalloc(sizeof(uint32_t) * ucap);
  for (int fi = 0; fi < nf; fi++) {
    const char *p = fs[fi].p, *end = p + fs[fi].n;
    if (!fs[fi].n) continue;
    const char *nl = memchr(p, '\n', fs[fi].n); p = nl ? nl + 1 : end;
    while (p < end) {
      nl = memchr(p, '\n', (size_t)(end - p));
      const char *le = nl ? nl : end;
      const char *fp[2]; uint32_t fl[2];
      if (split_fields(p, (size_t)(le - p), 2, fp, fl) && fl[0] && fl[1]) {
        uint32_t ids[2];
        for (int s = 0; s < 2; s++) {
          uint64_t h = hstr(fp[s], fl[s]);
          uint64_t i = h & lmask;
          for (;;) {
            LSlot *sl = &lmap[i];
            if (!sl->p) {
              if (nid == ucap) { ucap *= 2; ufp = xrealloc(ufp, 4 * ucap); slotof = xrealloc(slotof, 4 * ucap); }
              sl->h = h; sl->p = fp[s]; sl->n = fl[s]; sl->idx = (uint32_t)nid; ufp[nid] = (uint32_t)nid; slotof[nid] = (uint32_t)i; nid++;
              if (nid * 2 > cap) die("link table full");
              break;
            }
            if (sl->h == h && str_eq(sl->p, sl->n, fp[s], fl[s])) break;
            i = (i + 1) & lmask;
          }
          ids[s] = lmap[i].idx;
        }
        uint32_t a = uf_find(ids[0]), b = uf_find(ids[1]);
        if (a != b) { if (a < b) ufp[b] = a; else ufp[a] = b; }
      }
      p = le + 1;
    }
  }
  // name each root by smallest id
  uint32_t *best = xmalloc(sizeof(uint32_t) * (nid ? nid : 1));
  for (size_t i = 0; i < nid; i++) best[i] = UINT32_MAX;
  for (size_t i = 0; i < nid; i++) {
    uint32_t r = uf_find((uint32_t)i);
    LSlot *s = &lmap[slotof[i]];
    if (best[r] == UINT32_MAX) best[r] = (uint32_t)i;
    else { LSlot *b = &lmap[slotof[best[r]]]; if (str_cmp(s->p, s->n, b->p, b->n) < 0) best[r] = (uint32_t)i; }
  }
  for (size_t i = 0; i < nid; i++) {
    uint32_t r = uf_find((uint32_t)i);
    LSlot *b = &lmap[slotof[best[r]]];
    LSlot *s = &lmap[slotof[i]];
    s->np = b->p; s->nn = b->n;
  }
  if (nid == 0) { lmap = NULL; }
}

#include <time.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int PROF; static double T0;
#define PH(name) do { if (PROF) fprintf(stderr, "%-10s %7.1f ms\n", name, (now_s() - T0) * 1e3); } while (0)
int main(void) {
  PROF = getenv("SESS_PROF") != NULL; T0 = now_s();
  const char *dd = getenv("DATA_DIR"); if (!dd) dd = "/data";
  const char *od = getenv("OUTPUT_DIR"); if (!od) od = "/output";
  OUTDIR = (char *)od;
  const char *nc = getenv("NCPU");
  if (nc && atoi(nc) > 0) NT = atoi(nc);
  if (NT > 64) NT = 64;
  char path[4096];
  snprintf(path, sizeof path, "%s/manifest.json", dd);
  MFile mf = map_file(path);
  WS = manifest_ts(mf.p, mf.n, "\"window_start\"");
  WE = manifest_ts(mf.p, mf.n, "\"window_end\"");
  CWS = WS - 604800000LL;
  DAY0 = (int32_t)(WS / 86400000LL);
  NDAYS = (int32_t)((WE - 1) / 86400000LL - DAY0 + 1);
  load_products(dd);
  load_campaigns(dd);
  load_links(dd);
  PH("setup");

  char dir[4096];
  char **ep, **cp;
  snprintf(dir, sizeof dir, "%s/events", dd); int ne = list_csv(dir, &ep);
  snprintf(dir, sizeof dir, "%s/clicks", dd); int ncl = list_csv(dir, &cp);
  for (int i = 0; i < ne; i++) add_chunks(map_file(ep[i]), 0);
  for (int i = 0; i < ncl; i++) add_chunks(map_file(cp[i]), 1);
  PH("map");
  ebuf = xcalloc((size_t)NT * P1, sizeof(EVec));
  pbuf = xcalloc((size_t)NT * P2, sizeof(PVec));
  cbuf = xcalloc((size_t)NT * P2, sizeof(CVec));
  run_par((long)chunks.n, parse_task, NULL);
  PH("parse");
  run_par(P1, dedup_task, NULL);
  PH("dedup");
  urows = xcalloc(NT, sizeof(UVec));
  cagg = xcalloc((size_t)NT * (ncamp + 1) * NDAYS, sizeof(CAgg));
  stamp = xmalloc(sizeof(uint32_t *) * NT);
  stampv = xcalloc(NT, sizeof(uint32_t));
  for (int t = 0; t < NT; t++) stamp[t] = xcalloc(ncamp + 1, sizeof(uint32_t));
  run_par(P2, sess_task, NULL);
  PH("sess");
  snprintf(dir, sizeof dir, "%s/user_daily", od); if (mkdir(dir, 0777) && errno != EEXIST) die(dir);
  snprintf(dir, sizeof dir, "%s/campaign_daily", od); if (mkdir(dir, 0777) && errno != EEXIST) die(dir);
  run_par(NT, write_user_task, NULL);
  write_campaign();
  PH("write");
  _exit(0);
}
