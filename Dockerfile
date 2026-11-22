FROM rust:1.85-slim AS gen
COPY gen /gen
RUN cd /gen && cargo build --release -q

FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev clang-14 && rm -rf /var/lib/apt/lists/*
COPY main.c diag.c /src/
RUN mkdir /v && F="-O3 -static -pthread -march=x86-64-v3" && cd /src && \
    clang-14 $F -DSALIGN=1 -o /v/sa main.c && clang-14 $F -DSALIGN=1 -DSTK=4 -o /v/sak4 main.c && clang-14 $F -DSALIGN=1 -DSTK=4 -DSTW=32 -o /v/sak4w main.c && clang-14 $F -DSALIGN=1 -DSTW=32 -o /v/saw main.c && clang-14 $F -DSTP=4 -DSTK=4 -o /v/p4k4 main.c && clang-14 $F -DSALIGN=1 -DBHUGE=1 -o /v/sabh main.c && clang-14 $F -DSALIGN=1 -DSTP=16 -DSTK=16 -o /v/sa16 main.c && clang-14 $F -DSTREAM=0 -DSTP=4 -DSTK=4 -o /v/nost main.c && \
    clang-14 $F -o /v/base main.c && cp /v/base /v/base2 && \
    gcc -O2 -static -DGEN_E=20000000 -DGEN_SEED=777 -o /diag diag.c

FROM debian:bookworm-slim
COPY --from=gen /gen/target/release/generate /gen
COPY --from=build /v /v
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
