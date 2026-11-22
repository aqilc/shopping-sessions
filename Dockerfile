FROM rust:1.85-slim AS gen
COPY gen /gen
RUN cd /gen && cargo build --release -q

FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev clang-14 && rm -rf /var/lib/apt/lists/*
COPY main.c r1.c r2.c r3.c diag.c /src/
RUN mkdir /v && F="-O3 -static -pthread -march=x86-64-v3" && cd /src && \
    for v in r1 r2 r3; do clang-14 $F -o /v/$v $v.c; done && \
    clang-14 $F -DLFILTER=1 -o /v/r3lf r3.c && clang-14 $F -DPREFAULT=1 -o /v/r3pf r3.c && \
    clang-14 $F -o /v/base main.c && cp /v/base /v/base2 && \
    gcc -O2 -static -DGEN_E=20000000 -DGEN_SEED=777 -o /diag diag.c

FROM debian:bookworm-slim
COPY --from=gen /gen/target/release/generate /gen
COPY --from=build /v /v
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
