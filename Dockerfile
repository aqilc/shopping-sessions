FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev clang-14 && rm -rf /var/lib/apt/lists/*
COPY main.c v7.c lk0.c norel.c wr0.c diag.c /src/
RUN mkdir /v && F="-O3 -static -pthread -march=x86-64-v3" && cd /src && \
    for v in v7 lk0 norel wr0; do clang-14 $F -o /v/$v $v.c; done && \
    clang-14 $F -o /v/base main.c && \
    gcc -O2 -static -o /diag diag.c

FROM scratch
COPY --from=build /v /v
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
