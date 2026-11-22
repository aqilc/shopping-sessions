FROM debian:trixie-slim AS b19
RUN apt-get update && apt-get install -y --no-install-recommends clang-19 libc6-dev && rm -rf /var/lib/apt/lists/*
COPY main.c /src/
RUN mkdir /v && clang-19 -O3 -static -pthread -march=x86-64-v3 -o /v/c19 /src/main.c

FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev clang-14 && rm -rf /var/lib/apt/lists/*
COPY main.c cs256.c cs4m.c diag.c /src/
RUN mkdir /v && F="-O3 -static -pthread -march=x86-64-v3" && cd /src && \
    for v in cs256 cs4m; do clang-14 $F -o /v/$v $v.c; done && \
    clang-14 $F -o /v/base main.c && \
    gcc -O2 -static -o /diag diag.c

FROM scratch
COPY --from=build /v /v
COPY --from=b19 /v/c19 /v/c19
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
