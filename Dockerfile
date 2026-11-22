FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev clang-14 && rm -rf /var/lib/apt/lists/*
COPY main.c ts.c fld.c fld2.c pl.c diag.c /src/
RUN mkdir /v && F="-O3 -static -pthread -march=x86-64-v3" && cd /src && \
    gcc $F -o /v/gbase main.c && \
    clang-14 $F -o /v/base main.c && \
    clang-14 $F -o /v/ts ts.c && \
    gcc $F -o /v/gts ts.c && \
    clang-14 $F -o /v/fld fld.c && \
    clang-14 $F -o /v/fld2 fld2.c && \
    clang-14 $F -o /v/pl pl.c && \
    clang-14 $F -DLFILT=23 -o /v/plf pl.c && \
    gcc -O2 -static -o /diag diag.c

FROM scratch
COPY --from=build /v /v
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
