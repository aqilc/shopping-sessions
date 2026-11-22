FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY main.c dmain.c c1.c c2.c c2x.c diag.c /src/
RUN mkdir /v && F="-O3 -static -pthread -march=x86-64-v3" && \
    gcc $F -o /v/base /src/main.c && \
    gcc $F -DSTAGE=4 -o /v/s4 /src/dmain.c && \
    gcc $F -DSTAGE=5 -o /v/s5 /src/dmain.c && \
    gcc $F -DSTAGE=6 -o /v/s6 /src/dmain.c && \
    gcc $F -DFORCEPB=8 -o /v/pb8 /src/dmain.c && \
    gcc $F -o /v/c1 /src/c1.c && \
    gcc $F -o /v/c2 /src/c2.c && \
    gcc $F -o /v/c2x /src/c2x.c && \
    gcc -O2 -static -o /diag /src/diag.c

FROM scratch
COPY --from=build /v /v
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
