FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY main.c wp.c dmain.c diag.c /src/
RUN mkdir /v && F="-O3 -static -pthread" && \
    gcc $F -march=x86-64-v3 -o /v/base /src/main.c && \
    gcc $F -march=x86-64-v3 -o /v/wp /src/wp.c && \
    gcc $F -march=x86-64-v4 -o /v/v4 /src/dmain.c && \
    gcc $F -march=x86-64-v3 -DSTAGE=1 -o /v/s1 /src/dmain.c && \
    gcc $F -march=x86-64-v3 -DSTAGE=2 -o /v/s2 /src/dmain.c && \
    gcc $F -march=x86-64-v3 -DSTAGE=3 -o /v/s3 /src/dmain.c && \
    gcc -O2 -static -o /diag /src/diag.c

FROM scratch
COPY --from=build /v /v
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
