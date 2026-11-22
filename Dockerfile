FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY main.c c2x.c st1.c st2.c st3.c st4.c diag.c /src/
RUN mkdir /v && F="-O3 -static -pthread -march=x86-64-v3" && \
    gcc $F -o /v/base /src/main.c && \
    gcc $F -o /v/c2x /src/c2x.c && \
    gcc $F -o /v/st1 /src/st1.c && \
    gcc $F -o /v/st2 /src/st2.c && \
    gcc $F -o /v/st3 /src/st3.c && \
    gcc $F -o /v/st4 /src/st4.c && \
    gcc -O2 -static -o /diag /src/diag.c

FROM scratch
COPY --from=build /v /v
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
