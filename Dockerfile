FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY main.c /src/main.c
COPY diag.c /src/diag.c
RUN gcc -O3 -march=x86-64-v3 -static -pthread -o /sess /src/main.c && gcc -O2 -static -o /diag /src/diag.c

FROM scratch
COPY --from=build /sess /sess
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
