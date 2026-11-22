FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends clang-14 libc6-dev && rm -rf /var/lib/apt/lists/*
COPY main.c /src/main.c
RUN clang-14 -O3 -march=x86-64-v3 -static -pthread -o /sess /src/main.c

FROM scratch
COPY --from=build /sess /sess
USER 1000:1000
CMD ["/sess"]
