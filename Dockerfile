FROM rust:1.85-slim AS gen
COPY gen /gen
RUN cd /gen && cargo build --release -q && /gen/target/release/generate --seed 4242 --events 400000 --clicks 100000 --output /train --threads 4

FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev clang-14 llvm-14 libclang-rt-14-dev && rm -rf /var/lib/apt/lists/*
COPY --from=gen /train /train
COPY main.c diag.c /src/
RUN mkdir /v /p /o && F="-O3 -static -pthread -march=x86-64-v3" && cd /src && \
    gcc $F -o /v/base main.c && \
    gcc $F -mtune=cascadelake -o /v/tune main.c && \
    clang-14 $F -o /v/clang main.c && \
    gcc $F -fprofile-generate -fprofile-update=atomic -fprofile-dir=/p -c main.c -o m.o && gcc $F -fprofile-generate -Wl,-u,__gcov_dump m.o -o /tmp/gi && \
    for i in 1 2; do rm -rf /o/*; DATA_DIR=/train OUTPUT_DIR=/o NCPU=8 /tmp/gi; done && \
    rm m.o && gcc $F -fprofile-use -fprofile-dir=/p -Werror=missing-profile -c main.c -o m.o && gcc $F m.o -o /v/pgo && ls -R /p | head -5 && \
    clang-14 $F -fprofile-instr-generate -Wl,-u,__llvm_profile_write_file -o /tmp/ci main.c && \
    for i in 1 2; do rm -rf /o/*; LLVM_PROFILE_FILE=/p/c$i.profraw DATA_DIR=/train OUTPUT_DIR=/o NCPU=8 /tmp/ci; done && \
    llvm-profdata-14 merge -o /p/c.profdata /p/c1.profraw /p/c2.profraw && \
    clang-14 $F -fprofile-instr-use=/p/c.profdata -o /v/cpgo main.c && \
    gcc -O2 -static -o /diag diag.c

FROM scratch
COPY --from=build /v /v
COPY --from=build /diag /diag
USER 1000:1000
CMD ["/diag"]
