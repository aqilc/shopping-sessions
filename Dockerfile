FROM alpine:3.20 AS build
RUN apk add --no-cache gcc musl-dev
COPY main.c /src/main.c
RUN gcc -O3 -march=x86-64-v3 -static -pthread -o /sess /src/main.c

FROM scratch
COPY --from=build /sess /sess
USER 1000:1000
CMD ["/sess"]
