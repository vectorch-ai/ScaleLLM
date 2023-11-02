FROM golang:1.21-alpine as build
WORKDIR /build
COPY ./ ./
RUN go build -o /gateway

FROM golang:1.21-alpine as runtime
EXPOSE 8080
COPY --from=build /gateway /gateway

ENTRYPOINT [ "/gateway" ]

