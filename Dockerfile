# 빌드: docker build -t ghcr.io/fashion-agent-team/mcp-face-check:dev .
FROM eclipse-temurin:21-jdk AS build
WORKDIR /src
COPY gradlew settings.gradle build.gradle ./
COPY gradle gradle
RUN ./gradlew --no-daemon dependencies > /dev/null
COPY src src
RUN ./gradlew --no-daemon bootJar -x test

FROM eclipse-temurin:21-jre
WORKDIR /app
ENV TZ=Asia/Seoul
COPY --from=build /src/build/libs/mcp-face-check-*.jar app.jar
EXPOSE 8094
USER 1000
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
