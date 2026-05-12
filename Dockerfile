# map-service-agent — Docker 빌드 스켈레톤
#
# 본 파일은 의도적으로 비어 있다. 실제 코드 작성 단계에서 아래 항목을 채운다.
#
# 작성 예정:
#   1. base 이미지       — python:3.12-slim
#   2. 멀티스테이지       — builder(wheel 빌드) → runtime(슬림 이미지)
#   3. 시스템 의존성      — 필요 시 build-essential, libpq-dev 등
#   4. 의존성 설치        — pip install -r requirements.txt
#   5. 비-root 사용자     — useradd app && USER app
#   6. WORKDIR /app + 소스 COPY
#   7. EXPOSE 8000
#   8. HEALTHCHECK        — GET /health
#   9. ENTRYPOINT         — uvicorn main:app --host 0.0.0.0 --port 8000
