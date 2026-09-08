# syntax=docker/dockerfile:1.7
# map-service-agent — FastAPI + LangGraph 추천 엔진
#
# 멀티 스테이지 이미지 구성:
#   1) builder stage  : 컴파일 도구를 설치하고 requirements.txt 를 휠로 빌드해 /wheels 에 모은다.
#   2) runtime stage  : 슬림 베이스 위에 builder 가 만든 휠만 사용해 의존성을 설치한다.
#                       빌드 도구가 빠져 이미지 크기가 작고, 런타임에는 네트워크 의존성이 없다.
# 비루트 사용자(app, uid=10001) 로 컨테이너를 띄우며, uvicorn 으로 app.main:app 을 노출한다.

# ARG PYTHON_VERSION
#   - builder/runtime 양쪽 FROM 에서 공통으로 쓰는 Python 베이스 태그.
ARG PYTHON_VERSION=3.12

# ─── builder stage ───────────────────────────────────────────────
# 빌드 전용 스테이지. 휠을 만들기 위해 build-essential 을 설치하지만
# 결과적으로 runtime 이미지에는 포함되지 않는다.
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# pip wheel 로 requirements.txt 의 모든 패키지를 사전 컴파일된 휠로 생성한다.
RUN pip wheel --wheel-dir=/wheels -r requirements.txt

# ─── runtime stage ───────────────────────────────────────────────
# 실제 서비스가 동작하는 최소 런타임. /wheels 를 인덱스로 삼아 오프라인 설치한다.
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# 권한 최소화를 위해 비루트 사용자 app(uid 10001) 생성.
RUN useradd -m -u 10001 app
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
# --no-index --find-links=/wheels: PyPI 접근 없이 builder 의 휠만으로 설치.
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && pip check \
 && rm -rf /wheels
# Immutable serving images do not invoke Perl or account-management tools.
# Remove the installed package (including its dpkg record), never just scanner metadata.
# This runs only after all apt/user creation steps; do not run it on a serving host.
RUN apt-get purge -y --allow-remove-essential perl-base \
 && find /usr/bin /usr/sbin -xdev -type f -perm /6000 -exec chmod a-s {} + \
 && test ! -e /usr/bin/perl \
 && test -z "$(find /usr/bin /usr/sbin -xdev -type f -perm /6000 -print -quit)"
# 애플리케이션 소스를 app 사용자 소유로 복사.
COPY --chown=app:app app ./app
USER app
RUN python -c "import app.main; from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver"
EXPOSE 8000
# HEALTHCHECK
#   - /health 엔드포인트가 200 을 돌려주면 healthy.
#   - start-period 동안은 실패해도 unhealthy 로 집계하지 않아 lifespan 초기화 여유를 둔다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"
# ENTRYPOINT
#   - uvicorn 으로 app.main 모듈의 `app` (FastAPI 인스턴스) 를 0.0.0.0:8000 에 바인딩.
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
