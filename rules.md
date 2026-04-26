# 프로젝트 코딩 가이드라인 (Rules)

# K3A_S2_MTF 프로젝트 코딩 가이드라인


# 프로젝트 핵심

**Kompsat-3A 영상을 Sentinel-2 영상에 모사**
1. 주어진 K3A 영상과 100% 겹치는 Sentinel-2 영상 검색
2. 방사모사
3. 상대적 MTF값 계산
4. 최종 2,3번이 완료된 Simulated K3A 영상 생성


## 1. 프로젝트 구조 규칙 (Project Structure Rules)

### 1.1. 기본 디렉토리 구조
프로젝트는 다음 구조를 엄격히 준수해야 합니다.
```
K3A_S2_MTF/
├── src/
│   ├── frontend/          # React 클라이언트
│   ├── backend/           # Flask 백엔드
│   ├── agents/            # LangGraph 에이전트
│   └── scripts/           # 유틸리티 스크립트
├── tests/
├── docs/
└── requirements.txt
```

### 1.2. 마이크로커널 아키텍처
- **모든 기능은 독립적인 에이전트로 구현**되어야 함
- 각 에이전트는 `src/agents/` 디렉토리에 별도 모듈로 존재
- 에이전트 간 통신은 LangGraph `workflow.invoke()` 메서드로만 가능
- 직접 함수 호출 금지, 반드시 LangChain Expression Language (LCEL) 사용

### 1.3. 코드 모듈화 규칙
- 하나의 파일은 500줄 이하로 작성
- 주석: 파일 상단에 10줄 이내로 기능 요약
- 타입 힌트: 모든 함수와 클래스에 필수 적용

## 2. 언어 및 프레임워크 규칙

### 2.1. 백엔드 (Backend)
- **환경 변수**: `.env` 파일 사용, 절대 경로 저장 금지

### 2.3. 에이전트 (Agents)
- **프레임워크**: LangGraph 1.x
- **LLM**: OpenAI GPT-4o 또는 Claude Opus 4.5 사용
- **메모리**: `MessageMemory` 사용, 1000개 메시지 제한
- **도구**: `tool` 데코레이터 사용, 타입 힌트 필수

## 3. API 설계 규칙

### 3.1. RESTful API 명명 규칙
- **명사 사용**: `/users`, `/orders` 등
- **HTTP 메서드**: GET, POST, PUT, DELETE만 사용
- **버전 관리**: `/v1/` 접두사 사용
- **응답 형식**: JSON only, Content-Type: application/json

### 3.2. API 보안 규칙
- 모든 엔드포인트에 rate limiting 적용 (100회/분)
- SQL Injection 방지: 매개변수화된 쿼리 사용
- 입력값 검증: Joi 또는 Pydantic 사용
- CORS: 특정 도메인만 허용

## 4. 데이터베이스 규칙

### 4.1. 스키마 설계
-snake_case 사용
-모든 테이블에 soft delete 컬럼(`is_deleted`, `deleted_at`) 추가
-인덱스: 쿼리 100ms 초과 시 인덱스 추가
-데이터 타입: JSON 필드는 PostgreSQL JSONB 타입 사용

## 5. 성능 최적화 규칙

### 5.1. API 성능 규칙
- 응답 시간: 95 percentile 200ms 이하
- 페이징: limit 100, offset 0 기본값
- 캐싱: Redis 5분 캐싱 적용
- CDN: 모든 정적 파일은 CDN 사용

### 5.2. LLM 최적화 규칙
-temperature = 0.7 고정
-prompt 길이: 5000 토큰 이하
-비용 절감: 20k 토큰 초과 시 경고 로그

## 6. 보안 규칙

- 비밀번호: bcrypt 해시 사용
- 입력값 검증: 모두 필수
- 로깅: 보안 이벤트는 별도 DB에 저장

