# Module 4. 웹사이트 프론트엔드 구현 계획

## 기술 스택 선정
- **코어 프레임워크**: `Vite` 기반의 순수 바닐라 환경 (HTML/JS/CSS).
  - *이유*: React 등 무거운 프레임워크의 의존성을 배제하고, Canvas 렌더링과 DOM 조작에 있어 최대한의 렌더링 퍼포먼스(60FPS 스와이프)를 확보하기 위함입니다.
- **스타일링**: Vanilla CSS3 + CSS Variables (Tailwind 등 미사용, 완전 맞춤 제어).
- **로컬 서버구동**: Python 내장 `http.server` 또는 `vite` 자체 dev 서버를 사용해 처리된 이미지 파일(정적 파일)을 직접 호스팅.

## 디렉토리 구조 예상
프로젝트 루트 내 별도 `webapp/` 폴더를 생성하여 관리합니다.

```text
webapp/
├── index.html          # 메인 엔트리 (대시보드 쉘)
├── style.css           # 글로벌 디자인 토큰 및 UI 스타일링
├── main.js             # 애플리케이션 초기화 및 이벤트 리스너
├── components/
│   ├── Slider.js       # Before/After 비교 핸들러 클래스
│   ├── Stitcher.js     # 다중 칩 렌더링 캔버스 클래스
│   └── Panel.js        # 데이터셋 선택 드롭다운 및 메뉴
└── assets/             # 아이콘 등 정적 리소스
```

## 단계별 개발 마일스톤
1. **[기반 작업]**: `Vite` 환경 세팅, 다크모드 디자인 시스템(CSS Variables) 작성.
2. **[컴포넌트 1]**: `Slider.js` 개발. 마우스 드래그 이벤트 기반의 `clip-path` 제어 로직 완성.
3. **[컴포넌트 2]**: `Stitcher.js` 개발. 더미 타일 이미지를 로드하여 Zoom & Pan 캔버스 뷰어 구현.
4. **[통합 및 데이터 연결]**: Python 모듈(Module 3)에서 생성한 폴더 경로(`data/output/`)를 읽어, 웹상에서 K3A/S2 씬 리스트를 동적으로 구성하도록 연결.
