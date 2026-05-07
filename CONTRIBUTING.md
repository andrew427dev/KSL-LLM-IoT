# Contributing Guide — KSL-LLM-IoT

## 브랜치 전략

```
main    — 안정 버전 (발표/제출용)
dev     — 통합 개발 브랜치
feature/<기능명>  — 기능 단위 개발 (dev에서 분기, dev로 머지)
fix/<버그명>      — 버그 수정
```

**작업 흐름:**
```
feature/xxx → PR → dev → (통합 테스트 통과) → PR → main
```

## 커밋 메시지 컨벤션

```
<type>: <요약>

type 목록:
  feat     — 새 기능
  fix      — 버그 수정
  refactor — 동작 변경 없는 코드 개선
  test     — 테스트 추가/수정
  docs     — 문서/주석
  chore    — 빌드, 설정, 의존성
```

예시:
```
feat: add silence trigger to sentence_builder
fix: clear sequence buffer after successful prediction
```

## PR 규칙

- `main`에 직접 push 금지 — 반드시 PR을 통해 머지
- PR 머지 전 상대방 리뷰 1건 필수
- PR 제목은 커밋 컨벤션과 동일한 형식 사용

## 환경 설정

```bash
# PC (훈련/개발)
pip install -r requirements.txt

# Raspberry Pi (배포)
pip install -r requirements-rpi.txt
```

- `.env` 파일은 절대 커밋하지 않는다 (`.gitignore` 적용됨)
- 모델 파일(`.tflite`, `.keras`)도 커밋하지 않는다
