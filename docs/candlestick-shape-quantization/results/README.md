# 연구 결과 해석 문서

run 산출물(`notebooks/runs/`, gitignore 대상)에 대한 해석 문서를 커밋 대상으로 모아둔 폴더다. 각 문서 상단에 원본 run 위치가 명시되어 있으며, 문서 안의 figure/table 링크는 로컬 체크아웃에서만 열린다. 게이트 판정의 요지는 [08. 연구 로드맵](../08-research-roadmap.md)에도 반영한다.

읽기 순서 (연구 진행 순):

1. [Phase 1 step-01: Shape Feature 검증](phase-01-step-01-shape-feature-validation.md) — shape core `(s1, s2)` 분포 검증, boundary candle point mass 발견
2. [Phase 1 step-01 미니스텝: 병합 분포 검증](phase-01-step-01-merge-distribution-check.md) — JSD 기반 D2 병합 설계 근거, "일봉·분봉 혼합 금지" 규칙
3. [Phase 1 step-02: Tokenizer Baselines](phase-01-step-02-tokenizer-baselines.md) — kmeans/GMM/bins 비교, kmeans K=32 최선, D2 병합 페널티 ~0 (H1 1차 근거)
4. [Phase 1 step-03: VQ Latent Clustering](phase-01-step-03-vq-latent-clustering.md) — 비교군 B 첫 구현, VQ-VAE latent clustering 1차 기각
5. [Phase 1 step-03: VQ Final Gate](phase-01-step-03-vq-final-gate.md) — 5-모델 timebox 비교, `kmeans_boundary_aware K=32` 최종 채택
6. [Phase 2 step-01: Token Corpus와 Label](phase-02-step-01-token-corpus.md) — corpus/label store 구축, entropy gate 판정
7. [Phase 2 step-01: Daily-only Gate 재집계](phase-02-step-01-daily-only-gate.md) — daily 트랙 단독 판정, 효과 크기 교정(유한표본 편향), daily 트랙의 잔존 역할
8. [Phase 2 step-01 미니스텝: 1m 구조 진단](phase-02-step-01-1m-structure-diagnosis.md) — persistence/시간대 효과/고차 구조 분해, interior 감사 종결
