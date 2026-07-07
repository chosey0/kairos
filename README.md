# kairos

OHLC 캔들을 해석 가능한 discrete shape token으로 양자화하고, token sequence에서 다중 캔들 motif vocabulary를 학습하는 연구 프로젝트다.

핵심 질문:

> 절대 가격, 변동폭, 시장 단위가 달라도 캔들 내부 모양이 비슷하면 같은 shape token으로 묶을 수 있는가?
> 그리고 그 token sequence에는 우연 수준을 넘어 반복되고 예측에 유용한 multi-candle motif가 존재하는가?

## 연구 개요

캔들 하나를 가격 수준·변동폭과 분리된 2D shape core로 표현하고, 이를 discrete token으로 양자화한 뒤, token sequence에서 motif vocabulary를 학습한다.

```text
OHLC
  -> shape core (logit λ_o, logit λ_c)         # scale-invariant 캔들 내부 모양
  -> shape_token                                # k-means/GMM/VQ-VAE/FSQ 비교로 선택
  -> token sequence + side channels             # rel_range, gap은 별도 연속 채널
  -> multi-candle motif vocabulary              # BPE/Unigram/Sequitur + surrogate 검정
  -> 해석성 평가 + downstream 예측 유용성 평가
```

전체 계획은 Phase 0(데이터/leakage 프로토콜)부터 Phase 6(결론)까지 결정 게이트가 있는 로드맵으로 관리한다: [docs/candlestick-shape-quantization/08-research-roadmap.md](docs/candlestick-shape-quantization/08-research-roadmap.md)

데이터는 3단계로 확장한다.

```text
D1. 단일 지수 (KOSPI/KOSDAQ/NASDAQ/SPX/DJI/NDX/SOX)  파이프라인·설계 확정
D2. 국가별 주요 지수 (한국, 미국 master-validated)       같은 시장 내 일반화 검증
D3. 세계 주요 지수 (9개 master-validated universe)       cross-market transfer 검증
```

## 연구 진행 현황 (2026-07-07 기준)

Phase 0~1이 완료되고 Phase 2 게이트 판정까지 끝났다. 확정 tokenizer는 `kmeans_boundary_aware K=32` — 전체 vocabulary 41 = interior codebook 32 + boundary discrete token 8 + zero-range special token 1.

| 단계 | 핵심 결과 | 해석 문서 |
| --- | --- | --- |
| Phase 0 데이터/프로토콜 | 8개 dataset 확정 (D1 daily 4 + D1 KR 1m 2 + D2 KR 병합 2), daily/minute split·embargo 고정 | [protocol JSON](kairos/experiments/protocols/candlestick_shape_quantization.json) |
| Phase 1 step-01 shape feature 검증 | boundary candle point mass 발견 — daily 12~27%, KR 1m 73~81%. boundary 분리 설계(비교군 B)의 근거 | [FIGURE_EXPLANATION](docs/candlestick-shape-quantization/results/phase-01-step-01-shape-feature-validation.md) |
| 미니스텝: 병합 분포 검증 | 같은 interval 병합은 JSD 0.03~0.06으로 분포 유지, daily+1m 혼합은 0.30~0.52로 붕괴 → D2 병합 설계·"일봉과 분봉은 섞지 않는다" 규칙의 근거 | [MERGE_DISTRIBUTION_CHECK](docs/candlestick-shape-quantization/results/phase-01-step-01-merge-distribution-check.md) |
| Phase 1 step-02 tokenizer baselines | 전 dataset·정책에서 kmeans K=32 최선 (dead token 0, effective vocab 24~30). D2 병합 페널티 ~0.2% — **H1 1차 근거**. 분봉 boundary의 비용은 MSE가 아니라 codebook 용량 잠식 | [RESULTS_EXPLANATION](docs/candlestick-shape-quantization/results/phase-01-step-02-tokenizer-baselines.md) |
| Phase 1 step-03 VQ final gate | boundary-aware vocabulary(41) 확정. VQ-VAE latent/FSQ/BSQ는 reconstruction·stability·usage 전부 기각, `coarse_fine`은 Phase 3 저해상도 sequence ablation으로만 보류 | [VQ latent](docs/candlestick-shape-quantization/results/phase-01-step-03-vq-latent-clustering.md) · [final gate](docs/candlestick-shape-quantization/results/phase-01-step-03-vq-final-gate.md) |
| Phase 2 corpus + label store | 8개 dataset token corpus(join coverage 100%) + label store(fwd return/RV/drawdown/regime) + Williams Fractal 컬럼(Phase 3 leg segmentation 준비, `fractal_confirmed_at`으로 look-ahead 차단) | [RESULTS_EXPLANATION](docs/candlestick-shape-quantization/results/phase-02-step-01-token-corpus.md) |
| Phase 2 entropy gate | 1m은 순차 구조 유의(q=1.00)하나 효과 크기는 entropy의 ~1%. daily는 유한표본 편향 제거 시 사실상 구조 없음(KR full-vocab 예외) → **Phase 3 motif는 1m 트랙 집중** | [daily-only aggregate](docs/candlestick-shape-quantization/results/phase-02-step-01-daily-only-gate.md) |
| 미니스텝: 1m 구조 진단 | 1m 구조는 단순 run·시간대 효과로 소진되지 않음 — RLE 후에도 IG q=1.00, bucket 보존 surrogate에서도 유의, 2차 IG가 Markov-1 초과. interior 감사: 인공 adjacency 없음(daily 이상치는 plug-in bias) | [1M_STRUCTURE_DIAGNOSIS](docs/candlestick-shape-quantization/results/phase-02-step-01-1m-structure-diagnosis.md) |

가설 현황: **H1**(shape의 가격 분리·cross-index 일반화) 1차 검증, **H2**(반복 shape vocabulary 존재) 부분 검증 — daily effective vocab 29~33, **H3**(side-channel 분리) 설계 반영 완료, **H4**(multi-candle motif) daily는 사실상 기각·1m은 Markov-1을 넘는 고차 구조까지 확인. daily 트랙은 motif 경로만 접고 표현력·일반화·해석성·downstream 층위에서는 계속 주 트랙이다.

다음 단계: Phase 3 motif vocabulary — 1m 트랙 집중, RLE ablation·time-bucket 층화·Markov-1 surrogate 검정을 필수 guardrail로 적용.

해석 문서 전체 목록과 읽기 순서는 [docs/candlestick-shape-quantization/results/](docs/candlestick-shape-quantization/results/README.md)를 따른다. 해석 문서 안의 figure/table 링크는 gitignore된 로컬 run 산출물(`notebooks/runs/`)을 가리키므로 로컬 체크아웃에서만 열린다.

## 저장소 구조

```text
kairos/       연구 파이프라인 패키지
                core/                캔들 계약, feature 추출, 모델, metric, 학습 helper
                experiments/         protocol JSON, artifact helper, 단계별 실험 로직
                sources/             read-only market-data access helper
tests/        leakage-sensitive 순수 함수 단위 테스트
notebooks/    실험 노트북과 run 산출물
                candlestick-shape-quantization/   노트북 소스 (00 -> 05 순서 실행)
                runs/                             실험 config/metrics/figures
docs/         연구 설계 문서 (아래 문서 안내 참조)
references/   참조 논문 PDF
AGENTS.md     에이전트/기여자 작업 규칙 (연구 규칙, leakage 규칙 포함)
```

## 시작하기

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요하다.

```bash
uv sync                          # 의존성 설치
uv run --with pytest pytest -q   # 단위 테스트
```

feature 추출 예시:

```python
from kairos.core.features import extract_features, extract_shape

shape = extract_shape(candle)          # 단일 캔들 -> ShapeFeatures (s1, s2, 경계 flag)
rows = extract_features(candles)       # 시계열 -> shape core + rel_range/gap 채널
```

실험 노트북은 `notebooks/candlestick-shape-quantization/`의 번호 순서대로 실행한다. Phase 0 프로토콜 설정은 `kairos/experiments/protocols/candlestick_shape_quantization.json`을 source of truth로 사용한다. run 산출물 규약은 [notebooks/AGENTS.md](notebooks/AGENTS.md)를 따른다.

## 문서 안내

연구 설계 문서는 `docs/candlestick-shape-quantization/`에 있다. 권장 읽기 순서:

1. [01. 연구 목적과 범위](docs/candlestick-shape-quantization/01-problem-and-research-scope.md)
2. [08. 연구 로드맵](docs/candlestick-shape-quantization/08-research-roadmap.md) — source of truth
3. [02. Shape Tokenizer 설계](docs/candlestick-shape-quantization/02-shape-tokenizer.md)
4. [00. VQ-VAE Tokenizer 전제 검증 계획](docs/candlestick-shape-quantization/00-vq-vae-tokenizer-validation-plan.md)
5. [07. 근거와 참고 문헌](docs/candlestick-shape-quantization/07-evidence-and-references.md)
6. [연구 결과 해석 문서 모음](docs/candlestick-shape-quantization/results/README.md) — run별 결과 해석과 게이트 판정 기록

## 연구 규칙 요약

자세한 내용은 [AGENTS.md](AGENTS.md)와 로드맵을 따른다.

- **Leakage 방지 최우선.** 시간 기준 split 확정 전에는 어떤 모델·scaler·tokenizer도 fit하지 않는다. trailing 통계(ATR, volume median)는 t-1까지만 사용한다.
- **Shape core는 `(logit λ_o, logit λ_c)` 2D.** body/wick 4D feature는 파생 리포팅 전용이다. `rel_range`/`gap`은 token에 섞지 않는 별도 채널이다.
- **결정 게이트.** VQ 계열이 k-means를 이기지 못하면 k-means를 채택하고, motif가 Markov-1 surrogate 검정을 통과하지 못하면 단일 token 경로로 진행한다.
- **실험 기록.** 모든 run은 `experiment_config.json`/`metrics.json`을 남기고, seed 3개 이상, 지수별·regime별 분리 보고를 기본으로 한다.

## 참조 논문

PDF는 `references/`에 있으며, 논문별 설계 연결은 [로드맵의 Reference 논문 매핑](docs/candlestick-shape-quantization/08-research-roadmap.md#reference-논문-매핑)을 따른다.

### 표현 학습과 양자화

| 논문 | 링크 | 역할 |
| --- | --- | --- |
| Dimension reduction of OHLC data based on pseudo-PCA | [arXiv:2103.16908](https://arxiv.org/abs/2103.16908) | 제약 없는 bijective OHLC 표현. shape core의 직접 근거 |
| Kronos: A Foundation Model for the Language of Financial Markets | [arXiv:2508.02739](https://arxiv.org/abs/2508.02739) · [code](https://github.com/shiyu-coder/Kronos) | K-line 전용 tokenizer(BSQ), coarse/fine 계층 토큰, 공개 pretrained baseline |
| Neural Discrete Representation Learning (VQ-VAE) | [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) | discrete codebook 학습의 원형 |
| TimeVQVAE | [arXiv:2303.04743](https://arxiv.org/abs/2303.04743) | coarse/fine codebook 분리, VQ 기반 시계열 생성 |
| TSAX: A Novel Trend Symbolic Aggregate Approximation | [arXiv:1905.00421](https://arxiv.org/abs/1905.00421) | 심볼화에서 방향/trend 보존의 근거, Phase 5 baseline |
| TS2Vec | [arXiv:2106.10466](https://arxiv.org/abs/2106.10466) | 양자화 전 continuous representation baseline |
| VQ-AR | [arXiv:2205.15894](https://arxiv.org/abs/2205.15894) | discrete representation 기반 forecasting 비교 프레임 |

### 캔들 패턴 인식과 해석성

| 논문 | 링크 | 역할 |
| --- | --- | --- |
| Encoding Candlesticks as Images (GAF-CNN) | [arXiv:1901.05237](https://arxiv.org/abs/1901.05237) | 8개 표준 패턴 분류 체계. 해석성 평가의 ground truth |
| Explainable Deep Convolutional Candlestick Learner | [arXiv:2001.02767](https://arxiv.org/abs/2001.02767) | perturbation 기반 shape 민감도 분석 방법론 |
| Dynamic Deep Convolutional Candlestick Learner | [arXiv:2201.08669](https://arxiv.org/abs/2201.08669) | variable-length 패턴 위치 탐지 관점 |
| DL + Candlestick Chart Representation | [arXiv:1903.12258](https://arxiv.org/abs/1903.12258) | 캔들 이미지 CNN 예측 baseline, window/volume ablation |
| Deep Stock Representation Learning | [arXiv:1709.03803](https://arxiv.org/abs/1709.03803) | 비지도 캔들 representation을 downstream 의사결정으로 연결 |

주의: arXiv `2508.02739`는 Kronos 논문이다. VLM candlestick 벤치마크로 오인하지 않는다.
