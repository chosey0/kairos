# Phase 2 Token Sequence Corpus and Label 결과 해석

> 원본 run 위치: `notebooks/runs/candlestick-shape-quantization/phase-02-token-corpus/step-01-corpus-and-labels/` — 이 문서의 figure/table 링크는 gitignore된 로컬 run 산출물을 가리키므로 로컬 체크아웃에서만 열린다.

이 문서는 `04_token_corpus.ipynb`가 생성한 Phase 2 결과를 해석한다. 기준 run은 각 dataset의 최신 `seed-7` run이다.

관련 문서:

- [Phase 1 final gate](phase-01-step-03-vq-final-gate.md)
- [Research Roadmap](../08-research-roadmap.md)

## 실험 목적

Phase 2의 목적은 예측 모델 학습이 아니라, 다음 두 가지를 검증하는 것이다.

1. Phase 1에서 확정한 `kmeans_boundary_aware K=32` tokenizer로 8개 dataset의 token sequence corpus와 label store를 만든다.
2. token sequence의 bigram information gain이 symbol별 surrogate shuffle의 우연 수준과 구분되는지 확인해 Phase 3 motif 기대치를 조정한다.

Token vocabulary는 고정이다.

```text
token 0..31  : interior continuous KMeans codebook
token 32..39 : boundary discrete token 8개
token 40     : zero-range special token
```

## Dataset 처리

shape row는 step-01 최신 run의 `shape_sample.csv`에서 읽고, OHLC는 `01_shape_feature_validation.ipynb`와 같은 broker path로 재수집한 뒤 `(timestamp, symbol)` inner join했다. 모든 dataset의 join coverage는 100%로 acceptance 기준 99.5%를 통과했다.

| Dataset | Run | Rows | Coverage | Fractal pivots | Pivot ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `d1_kosdaq_1m` | `cfg-69ca7d50/run-20260707-092309_seed-7` | 94,236 | 100.00% | 19,444 | 20.63% |
| `d1_kosdaq_daily` | `cfg-9c8f1007/run-20260707-092126_seed-7` | 7,365 | 100.00% | 1,775 | 24.10% |
| `d1_kospi_1m` | `cfg-30e0ce9d/run-20260707-092234_seed-7` | 94,149 | 100.00% | 22,853 | 24.27% |
| `d1_kospi_daily` | `cfg-ac2bc6b5/run-20260707-092121_seed-7` | 9,420 | 100.00% | 2,322 | 24.65% |
| `d1_nasdaq_daily` | `cfg-1d685e8c/run-20260707-092129_seed-7` | 6,664 | 100.00% | 1,687 | 25.32% |
| `d1_spx_daily` | `cfg-ec5baf2d/run-20260707-092205_seed-7` | 5,302 | 100.00% | 1,343 | 25.33% |
| `d2_kr-kospi-kosdaq_1m` | `cfg-4d468e7b/run-20260707-092348_seed-7` | 188,385 | 100.00% | 42,297 | 22.45% |
| `d2_kr-kospi-kosdaq_daily` | `cfg-38bbcebe/run-20260707-092341_seed-7` | 16,785 | 100.00% | 4,097 | 24.41% |

Williams Fractal 컬럼은 Phase 3 segmentation 준비용이다. `fractal_confirmed_at`은 pivot timestamp가 아니라 pivot으로부터 n=2번째 뒤 candle timestamp이며, downstream에서는 이 시점 이후에만 fractal 정보를 사용할 수 있다.

## Label 처리

label은 모두 t+1 이후 정보만 사용한다.

- `fwd_log_return_h`, `direction_h`: h in `{1, 5, 20}`
- `direction_thr_h`: train split의 `|fwd_log_return_h|` 중앙값으로 neutral threshold 산정
- `fwd_rv_h`: Parkinson RV, h in `{5, 20}`
- `vol_expansion_h`: `fwd_rv_h / trailing_rv_h > 1.5`
- `drawdown_event_20`: train split max drawdown 90% quantile 초과
- `regime`: daily track only, 200-bar MA trend x train trailing vol tercile

Embargo 규칙 때문에 split boundary 근처 row와 horizon이 split을 넘는 row는 label null로 남겼다. 1m track은 200-bar daily regime의 의미가 달라 regime을 보류했고, `regime` null은 의도된 결과다.

## Token 분포 안정성

표는 canonical seed 7의 token entropy와, seeds 17/37 unigram distribution이 seed 7과 얼마나 다른지 KL divergence로 본 것이다.

| Dataset | seed 7 entropy | seed 17 KL | seed 37 KL |
| --- | ---: | ---: | ---: |
| `d1_kosdaq_1m` | 3.794 | 0.015238 | 0.017922 |
| `d1_kosdaq_daily` | 4.978 | 0.232803 | 0.412569 |
| `d1_kospi_1m` | 4.024 | 0.074206 | 0.088224 |
| `d1_kospi_daily` | 4.929 | 0.233818 | 0.193066 |
| `d1_nasdaq_daily` | 5.046 | 0.229638 | 0.207464 |
| `d1_spx_daily` | 4.847 | 0.168878 | 0.316327 |
| `d2_kr-kospi-kosdaq_1m` | 3.916 | 0.042599 | 0.053916 |
| `d2_kr-kospi-kosdaq_daily` | 4.949 | 0.220819 | 0.284877 |

해석:

- 1m은 boundary token 비중이 높아 seed 간 unigram KL이 작다.
- daily는 interior KMeans assignment가 더 큰 비중을 차지해 seed별 token id 분포 차이가 상대적으로 크다.
- Phase 3에서는 seed 7 canonical corpus를 기준으로 진행하되, motif가 seed-specific artifact인지 확인하는 stability check가 필요하다.

## Entropy Gate

Phase 2 gate는 train split 기준이다. `q`는 100회 symbol별 shuffle surrogate에서 관측 information gain이 위치한 quantile이다. 높을수록 surrogate보다 큰 순차 구조가 있다는 뜻이다.

| Dataset | Full IG | Full q | Interior IG | Interior q | Boundary IG | Boundary q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d1_kosdaq_1m` | 0.0428 | 1.00 | 0.3653 | 0.72 | 0.0294 | 1.00 |
| `d1_kosdaq_daily` | 0.3695 | 0.76 | 0.4118 | 0.51 | 0.2231 | 0.78 |
| `d1_kospi_1m` | 0.0273 | 1.00 | 0.1859 | 0.98 | 0.0095 | 1.00 |
| `d1_kospi_daily` | 0.3554 | 0.96 | 0.3777 | 0.45 | 0.2166 | 0.78 |
| `d1_nasdaq_daily` | 0.3551 | 0.80 | 0.3421 | 0.53 | 0.2767 | 0.46 |
| `d1_spx_daily` | 0.3541 | 0.51 | 0.4942 | 0.03 | 0.0603 | 0.43 |
| `d2_kr-kospi-kosdaq_1m` | 0.0234 | 1.00 | 0.1188 | 0.98 | 0.0162 | 1.00 |
| `d2_kr-kospi-kosdaq_daily` | 0.1946 | 0.97 | 0.1985 | 0.34 | 0.1771 | 0.92 |

해석:

- 1m full/boundary vocab은 q=1.00으로 surrogate보다 뚜렷하다. 다만 이 구조는 boundary token의 장중 미세구조 반복에 지배될 수 있다.
- 1m interior-only도 KOSPI와 D2 KR에서는 q=0.98로 강하지만, KOSDAQ 1m은 q=0.72로 약하다.
- daily full vocab은 dataset별로 혼재한다. KOSPI daily와 D2 KR daily는 높고, SPX daily는 q=0.51로 surrogate와 구분되지 않는다.
- daily interior-only는 대체로 약하다. SPX는 q=0.03으로 관측 IG가 surrogate보다 낮다.
- 따라서 Phase 3 motif는 full-vocab 전체 평균으로 밀어붙이면 boundary-dominated motif를 과대평가할 수 있다. 반드시 full/interior/boundary decomposition을 유지해야 한다.

## Gate 판정

Phase 2 gate는 go/no-go가 아니라 기대치 조정 gate다.

판정:

- 순차 구조는 존재한다. 특히 1m full/boundary track과 KR daily full track은 surrogate 우연 수준을 넘는다.
- 하지만 interior-only 순차 구조는 dataset별로 불안정하다. daily interior motif는 약한 후보로 취급해야 한다.
- Phase 3는 진행하되, 기대치를 다음처럼 조정한다.

```text
high-priority:
  1m full/boundary-aware motif
  KR daily full-vocab motif

controlled ablation:
  interior-only motif
  coarse_fine low-resolution sequence
  Williams Fractal leg segmentation(H->L / L->H)

required guardrail:
  Markov-1 surrogate test와 vocab decomposition을 유지
```

## 판정 범위와 효과 크기 노트

- **통계적 유의성과 효과 크기를 구분해야 한다.** 1m의 q=1.00은 표본이 9만~19만 row라 미세한 구조도 유의하게 잡힌 결과이며, IG 절대값은 entropy 대비 약 1%(0.02~0.04 / 3.8~4.0)에 불과하다. "이전 캔들이 다음 캔들을 알려준다"가 아니라 "아주 약한 반복 구조가 확실히 존재한다"로 읽어야 한다.
- daily의 IG 절대값(~0.35)은 대부분 유한표본 편향이다. daily만 재집계한 [daily-only aggregate run](phase-02-step-01-daily-only-gate.md)에서 full vocab의 진짜 초과 구조는 0.01 수준(entropy의 ~0.2%)으로 확인됐다. surrogate 대조 없는 IG 수치는 인용하지 않는다.
- **이 gate는 H4(multi-candle motif 존재)에 대한 판정이다.** daily에서 순차 구조가 약하다는 것이 daily 토큰화 연구의 무의미를 뜻하지 않는다 — H1/H2 검증, Phase 4 해석성, Phase 5 단일 token downstream은 daily가 주 무대다. 상세는 daily-only 문서의 "판정 범위와 daily 트랙의 잔존 역할" 참조.

## Figure 읽는 법

각 dataset run에는 figure 4종이 있다.

- `token_frequency_by_symbol_split.png`: symbol/split별 token 분포. D2에서는 KOSPI/KOSDAQ 분해를 확인한다.
- `jsd_heatmap.png`: symbol, split, regime 조합 간 Jensen-Shannon distance.
- `transition_matrix_interior_train.png`: train split interior-only transition matrix.
- `information_gain_vs_surrogate_train.png`: full/interior/boundary vocab의 observed IG와 surrogate mean 비교.

## Caveats

- 1m 데이터는 provider history 제약으로 약 1년 구간이다. Phase 3에서 과도한 일반화는 금물이다.
- 1m regime은 보류했다. daily의 200-bar MA x trailing vol tercile 정의를 그대로 적용하지 않는다.
- Corpus는 seed 7을 canonical으로 저장했다. seed 17/37은 token distribution stability 확인용이다.
- Williams Fractal은 저장만 했다. leg segmentation과 예측 anchor는 Phase 3/5에서 `fractal_confirmed_at` 이후로만 사용할 수 있다.
- run artifacts(config/metrics/figures)는 `notebooks/runs/` 아래 로컬 산출물이며 gitignore 대상이다. 이 해석 문서만 저장소에 커밋된다.
