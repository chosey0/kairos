# Phase 2 Daily-only Token Sequence Gate 결과 해석

> 원본 run 위치: `notebooks/runs/candlestick-shape-quantization/phase-02-token-corpus/step-01-corpus-and-labels/daily-only/cfg-1b2fb19c/run-20260707-100018_seed-7/` — 이 문서의 figure/table 링크는 gitignore된 로컬 run 산출물을 가리키므로 로컬 체크아웃에서만 열린다.

이 문서는 Phase 2 corpus 결과 중 daily dataset만 다시 묶어 순차 구조 gate를 재판정한 aggregate run을 해석한다. 기준 run은 `run-20260707-100018_seed-7`이다.

## 목적

전체 8개 dataset 판정에서는 1m full/boundary vocab의 강한 반복 구조가 전체 결론을 지배할 수 있다. 이 daily-only run은 `d1_*_daily` 4개와 `d2_kr-kospi-kosdaq_daily`만 사용해 daily sequence에서 motif 학습 근거가 충분한지 따로 확인한다.

## Dataset

| Dataset | Source run | Rows | Coverage | Fractal pivots | Pivot ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `d1_kospi_daily` | `cfg-ac2bc6b5/run-20260707-092121_seed-7` | 9,420 | 100.00% | 2,322 | 24.65% |
| `d1_kosdaq_daily` | `cfg-9c8f1007/run-20260707-092126_seed-7` | 7,365 | 100.00% | 1,775 | 24.10% |
| `d1_nasdaq_daily` | `cfg-1d685e8c/run-20260707-092129_seed-7` | 6,664 | 100.00% | 1,687 | 25.32% |
| `d1_spx_daily` | `cfg-ec5baf2d/run-20260707-092205_seed-7` | 5,302 | 100.00% | 1,343 | 25.33% |
| `d2_kr-kospi-kosdaq_daily` | `cfg-38bbcebe/run-20260707-092341_seed-7` | 16,785 | 100.00% | 4,097 | 24.41% |

모든 daily dataset의 OHLC join coverage는 100%다.

## Entropy Gate

표의 값은 `information_gain / surrogate_quantile`이다. train split 기준이며 surrogate는 symbol별 token shuffle 100회다.

| Dataset | Full | Interior | Boundary |
| --- | ---: | ---: | ---: |
| `d1_kospi_daily` | 0.3554 / 0.96 | 0.3777 / 0.45 | 0.2166 / 0.78 |
| `d1_kosdaq_daily` | 0.3695 / 0.76 | 0.4118 / 0.51 | 0.2231 / 0.78 |
| `d1_nasdaq_daily` | 0.3551 / 0.80 | 0.3421 / 0.53 | 0.2767 / 0.46 |
| `d1_spx_daily` | 0.3541 / 0.51 | 0.4942 / 0.03 | 0.0603 / 0.43 |
| `d2_kr-kospi-kosdaq_daily` | 0.1946 / 0.97 | 0.1985 / 0.34 | 0.1771 / 0.92 |

Weighted aggregate:

| Vocab | IG weighted | Surrogate mean weighted | Mean quantile | Mean z-score |
| --- | ---: | ---: | ---: | ---: |
| `full` | 0.3038 | 0.2933 | 0.80 | 1.10 |
| `interior` | 0.3301 | 0.3361 | 0.37 | -0.45 |
| `boundary` | 0.1559 | 0.1242 | 0.67 | 0.50 |

## Seed Stability

| Dataset | seed 7 entropy | seed 17 KL | seed 37 KL |
| --- | ---: | ---: | ---: |
| `d1_kospi_daily` | 4.929 | 0.233818 | 0.193066 |
| `d1_kosdaq_daily` | 4.978 | 0.232803 | 0.412569 |
| `d1_nasdaq_daily` | 5.046 | 0.229638 | 0.207464 |
| `d1_spx_daily` | 4.847 | 0.168878 | 0.316327 |
| `d2_kr-kospi-kosdaq_daily` | 4.949 | 0.220819 | 0.284877 |

## Gate 판정

Daily-only에서는 순차 구조가 전 dataset에서 균일하게 강하지 않다.

- Full vocab: KOSPI daily와 D2 KR daily는 강하고, KOSDAQ/NASDAQ은 중간, SPX는 surrogate와 거의 구분되지 않는다.
- Interior-only: 대부분 surrogate와 구분되지 않으며, SPX는 관측 IG가 surrogate보다 낮다.
- Boundary-only: KR daily 쪽은 비교적 강하지만 NASDAQ/SPX는 약하다.

따라서 Phase 3 daily motif는 진행하되 기대치를 낮춘다. Daily track에서는 `full vocab`과 `KR daily`를 우선 확인하고, `interior-only daily motif`는 강한 주 후보가 아니라 ablation으로 둔다.

## 효과 크기 해석 (weighted aggregate)

Weighted aggregate 표가 말하는 가장 중요한 사실은 quantile이 아니라 IG와 surrogate mean의 차이다.

```text
full:     0.3038 - 0.2933 = 0.0105  (seed 7 entropy ~4.9의 약 0.2%)
interior: 0.3301 - 0.3361 = -0.0060 (초과 구조 없음)
boundary: 0.1559 - 0.1242 = 0.0317
```

daily IG 절대값(~0.35)의 대부분은 순차 구조가 아니라 유한표본 편향이다. 5천~1.7만 row로 41-vocab bigram 행렬을 추정하면 셔플로 구조를 완전히 파괴한 sequence에서도 IG가 0.29 수준으로 나온다. 진짜 초과 구조는 full vocab 기준 0.01 수준이며, 이는 raw IG만 보고 "daily에 7% 수준의 순차 구조가 있다"고 읽으면 안 된다는 뜻이다. surrogate 대조 없는 IG 수치는 인용하지 않는다.

## 판정 범위와 daily 트랙의 잔존 역할

이 gate 판정은 **H4(token sequence에 multi-candle motif가 존재하는가)에 한정된 판정**이다. "daily 토큰화 연구가 의미 없다"로 읽으면 안 된다. daily 트랙에서 접는 것은 motif 경로뿐이며, 나머지 층위는 오히려 daily가 주 무대다.

- **H2 (shape vocabulary 존재)**: daily는 boundary 12~27%에 effective vocab 29~33으로 codebook이 가장 풍부하게 쓰이는 트랙이다. 1m(boundary 73~81%, effective vocab 14~16)보다 tokenizer 관점에서 건강하다.
- **H1 (cross-index 일반화)**: D2 병합 무손실, 지수 간 token 점유율 유사성 등 H1의 1차 증거가 전부 daily에서 나왔고, 29~36년 커버리지의 regime 다양성은 daily에만 있다. 1m(약 1년)은 이 검증을 대체할 수 없다.
- **Phase 4 (해석성)**: 전통 캔들 패턴 ground truth와의 overlap 평가는 daily 패턴 정의 위에 서 있다.
- **Phase 5 (downstream)**: "token이 미래 label에 정보를 갖는가"는 순차 구조 없이 단일 token + continuous side-channel로 검증하며, label store(regime 포함)는 daily에만 완전하게 구축되어 있다. 순차 구조 부재는 roadmap Phase 5의 "single token으로 충분한가" 비교를 오히려 명확하게 만든다.

요약: daily의 sequence 층위(H4)는 사실상 기각(KR full 예외, interior 감사 결과 대기)이고, 표현력(H2)·일반화(H1)·해석성(Phase 4)·downstream(H5) 층위는 daily가 계속 주 트랙이다.

## Figures

- [Daily information gain by vocab](../../../notebooks/runs/candlestick-shape-quantization/phase-02-token-corpus/step-01-corpus-and-labels/daily-only/cfg-1b2fb19c/run-20260707-100018_seed-7/figures/daily_information_gain_by_vocab.png)
- [Daily surrogate quantile heatmap](../../../notebooks/runs/candlestick-shape-quantization/phase-02-token-corpus/step-01-corpus-and-labels/daily-only/cfg-1b2fb19c/run-20260707-100018_seed-7/figures/daily_surrogate_quantile_heatmap.png)

## Caveats

- 이 run은 기존 Phase 2 per-dataset run을 daily dataset만 재집계한 aggregate다.
- Williams Fractal 컬럼은 여전히 저장 전용이며, 예측 anchor는 `fractal_confirmed_at` 이후여야 한다.
- run artifacts(config/metrics/figures)는 `notebooks/runs/` 아래 로컬 산출물이며 gitignore 대상이다. 이 해석 문서만 저장소에 커밋된다.
