# 1m Structure Diagnosis

> 원본 run 위치: `notebooks/runs/candlestick-shape-quantization/phase-02-token-corpus/step-01-corpus-and-labels/1m-structure-diagnosis/` — 이 문서의 figure/table 링크는 gitignore된 로컬 run 산출물을 가리키므로 로컬 체크아웃에서만 열린다.

관련 문서: [Phase 2 Token Corpus 결과 해석](phase-02-step-01-token-corpus.md)

## 목적

Phase 2에서 1m corpus의 full/boundary vocab information gain은 surrogate 대비 q=1.00이었지만,
절대값은 entropy의 약 1% 수준이었다. 이 문서는 모델 학습 없이 기존 seed-7 corpus CSV만 사용해
그 순차 구조가 무엇인지 진단한다.

사용 데이터는 Phase 2 canonical corpus의 최신 seed-7 run이다. timestamp는 timezone 없는 거래소 현지시각으로
기록되어 있으며, 1m bucket은 KST 정규장 09:00-15:30 기준으로 배정했다.

## Q1. Persistence 여부

MI는 `I = sum p(i,j) log2(p(i,j)/(p(i)p(j)))`로 재계산했고, 대각(i=j)과 비대각을 분리했다.
대각+비대각 합은 전체 MI와 일치한다.

| dataset | mi_bits | diagonal_bits | off_diagonal_bits | diagonal_share | bigram_count |
| --- | --- | --- | --- | --- | --- |
| d1_kospi_1m | 0.0296 | 0.0115 | 0.0181 | 0.3877 | 55185 |
| d1_kosdaq_1m | 0.0449 | 0.0360 | 0.0090 | 0.8006 | 55185 |
| d2_kr-kospi-kosdaq_1m | 0.0256 | 0.0250 | 0.0006 | 0.9761 | 110370 |

RLE(연속 중복 제거) 후 full-vocab IG는 다음과 같다.

| dataset | raw_ig_bits | rle_ig_bits | rle_surrogate_mean | rle_surrogate_quantile | rle_z_score |
| --- | --- | --- | --- | --- | --- |
| d1_kospi_1m | 0.0273 | 0.1599 | 0.0239 | 1.0000 | 138.3303 |
| d1_kosdaq_1m | 0.0428 | 0.1950 | 0.0260 | 1.0000 | 166.2692 |
| d2_kr-kospi-kosdaq_1m | 0.0234 | 0.1625 | 0.0129 | 1.0000 | 284.4867 |

판정: same-token persistence 비중이 크지만 RLE 후에도 강한 transition 구조가 남는다. raw/RLE 양쪽 ablation이 필요하다.

## Q2. Time-Bucket Effect 여부

bucket은 `[09:00, 09:30)`, `[09:30, 15:00)`, `[15:00, 15:30]`로 나눴다.
bucket-preserving surrogate는 symbol별로 같은 bucket 안에서 token만 섞어 시간대 marginal을 보존하고 순서를 파괴했다.

| dataset | raw_ig_bits | bucket_stratified_ig_bits | bucket_preserving_surrogate_mean | bucket_preserving_surrogate_quantile | bucket_preserving_z_score |
| --- | --- | --- | --- | --- | --- |
| d1_kospi_1m | 0.0273 | 0.0265 | 0.0212 | 1.0000 | 7.8581 |
| d1_kosdaq_1m | 0.0428 | 0.0419 | 0.0223 | 1.0000 | 24.5505 |
| d2_kr-kospi-kosdaq_1m | 0.0234 | 0.0225 | 0.0113 | 1.0000 | 24.9945 |

판정: bucket marginal을 보존해도 observed IG가 남는다. 시간대 층화는 필요하지만 구조 전체가 시간대 효과는 아니다.

## Q3. Bigram 너머 구조

2차 IG는 `H(X_t | X_{t-1}) - H(X_t | X_{t-1}, X_{t-2})`로 계산했다.
대조군은 train transition matrix에서 생성한 Markov-1 surrogate 100회다.

| dataset | second_order_ig_bits | markov1_surrogate_mean | markov1_surrogate_quantile | markov1_z_score | trigram_count |
| --- | --- | --- | --- | --- | --- |
| d1_kospi_1m | 0.3366 | 0.3208 | 1.0000 | 3.9505 | 55040 |
| d1_kosdaq_1m | 0.2814 | 0.2564 | 1.0000 | 7.2915 | 55040 |
| d2_kr-kospi-kosdaq_1m | 0.2086 | 0.1960 | 1.0000 | 6.1173 | 110080 |

판정: 세 1m dataset 모두 Markov-1을 넘는 2차 구조 신호가 있다. Phase 3에서 고차 motif 검정을 유지한다.

## Q4. Interior-Only Sequence Audit

`corpus.py`의 Phase 2 `information_gain()`은 vocab 필터 후에도 원래 sequence position이 연속인 경우에만
bigram을 센다. 따라서 boundary row 제거로 실제로 인접하지 않았던 interior token이 새로 인접해지는
인공 adjacency는 생성하지 않는다. 아래 표의 `compressed_interior_ig_bits`는 의도적으로 interior만 압축했을 때의
참고값이며, Phase 2 계산값은 `true_adjacent_interior_ig_bits` 쪽이다.

| dataset | true_adjacent_interior_ig_bits | compressed_interior_ig_bits | true_adjacent_bigram_count | compressed_bigram_count | phase2_code_creates_artificial_adjacency |
| --- | --- | --- | --- | --- | --- |
| d1_kospi_daily | 0.3777 | 0.3017 | 1901 | 2378 | False |
| d1_kosdaq_daily | 0.4118 | 0.3277 | 1863 | 2362 | False |
| d1_nasdaq_daily | 0.3421 | 0.2902 | 2248 | 2604 | False |
| d1_spx_daily | 0.4942 | 0.3787 | 1376 | 2003 | False |
| d1_kospi_1m | 0.1859 | 0.0466 | 5014 | 16159 | False |
| d1_kosdaq_1m | 0.3653 | 0.0686 | 2390 | 11002 | False |
| d2_kr-kospi-kosdaq_daily | 0.1985 | 0.1631 | 3764 | 4740 | False |
| d2_kr-kospi-kosdaq_1m | 0.1188 | 0.0279 | 7404 | 27161 | False |

daily-only aggregate에서 interior observed IG가 surrogate 평균보다 낮았던 것은 adjacency bug가 아니라
유한표본에서 high-vocab shuffle surrogate가 양의 plug-in bias를 갖는 현상으로 해석한다. 특히 SPX daily는
interior bigram 표본 수가 적어 32-vocab transition 추정의 sparse-cell bias가 크다.

## Figures

- `figures/01_mi_diagonal_offdiagonal.png`: full-vocab MI의 대각/비대각 기여.
- `figures/02_boundary_run_lengths.png`: 주요 boundary token run-length와 iid geometric baseline 비교.
- `figures/03_bucket_jsd_and_ig.png`: bucket 간 token 분포 JSD와 bucket control 후 IG.
- `figures/04_second_order_markov1.png`: 2차 IG와 Markov-1 surrogate 비교.

## Phase 3 함의

1m motif는 raw token sequence만으로 BPE를 바로 적용하기 전에 RLE ablation과 time-bucket stratification을
반드시 포함해야 한다. Markov-1 surrogate를 통과하지 못하는 3-candle 이상 motif는 고차 패턴으로 해석하지 않고,
단순 transition model이 설명한 구조로 처리한다.

## Caveats

- 이 분석은 canonical seed-7 corpus 위의 통계 진단이며 tokenizer 재학습이나 label 사용은 없다.
- time bucket은 KR 정규장 기준이다. 해외 1m corpus가 추가되면 거래소별 session calendar가 필요하다.
- surrogate 판정은 100회 반복 기준이므로 경계적 z-score는 방향성 판단에만 사용한다.
