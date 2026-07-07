# 00. VQ-VAE Tokenizer 전제 검증 계획

상위 문서:
- [Candlestick Shape Quantization](README.md)

연계 문서:
- [01. 연구 목적과 범위](01-problem-and-research-scope.md)
- [02. Shape Tokenizer 설계](02-shape-tokenizer.md)
- [07. 근거와 참고 문헌](07-evidence-and-references.md)
- [08. 연구 로드맵](08-research-roadmap.md)

## 목적

이 문서는 VQ-VAE를 candle shape tokenizer로 사용하기 전에, VQ-VAE가 실제로 candle shape 공간을 안정적인 discrete vocabulary로 묶을 수 있는지 검증하는 사전 계획이다.

본 실험의 핵심 질문은 다음과 같다.

```text
가격 수준과 range 크기를 제거한 relative candle shape 공간에서
VQ-VAE codebook이 반복 가능한 candle shape prototype을 학습하는가?
```

이 질문에 답하기 전에는 `shape_token_t1`을 prediction target으로 사용하는 후속 모델의 의미가 약하다.

## 진행 현황 (2026-07-07 기준)

| 단계 | Notebook | 상태 | 결과 기록 |
| --- | --- | --- | --- |
| Phase 0 데이터 품질 | `00_data_protocol.ipynb` | 완료 | protocol JSON, step-01 run |
| Phase 1 shape 좌표 | `01_shape_feature_validation.ipynb` | 완료 | [results/phase-01-step-01-shape-feature-validation.md](results/phase-01-step-01-shape-feature-validation.md) |
| Phase 2 baseline clustering | `02_tokenizer_baselines.ipynb` | 완료 (D1 6개 + D2 KR 2개) | [results/phase-01-step-02-tokenizer-baselines.md](results/phase-01-step-02-tokenizer-baselines.md) |
| Phase 3 VQ 계열 검증 | `03_vq_tokenizer.ipynb` | 완료 | [results/phase-01-step-03-vq-final-gate.md](results/phase-01-step-03-vq-final-gate.md) |
| Phase 4 Baseline 대비 채택 판단 | 문서 판정 | 완료 | main tokenizer는 `kmeans_boundary_aware K=32`, `coarse_fine`은 Phase 3 motif ablation으로 보류 |

Phase 2까지의 핵심 결정: 주 baseline은 `kmeans K=32`(전 dataset·정책에서 최선, dead token 0), boundary는 비교군 B(경계 조합 8개 discrete token + interior-only 연속 codebook), 분봉은 minute split(train 2025-07~2026-01) 적용.

Phase 3/4 판정: `vqvae_latent_kmeans`, `fsq`, `bsq`는 KMeans-B 대비 이점을 보이지 못했다. `coarse_fine`은 reconstruction을 희생하지만 seed stability와 effective vocab을 개선했으므로 주 tokenizer로 채택하지 않고 motif 단계의 low-resolution sequence 후보로만 유지한다.

## 검증할 전제

### P1. 2D relative endpoint가 shape 정보를 충분히 담는다

기본 입력은 다음으로 고정한다.

```text
range = high - low
open_rel  = (open  - low) / range
close_rel = (close - low) / range
```

이 입력에서 리포트용 4D feature를 복원할 수 있어야 한다.

```text
signed_body_ratio = close_rel - open_rel
upper_ratio = 1 - max(open_rel, close_rel)
lower_ratio = min(open_rel, close_rel)
body_center_location = (open_rel + close_rel) / 2
```

검증 기준:
- `open_rel`, `close_rel`이 대부분 `[0, 1]` 범위에 있어야 한다.
- zero-range candle 비율을 별도 보고해야 한다.
- derived 4D feature가 기존 OHLC 정의와 일치해야 한다.

### P2. Shape 공간에 반복되는 prototype이 존재한다

모든 sample이 고유한 연속 shape처럼 분포한다면 discrete tokenization의 의미가 약하다.

검증 기준:
- k-means, GMM, hand-crafted bins가 해석 가능한 cluster/prototype을 만드는지 확인한다.
- `K = 8, 16, 32`(baseline 구현 기준)에서 reconstruction error와 token utilization을 비교한다.
- prototype 시각화에서 중복 token과 의미 없는 token이 많지 않아야 한다.

### P3. VQ-VAE가 단순 baseline보다 나은 이유가 있어야 한다

입력이 2D이므로 VQ-VAE는 과한 모델일 수 있다. 따라서 VQ-VAE는 단순 clustering baseline을 이겨야 채택할 수 있다.

비교 대상:

```text
1. k-means on (logit λ_o, logit λ_c)
2. GMM on (logit λ_o, logit λ_c)
3. hand-crafted lambda bins
4. VQ 계열 tokenizer
```

채택 조건:
- reconstruction error가 baseline보다 낮거나,
- held-out index transfer가 더 안정적이거나,
- seed별 prototype stability가 더 좋거나,
- downstream prediction 전 단계에서 token distribution이 더 해석 가능해야 한다.

### P4. Codebook collapse가 없어야 한다

VQ-VAE가 대부분의 sample을 소수 token에 몰면 tokenizer로 실패한 것이다.

필수 지표:

```text
token_count[k]
token_probability[k]
token_entropy = -sum(p_k * log(p_k))
effective_vocab_size = exp(token_entropy)
dead_token_count = count(token_count[k] < min_usage)
```

초기 판정 기준 ([08](08-research-roadmap.md)의 게이트와 동일):
- effective vocab size가 `K`의 1/4 이하이면 collapse 가능성이 높다.
- dead token이 반복적으로 발생하면 codebook 설정을 재검토한다.
- 참고: step-02 k-means baseline은 `K=32`에서 dead token 0, effective vocab 24~30이었다. VQ 계열이 이보다 나쁘면 구조적 문제다.
- collapse가 발생하면 먼저 `K`, initialization, EMA update, commitment weight, dead-code restart를 점검한다.
- Residual VQ는 collapse의 직접 해결책이 아니라 표현력 확장용 ablation으로만 둔다.

### P5. Token vocabulary가 시장 밖에서도 유지되어야 한다

공통 candle shape vocabulary를 주장하려면 특정 지수에만 맞는 token이어서는 안 된다.

검증 기준:
- held-out index에서 reconstruction error가 train index 대비 크게 악화되지 않아야 한다.
- held-out index에서도 token usage가 소수 token으로 붕괴하지 않아야 한다.
- index별 token distribution 차이는 Jensen-Shannon divergence로 보고한다.

## 실험 단계

### Phase 0. 데이터 품질 확인

입력:
- 세계 주요 주가 지수의 일봉 또는 분봉 OHLC 데이터
- 대상 index universe:
  ```text
  D1: KOSPI, KOSDAQ, NASDAQ, SPX, DJI, NDX, SOX
  D2: d2_kr-kospi-kosdaq_daily, d2_kr-kospi-kosdaq_1m, d2_us-nasdaq-spx-dji_daily
  D3: NASDAQ, SPX, DJI, KOSPI, KOSDAQ, N225, NDX, SOX, TWII
  ```
- 확보 현황(2026-07-06): D1은 daily 4개(KOSPI, KOSDAQ, NASDAQ, SPX)와 KR 1m 2개가 검증 대상으로 확정되었다. DJI는 endpoint 빈 응답으로 미확보, NDX/SOX는 protocol 미등록, 해외 1m(NASDAQ/SPX)은 KIS 제한(약 102 row)으로 제외 상태다. D2는 KR 묶음(daily, 1m)이 진행되었다.
- 해외지수 symbol은 KIS `download_overseas_index_info()` master 정보로 검증한다.
- train / validation / test를 시간 순서로 분리하고, split 밖의 row는 `excluded`로 기록한다. 일봉은 `split`(train 2005–2016), 분봉은 `split_minute`(train 2025-07-01~2026-01-31, 2026-07-06 확정)을 사용한다.
- held-out index split 별도 구성

주의:
- 일봉과 분봉은 같은 실험 run 안에서 섞지 않는다.
- 분봉을 사용할 경우 index별 거래 시간, 휴장일, session boundary를 정렬한 뒤 동일 주기 sample만 사용한다.
- API 호출용 symbol과 KIS master 검증용 `master_symbol`은 다를 수 있으므로 둘 다 run config에 기록한다. 예: NASDAQ은 chart API에서 `.IXIC`/`IXIC`, KIS master 검증에서 `COMP`를 사용한다.
- 이 검증은 order book, tick data, intraday execution 데이터를 사용하지 않는다.

산출물:

```text
row_count_by_index
date_range_by_index
missing_ohlc_count
invalid_ohlc_count
zero_range_count
zero_range_rate
```

필수 검증:
- `high >= max(open, close)` 위반 row를 제거 또는 별도 보고한다.
- `low <= min(open, close)` 위반 row를 제거 또는 별도 보고한다.
- `high - low <= eps` row는 tokenizer 학습에서 제외하고 개수를 보고한다.

### Phase 1. Shape 좌표 생성

생성 target:

```text
open_rel
close_rel
signed_body_ratio
upper_ratio
lower_ratio
body_center_location
```

Feature engineering 규칙:
- OHLC 결측, OHLC geometry 위반, zero-range candle을 순서대로 제거한다.
- tokenizer 입력은 winsorize(epsilon) 후 logit 변환한 shape core `(s1, s2) = (logit λ_o, logit λ_c)`다.
- endpoint 경계값(`<= eps` 또는 `>= 1 - eps`) row는 **데이터에서 제거하지 않는다**. 확정된 비교군 B 방식에 따라 boundary 조합 8개는 전용 discrete token으로 분리하고, interior × interior row만 연속 codebook 학습에 사용한다 ([08](08-research-roadmap.md)의 경계 캔들 처리 참조). 제외 방식은 ablation으로만 유지한다.
- boundary flag 개수와 비율을 별도 산출물에 기록한다.

필수 검증:
- `open_rel`, `close_rel` 범위 위반률
- `open_rel`, `close_rel` endpoint 경계값 제외 개수와 비율
- derived feature 관계식 위반률
- index별 shape coordinate 분포

### Phase 2. Baseline clustering

실험 (구현: `02_tokenizer_baselines.ipynb`, 완료):

```text
K = [8, 16, 32]
models = [k-means, GMM, hand-crafted lambda bins]
input = (s1, s2) = (logit λ_o, logit λ_c)
boundary policy = include_boundary / exclude_boundary A/B
seeds = 7, 17, 37
```

평가:
- reconstruction MSE on `(s1, s2)`
- cluster utilization (token entropy, effective vocab size, dead token count)
- prototype visualization (cluster shape atlas)
- seed stability

결과 요약: 8개 dataset(D1 daily 4, D1 KR 1m 2, D2 KR 2) 전부에서 `kmeans K=32`가 최선, dead token 0. 상세는 [results/phase-01-step-02-tokenizer-baselines.md](results/phase-01-step-02-tokenizer-baselines.md).

### Phase 3. VQ 계열 tokenizer 검증

실험 (구현: `03_vq_tokenizer.ipynb`, 완료):

```text
input = (s1, s2) = (logit λ_o, logit λ_c)
models = [
  kmeans_boundary_aware K=32,
  vqvae_latent_kmeans K=32,
  fsq levels=[6,5]          # continuous capacity 30
  bsq bits=5                # continuous capacity 32
  coarse_fine 8 x 4         # continuous capacity 32
]
boundary = 비교군 B (interior-only fit + boundary discrete token 8)
seeds = [7, 17, 37]
```

평가:
- reconstruction MSE
- token utilization
- effective vocab size
- dead token count
- D2 symbol별 token share
- seed stability

결과:
- reconstruction은 8개 dataset 전부에서 `kmeans_boundary_aware K=32`가 최선이다.
- `vqvae_latent_kmeans`, `fsq`, `bsq`는 seed stability와 token usage에서도 주 baseline을 이기지 못했다.
- `coarse_fine`은 seed stability와 effective vocab은 개선하지만 reconstruction MSE가 크게 악화된다.

### Phase 4. Baseline 대비 채택 판단

VQ-VAE를 tokenizer로 채택하려면 다음 중 최소 하나 이상을 보여야 한다.

```text
1. k-means/GMM 대비 reconstruction error 개선
2. held-out index에서 더 안정적인 reconstruction/token usage
3. seed 변경 후 prototype matching 안정성 개선
4. 더 해석 가능한 token prototype
```

반대로 다음 결과가 나오면 VQ-VAE 채택을 보류한다.

```text
1. VQ-VAE가 k-means보다 reconstruction/stability에서 낫지 않다.
2. effective vocab size가 낮다.
3. dead token이 반복적으로 발생한다.
4. prototype이 중복되거나 사람이 해석하기 어렵다.
5. held-out index에서 token usage가 붕괴한다.
```

판정(2026-07-07):

```text
main tokenizer:
  kmeans_boundary_aware K=32

do not tune further in Phase 1:
  vqvae_latent_kmeans
  fsq(levels=[6,5])
  bsq(bits=5)

carry to motif-stage ablation only:
  coarse_fine(coarse 8 x fine 4)
```

이 결정은 VQ 계열이 연구 실패라는 뜻이 아니라, Phase 1의 단일 candle shape reconstruction 목적에서는 단순 KMeans-B가 더 강한 baseline이라는 뜻이다.

## 평가 지표

### Reconstruction

```text
shape_core_mse = MSE((s1, s2), (s1_hat, s2_hat))   # 주 지표, 구현 기준
endpoint_mae = MAE([open_rel, close_rel], [open_rel_hat, close_rel_hat])  # lambda 공간 보조 지표

derived_feature_mae:
  signed_body_ratio
  upper_ratio
  lower_ratio
  body_center_location
```

### Utilization

```text
token_count[k]
token_probability[k]
token_entropy
effective_vocab_size
dead_token_count
max_token_share
```

### Stability

```text
Adjusted Rand Index across seeds
Normalized Mutual Information across seeds
prototype distance after matching
Jensen-Shannon divergence by index
held-out index reconstruction error
```

### Interpretability

각 token prototype에 대해 다음을 저장한다.

```text
prototype.open_rel
prototype.close_rel
prototype.signed_body_ratio
prototype.upper_ratio
prototype.lower_ratio
prototype.body_center_location
prototype.direction
prototype.sample_count
```

또한 token별 대표 sample과 prototype candle plot을 함께 저장한다.

## Acceptance Criteria

VQ-VAE tokenizer 사전 검증은 다음 조건을 만족할 때 통과로 본다.

```text
1. zero-range 및 invalid OHLC 처리 정책이 train/validation/test에 일관되게 적용된다.
2. [open_rel, close_rel]에서 derived 4D feature가 수식대로 복원된다.
3. K 후보별 baseline clustering 결과가 기록된다.
4. K 후보별 VQ-VAE tokenizer 결과가 기록된다.
5. token utilization과 collapse 지표가 모델별로 비교된다.
6. held-out index reconstruction과 token usage가 보고된다.
7. seed stability가 최소 3개 seed 이상에서 측정된다.
8. VQ-VAE 채택 또는 보류 판단이 baseline 대비 근거로 결정된다.
```

## 후속 의사결정

### VQ-VAE 채택

조건:
- baseline 대비 명확한 이점이 있고,
- codebook collapse가 없으며,
- held-out index에서도 token vocabulary가 유지된다.

후속 작업:
- [08. 연구 로드맵](08-research-roadmap.md)의 Phase 2(token sequence corpus 생성)로 진행한다.

### 단순 clustering 채택

조건:
- k-means/GMM이 VQ-VAE와 비슷하거나 더 안정적이다.

후속 작업:
- `shape_token`을 VQ-VAE code가 아니라 baseline cluster id로 정의한다.
- 후속 phase는 동일하게 진행하되 tokenizer complexity를 줄인다.

### Tokenizer 가설 보류

조건:
- 어떤 방식도 안정적인 token vocabulary를 만들지 못한다.

후속 작업:
- discrete shape token 대신 continuous shape feature를 그대로 사용하는 경로(TS2Vec 계열 continuous representation, [08](08-research-roadmap.md) Phase 5의 baseline 구성 참조)를 우선 검토한다.

## Residual VQ의 위치

Residual VQ는 단일 codebook의 quantization error가 큰 경우에 검토할 수 있는 확장 실험이다.

```text
z
  -> codebook_1
  -> residual_1
  -> codebook_2
  -> residual_2
  -> ...
```

다만 이 계획에서는 Residual VQ를 기본 collapse 해결책으로 보지 않는다.

이유:
- collapse 원인이 loss imbalance, initialization, commitment weight, EMA update라면 Residual VQ만으로 해결되지 않는다.
- 앞단 codebook이 대부분을 설명하면 뒤쪽 residual codebook이 죽을 수 있다.
- token이 단일 `shape_token`이 아니라 multi-level token tuple이 되어 prediction target이 복잡해진다.
- 현재 입력은 2D이므로 단일 codebook과 baseline clustering을 먼저 검증하는 편이 더 단순하고 방어 가능하다.

따라서 Residual VQ는 다음 조건에서만 ablation으로 추가한다.

```text
1. 단일 VQ-VAE가 collapse하지 않는다.
2. 하지만 reconstruction error가 baseline 대비 충분히 낮지 않다.
3. K를 키우면 utilization이 나빠진다.
4. coarse-to-fine token이 prediction 또는 reconstruction에 유리한지 확인할 필요가 있다.
```

Residual VQ를 실험할 경우 level별로 다음을 따로 보고한다.

```text
level_token_count
level_effective_vocab_size
level_dead_token_count
residual_norm_by_level
reconstruction_gain_by_level
```

## 최종 산출물

```text
tokenizer_validation_report.md
prototype_table.csv
token_usage_by_index.csv
reconstruction_error_by_model.csv
seed_stability_summary.csv
prototype_candle_plots/
```

최종 보고서는 다음 결론 중 하나를 명시해야 한다.

```text
1. VQ-VAE tokenizer를 채택한다.
2. 단순 clustering tokenizer를 채택한다.
3. discrete tokenizer 가설을 보류한다.
```
