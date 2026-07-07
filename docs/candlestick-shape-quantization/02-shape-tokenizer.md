# 02. Shape Tokenizer 설계

상위 문서:
- [Candlestick Shape Quantization](README.md)

연계 문서:
- [01. 연구 목적과 범위](01-problem-and-research-scope.md)
- [07. 근거와 참고 문헌](07-evidence-and-references.md)
- [08. 연구 로드맵](08-research-roadmap.md)

## 역할

Shape Tokenizer는 OHLC 캔들을 가격 수준과 range 크기에서 분리하고, 내부 모양만 discrete token으로 압축한다.

```text
OHLC
  -> relative candle shape
  -> discrete shape token
```

이 token은 motif 학습과 downstream 평가의 기본 단위이며, 시각화 시 prototype으로 shape를 복원하는 데 사용된다.

## 기존 4D shape feature

초기 설계의 shape feature는 다음과 같다.

```text
signed_body_ratio = (close - open) / (high - low)
upper_ratio = (high - max(open, close)) / (high - low)
lower_ratio = (min(open, close) - low) / (high - low)
body_center_location = ((open + close) / 2 - low) / (high - low)
```

장점:
- 캔들 해석과 직관적으로 연결된다.
- body direction, upper wick, lower wick, body location을 명시적으로 볼 수 있다.
- 분석 리포트와 시각화에 설명하기 쉽다.

단점:
- 네 값이 독립적이지 않다.
- 중복 좌표 때문에 VQ distance가 특정 성분을 과가중할 수 있다.
- 모델 입장에서는 같은 정보를 여러 방식으로 반복해서 받는다.

## 피처 중복성

다음 관계가 항상 성립한다.

```text
abs_body = abs(signed_body_ratio)
upper_ratio + lower_ratio + abs_body = 1
body_center_location = lower_ratio + abs_body / 2
```

즉 4D feature는 설명용으로는 좋지만 tokenizer 입력으로는 비효율적이다.

## 권장 입력: 2D relative endpoint

Tokenizer 입력은 아래 2D를 기본값으로 둔다.

```text
range = high - low
open_rel  = (open  - low) / range
close_rel = (close - low) / range
```

이 표현은 다음 조건을 만족한다.

```text
0 <= open_rel <= 1
0 <= close_rel <= 1
```

기존 4D feature는 모두 복원 가능하다.

```text
signed_body_ratio = close_rel - open_rel
upper_ratio = 1 - max(open_rel, close_rel)
lower_ratio = min(open_rel, close_rel)
body_center_location = (open_rel + close_rel) / 2
```

따라서 tokenizer 학습 입력은 `[open_rel, close_rel]`로 두고, 분석/리포트에서는 4D derived feature를 함께 출력하는 구성이 가장 명확하다.

추가로 [08](08-research-roadmap.md)의 Feature 설계에 따라, `[0,1]` 경계 제약을 제거한 logit 변환 버전 `(logit λ_o, logit λ_c)`을 기본 입력으로 하고 raw 버전은 ablation으로 비교한다.

## 피처 엔지니어링 필터 (boundary 처리)

경계 판정 기준:

```text
open_rel  <= eps or open_rel  >= 1 - eps
close_rel <= eps or close_rel >= 1 - eps
```

경계 캔들(marubozu류)은 실제로 반복되는 의미 있는 shape이며, 특히 분봉에서는 전체의 73~81%를 차지하므로 데이터에서 제거하지 않는다.

확정 정책(2026-07-06, 비교군 B — [08](08-research-roadmap.md)의 경계 캔들 처리 참조):

```text
lambda_o × lambda_c의 {low boundary, interior, high boundary} 9개 조합 중
  interior × interior     -> 연속 codebook 학습 대상
  boundary 조합 8개        -> 전용 discrete token
전체 vocabulary = 연속 codebook K + boundary token 8 + zero-range special token 1
```

이 분리는 boundary point mass가 연속 codebook의 용량을 잠식하는 것(step-02 baseline에서 분봉 include 시 effective vocab 24~26으로 하락, prototype 다수가 boundary에 배치됨)을 막으면서도 모든 캔들에 token을 부여한다. winsorize 포함(비교군 A)은 boundary 점유율 측정용 기준으로, 완전 제외는 interior-only fit의 ablation으로 유지하며, boundary 개수와 비율은 실험 산출물에 반드시 기록한다.

## 예외 처리

`high == low`인 zero-range candle은 별도 처리가 필요하다.

확정 처리:

```text
if high - low <= eps:
    mark as zero_range
    exclude from tokenizer fit/evaluation
    assign special token
```

zero-range candle은 lambda가 정의되지 않으므로 codebook fit에서 제외하고 special token을 부여하며, 개수와 비율을 보고한다. 실측(step-01)에서 daily는 0~0.16%, 국내 1m은 약 0.26% 수준이었다.

## VQ-VAE Tokenizer

VQ-VAE tokenizer는 continuous input을 encoder로 latent vector로 바꾼 뒤, 가장 가까운 codebook vector에 할당해 discrete token을 만든다.

```text
(logit λ_o, logit λ_c)
  -> encoder
  -> z_e
  -> nearest codebook vector
  -> shape_token
  -> decoder
  -> reconstructed (logit λ_o, logit λ_c)
```

권장 codebook 크기:

```text
K candidates = [8, 12, 16, 24, 32]   # 로드맵 Phase 1 K sweep
baseline 구현 sweep = [8, 16, 32]    # 02_tokenizer_baselines.ipynb
```

step-02 baseline 결과 전 dataset에서 `kmeans K=32`가 reconstruction 기준 최선이었고 dead token 0, effective vocab 24~30이었다. step-03 final gate(`cfg-d59bafed`)에서는 같은 boundary-aware wrapper 아래 `vqvae_latent_kmeans`, `fsq`, `bsq`, `coarse_fine`을 비교했다. 신경망 후보 3개는 reconstruction, seed stability, token usage에서 KMeans-B를 이기지 못했고, `coarse_fine`만 reconstruction을 크게 희생하면서 seed stability와 effective vocab을 개선했다.

따라서 shape reconstruction용 주 tokenizer는 `kmeans_boundary_aware K=32`로 둔다. `coarse_fine`은 최종 shape tokenizer가 아니라 Phase 3 motif 단계의 low-resolution sequence ablation으로만 유지한다.

## 반드시 필요한 baseline

입력이 2D 또는 4D로 매우 저차원이므로 VQ-VAE만 쓰면 복잡도가 과할 수 있다. 다음 baseline과 비교해야 한다.

```text
1. k-means on (logit λ_o, logit λ_c)
2. GMM on (logit λ_o, logit λ_c)
3. hand-crafted lambda bins
4. VQ 계열 tokenizer
```

VQ-VAE를 채택하려면 다음 중 하나 이상의 이점이 필요하다.

- lower reconstruction error
- better held-out index transfer
- more stable token prototypes across seeds
- downstream prediction 성능 개선
- token distribution의 해석 가능성 개선

## Token Prototype 정의

각 token의 prototype은 codebook vector 또는 해당 token에 할당된 실제 샘플의 평균으로 정의할 수 있다.

권장:

```text
prototype.open_rel  = mean(open_rel  | token = k)
prototype.close_rel = mean(close_rel | token = k)
```

그리고 derived feature를 함께 저장한다.

```text
prototype.signed_body_ratio
prototype.upper_ratio
prototype.lower_ratio
prototype.body_center_location
prototype.direction
```

이 방식은 OHLC 복원 시 codebook latent보다 직접 해석 가능한 shape 좌표를 사용하게 해준다.

## Tokenizer 출력 산출물

Tokenizer 학습 후 저장해야 할 산출물:

```text
shape_token_t
prototype_table
token_usage_by_index
token_usage_by_regime
reconstruction_error_by_index
dead_token_count
effective_vocab_size
```

## Codebook Collapse 감시

VQ 계열 모델은 일부 code만 사용하고 나머지는 죽는 codebook collapse 문제가 생길 수 있다.

따라서 다음 지표를 필수로 기록한다.

```text
token_count[k]
token_probability[k]
token_entropy = -sum(p_k * log(p_k))
effective_vocab_size = exp(token_entropy)
dead_token_count = count(token_count[k] < min_usage)
```

해석 기준 ([08](08-research-roadmap.md)의 게이트와 동일):

```text
effective_vocab_size가 K의 1/4 이하라면 vocabulary가 충분히 쓰이지 않는 것이다.
dead token이 반복적으로 발생하면 codebook 설정이나 학습 안정성을 재검토한다.
참고: step-02 k-means baseline은 K=32에서 dead token 0, effective vocab 24~30.
```

## 최종 권장안

연구 설계의 기본 tokenizer는 다음으로 고정한다.

```text
input:
  shape core (s1, s2) = (logit λ_o, logit λ_c)  # winsorize + logit

model:
  kmeans_boundary_aware K=32

baselines:
  GMM, hand-crafted lambda bins, VQ-VAE latent KMeans, FSQ, BSQ, coarse-fine

boundary:
  비교군 B — interior-only codebook fit + boundary discrete token 8 + zero-range special token

primary output:
  shape_token

analysis output:
  token prototype table (cluster shape atlas)
  derived 4D shape feature
  token stability metrics
```

Step-03 final gate 판정:

```text
main tokenizer:
  kmeans_boundary_aware K=32

rejected as main shape tokenizer:
  vqvae_latent_kmeans
  fsq(levels=[6,5])
  bsq(bits=5)

carry-forward ablation:
  coarse_fine(coarse 8 x fine 4)  # Phase 3 motif에서 저해상도 sequence로만 재평가
```
