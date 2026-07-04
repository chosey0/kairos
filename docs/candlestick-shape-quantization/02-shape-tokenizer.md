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

## 피처 엔지니어링 필터

Tokenizer 학습/평가에는 내부 endpoint shape만 사용한다. 즉 open 또는 close가 candle의 high/low와 정확히 같은 극단 endpoint에 붙은 row는 제외한다.

제외 기준:

```text
open_rel  <= eps or open_rel  >= 1 - eps
close_rel <= eps or close_rel >= 1 - eps
```

이 필터는 wick endpoint artifact가 codebook을 지배하는 것을 줄이고, VQ-VAE가 내부 body 위치와 wick balance를 더 안정적으로 학습하게 하기 위한 것이다. 제외 개수와 비율은 실험 산출물에 반드시 기록한다.

주의: 경계 캔들(marubozu류)은 실제로 반복되는 의미 있는 shape이므로, 제외 방식(B안)은 winsorize + boundary flag 방식(A안)과 비교 검증한다. [08](08-research-roadmap.md)의 경계 캔들 처리 참조.

## 예외 처리

`high == low`인 zero-range candle은 별도 처리가 필요하다.

권장 처리:

```text
if high - low <= eps:
    mark as zero_range
    exclude from tokenizer training
    or assign special token
```

연구 초기에는 zero-range candle을 tokenizer 학습에서 제외하고 개수를 보고하는 편이 안전하다. 주요 지수 daily candle에서는 빈도가 낮을 가능성이 높지만, 데이터 품질 문제와 휴장/비정상 row를 확인하는 용도로 유용하다.

## VQ-VAE Tokenizer

VQ-VAE tokenizer는 continuous input을 encoder로 latent vector로 바꾼 뒤, 가장 가까운 codebook vector에 할당해 discrete token을 만든다.

```text
[open_rel, close_rel]
  -> encoder
  -> z_e
  -> nearest codebook vector
  -> shape_token
  -> decoder
  -> reconstructed [open_rel, close_rel]
```

권장 codebook 크기:

```text
K candidates = [8, 12, 16, 24]
initial K = 12
```

12-class는 해석과 복잡도의 균형점으로 사용할 수 있지만, 최종 결정은 validation 결과로 해야 한다.

## 반드시 필요한 baseline

입력이 2D 또는 4D로 매우 저차원이므로 VQ-VAE만 쓰면 복잡도가 과할 수 있다. 다음 baseline과 비교해야 한다.

```text
1. k-means on [open_rel, close_rel]
2. GMM on [open_rel, close_rel]
3. hand-crafted candle bins
4. VQ-VAE tokenizer
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

해석 기준 예시:

```text
K = 12인데 effective_vocab_size가 3 이하라면 vocabulary가 충분히 쓰이지 않는 것이다.
K = 12인데 2개 이상 token이 거의 사용되지 않으면 codebook 설정이나 학습 안정성을 재검토한다.
```

## 최종 권장안

연구 설계의 기본 tokenizer는 다음으로 고정한다.

```text
input:
  [open_rel, close_rel]

model:
  VQ-VAE tokenizer

baselines:
  k-means, GMM, hand-crafted bins

primary output:
  shape_token

analysis output:
  token prototype table
  derived 4D shape feature
  token stability metrics
```

