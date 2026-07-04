# Candlestick Shape Quantization

## 문서 역할

이 문서는 주요 지수의 OHLC 캔들을 가격 수준이 아닌 상대적 price-shape의 discrete representation으로 바꾸는 연구 설계의 상위 문서이다.

목표는 다음 질문을 검증 가능한 형태로 바꾸는 것이다.

> 여러 지수에서 반복되는 candle price-shape를 같은 discrete token vocabulary로 안정적으로 묶을 수 있는가?

## 문서 관계

상위 문서:
- [Documentation](../README.md)

하위 문서:
- [00. VQ-VAE Tokenizer 전제 검증 계획](00-vq-vae-tokenizer-validation-plan.md)
- [01. 연구 목적과 범위](01-problem-and-research-scope.md)
- [02. Shape Tokenizer 설계](02-shape-tokenizer.md)
- [07. 근거와 참고 문헌](07-evidence-and-references.md)
- [08. 연구 로드맵](08-research-roadmap.md)

권장 읽기 순서:

```text
README
  -> 01-problem-and-research-scope
  -> 08-research-roadmap
  -> 02-shape-tokenizer
  -> 00-vq-vae-tokenizer-validation-plan
  -> 07-evidence-and-references
```

## 연구 구조

전체 연구는 [08. 연구 로드맵](08-research-roadmap.md)의 Phase 0-6을 따른다.

```text
Phase 0: 데이터와 leakage 프로토콜
Phase 1: 단일 candle shape tokenizer        <- 00, 02 문서가 상세 설계
Phase 2: token sequence corpus와 label
Phase 3: 다중 candle motif vocabulary
Phase 4: 해석성 평가
Phase 5: 예측 유용성 평가
Phase 6: 결론 정리
```

Tokenizer 검증의 순수 구조:

```text
Tokenizer Validation:
  input  = shape core (logit λ_o, logit λ_c)
  compare = k-means / GMM / hand-crafted bins / VQ-VAE / FSQ·LFQ
  decision = adopt VQ 계열, adopt simpler clustering, or defer discrete tokenization

Pure Shape Tokenizer:
  input  = shape core
  output = shape_token
```

## 설계상 중요한 주의점

직관적인 4개 shape feature(body, upper, lower, center)는 독립 피처가 아니다.

```text
abs_body = abs(signed_body_ratio)
upper_ratio + lower_ratio + abs_body = 1
body_center_location = lower_ratio + abs_body / 2
```

따라서 tokenizer 입력은 아래 2D 표현을 기본으로 하고, logit 변환으로 경계 제약을 제거한다([08](08-research-roadmap.md)의 Feature 설계 참조).

```text
λ_o = (open  - low) / (high - low)
λ_c = (close - low) / (high - low)
shape core = (logit λ_o, logit λ_c)
```

이 두 값에서 4개 shape feature를 모두 복원할 수 있다.

## 연구 성공 기준

이 연구가 성공했다고 말하려면 최소한 다음을 보여야 한다.

1. Shape token이 OHLC shape의 의미 있는 압축 표현으로 동작한다.
2. Token vocabulary가 특정 지수에 과적합되지 않고 held-out index에서도 유사하게 사용된다.
3. Token 사용률이 한두 개 token에 붕괴하지 않는다.
4. 단순 clustering baseline 대비 VQ 계열 tokenizer의 이점 여부가 근거와 함께 판정된다.
5. Token/motif representation의 downstream 유용성이 baseline feature와 walk-forward로 비교된다.
