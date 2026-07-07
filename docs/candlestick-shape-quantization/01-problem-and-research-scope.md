# 01. 연구 목적과 범위

상위 문서:
- [Candlestick Shape Quantization](README.md)

연계 문서:
- [02. Shape Tokenizer 설계](02-shape-tokenizer.md)
- [08. 연구 로드맵](08-research-roadmap.md)

## 목적

이 연구의 목적은 주요 지수의 OHLC 캔들을 예측 target 자체가 아니라 학습 가능한 discrete representation으로 재구성할 수 있는지 검증하는 것이다.

핵심 질문은 다음과 같다.

```text
절대 가격, 변동폭, 시장 단위가 달라도 캔들 내부 모양이 비슷하면
같은 shape token으로 묶을 수 있는가?
```

예를 들어 KOSPI의 특정 캔들과 NASDAQ의 특정 캔들이 가격 수준이나 변동폭은 다르더라도 내부 모양이 비슷하다면 같은 token으로 매핑되는지를 본다.

## 연구 가설

### H1. 상대 shape는 가격 수준과 분리될 수 있다

OHLC를 가격 자체로 보지 않고 `high-low range` 안의 상대 좌표로 보면, 서로 다른 지수의 캔들도 같은 shape 공간에 놓을 수 있다.

검증 방법:
- index별 price scale을 제거한 shape feature를 생성한다.
- held-out index에서 tokenizer reconstruction error와 token usage를 확인한다.

### H2. 반복되는 shape vocabulary가 존재한다

모든 캔들이 완전히 연속적인 고유 shape라면 discrete tokenization의 의미가 약하다. 반대로 특정 형태가 반복된다면 token vocabulary로 압축할 수 있다.

검증 방법:
- token 수 `K`를 바꿔 reconstruction error와 token utilization을 비교한다.
- k-means, GMM, hand-crafted candle bins와 비교한다.

### H3. Shape token은 position/range와 분리할 수 있다

캔들의 내부 형태와 캔들이 놓이는 위치, 크기는 서로 다른 정보 축으로 취급할 수 있다.

검증 방법:
- shape token은 `rel_range`, `gap` 같은 continuous side-channel과 분리해 정의한다.
- downstream 평가에서 채널 조합별 기여를 분리해 측정한다([08](08-research-roadmap.md) Phase 5).

### H4. Token sequence에 반복되는 multi-candle motif가 존재한다

단일 token을 넘어 여러 캔들에 걸친 motif가 우연 수준을 넘어 반복되고, downstream 신호를 가질 수 있다.

검증 방법:
- surrogate sequence 대비 motif 초과 빈도 검정([08](08-research-roadmap.md) Phase 3)
- motif feature의 downstream 예측 기여([08](08-research-roadmap.md) Phase 5)

## 연구 범위

포함 범위:
- 주요 주가 지수의 일봉 또는 분봉 OHLC 데이터
  - 단계적 확장: 단일 지수 -> 국가별 주요 지수 -> 세계 주요 지수 ([08](08-research-roadmap.md) Phase 0 참조)
  - 기본 index universe: `KOSPI`, `KOSDAQ`, `NASDAQ`, `SPX`, `DJI`, `N225`, `NDX`, `SOX`, `TWII`
  - D1 기본 대상: `KOSPI`, `KOSDAQ`, `NASDAQ`, `SPX`, `DJI`, `NDX`, `SOX`의 일봉/분봉
  - 해외지수는 KIS `download_overseas_index_info()` master 정보로 검증하며, master 검증 전까지 `NYSE`, `AMEX`, `RUT`, `VIX`는 기본 universe에 포함하지 않는다.
- OHLC 기반 shape tokenization
- token sequence 기반 multi-candle motif vocabulary 학습
- shape token과 continuous side-channel(rel_range, gap)의 분리 평가
- motif의 해석성 평가와 downstream 예측 유용성 평가

제외 범위:
- 거래량 기반 tokenization
- 개별 종목의 microstructure 분석
- 실거래 전략 또는 수익률 최적화
- order book, tick data, intraday execution
- 예측 결과를 바로 매매 신호로 사용하는 운영 전략

## 설계 원칙

1. Shape token은 순수해야 한다.
   - 가격 수준, range 크기, 전일 대비 위치 이동을 token에 섞지 않는다.

2. Position과 range는 연속 side-channel로 둔다.
   - discrete transition matrix로 모든 정보를 설명하려 하지 않는다.

3. 모든 모델은 시간 기준 split 확정 후에만 fit한다.
   - tokenizer, motif vocabulary, 정규화 통계는 train 구간에서만 학습한다.

4. 평가 지표는 다음 축으로 분리해 보고한다.
   - tokenizer reconstruction error
   - token utilization과 stability
   - motif 유의성(surrogate 대비)
   - downstream 예측 유용성(walk-forward)

## 성공 기준

최소 성공 기준:

```text
1. Tokenizer가 held-out index에서 안정적인 reconstruction error를 보인다.
2. Codebook이 collapse하지 않고 충분한 token을 사용한다.
3. Token distribution이 특정 시장에만 과도하게 치우치지 않는다.
4. Motif가 surrogate 대비 우연 수준을 넘어 반복된다.
5. Token/motif representation이 baseline feature 대비 downstream에서 유의한 개선 또는
   동등 성능 + 해석성 이점을 보인다.
```

## 실패로 해석해야 하는 경우

다음 결과가 나오면 shape tokenization 가설을 약하게 보아야 한다.

- 대부분의 샘플이 소수 token에 몰린다.
- held-out index에서 reconstruction error가 크게 증가한다.
- VQ-VAE가 k-means 대비 뚜렷한 이점을 보이지 못한다.
- token sequence의 순차 구조가 없어 motif가 surrogate와 구분되지 않는다.
- token/motif feature가 어떤 downstream task에서도 raw feature 대비 정보를 더하지 못한다.
