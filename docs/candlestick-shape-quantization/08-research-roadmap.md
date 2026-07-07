# 08. 연구 로드맵: Candle Token에서 Motif Vocabulary까지

상위 문서:
- [Candlestick Shape Quantization](README.md)

연계 문서:
- [00. VQ-VAE Tokenizer 전제 검증 계획](00-vq-vae-tokenizer-validation-plan.md)
- [01. 연구 목적과 범위](01-problem-and-research-scope.md)
- [02. Shape Tokenizer 설계](02-shape-tokenizer.md)
- [07. 근거와 참고 문헌](07-evidence-and-references.md)

## 문서 역할

이 문서는 단일 candle tokenizer(00-02 문서 범위)를 다중 캔들 motif vocabulary 연구로 확장하는 전체 로드맵이다. `references/`의 논문을 근거로 초기 계획에서 개선한 지점을 명시하고, 각 단계를 결정 게이트가 있는 실행 계획으로 구체화한다.

## Reference 논문 매핑

| 파일 | 논문 | 이 연구에 주는 근거 |
| --- | --- | --- |
| 2103.16908 | Dimension reduction of OHLC data based on pseudo-PCA | OHLC의 제약을 제거하는 bijective feature 표현 `(ln low, ln range, logit λ_o, logit λ_c)` |
| 2508.02739 | Kronos: A Foundation Model for the Language of Financial Markets | K-line 전용 tokenizer(BSQ, coarse/fine 계층 토큰), instance 단위 z-score + clip 전처리, autoregressive candle-LM, 공개 pretrained baseline |
| 2303.04743 | TimeVQVAE | VQ 기반 시계열 생성, low/high frequency 분리 codebook, 생성 품질 평가 지표 |
| 2106.10466 | TS2Vec | quantization 이전 continuous representation의 강한 baseline |
| 1901.05237 | Encoding Candlesticks as Images (GAF-CNN) | 8개 표준 캔들 패턴 분류 체계와 라벨링 방법. 해석성 평가의 ground truth |
| 2001.02767 | Explainable Deep Convolutional Candlestick Learner | perturbation 기반으로 모델이 보는 shape 요소를 검증하는 방법론 |
| 2201.08669 | Dynamic Deep Convolutional Candlestick Learner | 고정 window 분류가 아닌 variable-length 패턴 위치 탐지 관점 |
| 1903.12258 | DL + Candlestick Chart Representation | 캔들 이미지 CNN 예측 baseline, window 길이/volume 포함 여부 ablation 설계 |
| 1709.03803 | Deep Stock Representation Learning | 비지도 캔들 representation을 downstream 의사결정으로 연결하는 평가 구조 |

폴더 외 참고: VQ-VAE 원논문(1711.00937), TSAX(1905.00421), VQ-AR(2205.15894)은 [07](07-evidence-and-references.md)의 근거 체계에 준해 인용한다.

주의: `2508.02739`는 VLM candlestick 벤치마크가 아니라 Kronos 논문이다. 문헌 목록 관리 시 혼동하지 않는다.

## 초기 계획 대비 주요 개선점

| # | 초기 계획 | 개선 | 근거 |
| --- | --- | --- | --- |
| 1 | 4D feature(body, upper, lower, center) | 2D core `(logit λ_o, logit λ_c)` + 별도 채널. 4D는 파생 리포팅 지표로만 사용 | 2103.16908, [02](02-shape-tokenizer.md) |
| 2 | 경계 캔들(marubozu류) 제외 | 제외 대신 winsorize + boundary flag를 기본으로 두고 제외 방식과 비교 | vocabulary 편향 방지 |
| 3 | tokenizer 후보 k-means/GMM/SOM/VQ-VAE | FSQ/LFQ(BSQ) 추가, coarse/fine 계층 토큰 변형 추가. SOM은 optional로 격하 | 2508.02739, 2303.04743 |
| 4 | dataset 생성이 tokenizer 이후(2단계) | split/leakage 프로토콜을 Phase 0으로 선행. tokenizer는 train split에서만 fit | 시계열 leakage 방지 원칙 ([07](07-evidence-and-references.md)) |
| 5 | motif 학습이 segmentation 계열만 | autoregressive candle-LM(Kronos 방식)을 motif 경로에 추가하고 명시적 vocabulary와 비교 | 2508.02739 |
| 6 | motif 유의성 기준 없음 | Markov-1 surrogate 대비 likelihood-ratio 검정으로 우연 초과 여부를 판정 | 방법론 보강 |
| 7 | baseline에 사전학습 모델 없음 | Kronos 공개 pretrained 모델을 zero-shot baseline으로 추가 | 2508.02739 |
| 8 | continuous representation 비교 없음 | TS2Vec embedding을 non-quantized baseline으로 추가해 양자화 정보 손실을 측정 | 2106.10466 |
| 9 | 평가가 전체 평균 중심 | regime별, timeframe별(daily/weekly) 분리 평가를 필수화 | 2508.02739, 1903.12258 |
| 10 | 전통 패턴 비교 방법 미정 | GAF-CNN 8-pattern 규칙 라벨러를 구현해 token-pattern overlap을 정량화 | 1901.05237 |

## Feature 설계 (개선안)

### 표기

```text
range = high - low
λ_o = (open  - low) / range   # open의 상대 위치
λ_c = (close - low) / range   # close의 상대 위치
```

### Shape core (tokenizer 입력)

```text
s1 = logit(clip(λ_o, eps, 1 - eps))
s2 = logit(clip(λ_c, eps, 1 - eps))
eps = 1e-3 (sensitivity: 1e-2, 1e-4)
```

pseudo-PCA(2103.16908)의 feature-based representation에서 shape 성분만 취한 것이다. `[0,1]` 경계 제약이 사라져 k-means/GMM/VQ의 Euclidean distance 가정과 정합하고, `(low, range)`가 주어지면 OHLC로 완전 복원된다.

초기 계획의 4D feature는 모두 파생 가능하므로 입력이 아닌 분석/시각화 지표로 사용한다.

```text
signed_body_ratio    = λ_c - λ_o
upper_ratio          = 1 - max(λ_o, λ_c)
lower_ratio          = min(λ_o, λ_c)
body_center_location = (λ_o + λ_c) / 2
```

### 별도 채널 (shape token에 섞지 않는다)

```text
rel_range = ln(range_t / ATR_14(t-1))          # 상대 크기
gap       = (open_t - close_{t-1}) / ATR_14(t-1)  # 갭
vol_spike = ln(volume_t / median(volume, 20))  # optional, ablation 전용
```

ATR/median은 t-1까지의 정보만 사용한다. 방향은 `sign(λ_c - λ_o)`로 shape core에 이미 내재하므로 별도 채널을 만들지 않는다. volume은 [01](01-problem-and-research-scope.md)의 제외 범위이며 지수별 품질 편차가 크므로(VIX는 거래량 자체가 없음) core에 넣지 않는다.

### 경계 캔들 처리

open/close가 high/low에 정확히 붙은 캔들(marubozu, open=low 등)은 실제로 반복되는 의미 있는 shape다. [02](02-shape-tokenizer.md)의 제외 방침을 다음과 같이 수정 검토한다.

```text
A안: winsorized logit + boundary flag 4bit (open==low, open==high, close==low, close==high)
B안(비교): 기존대로 제외
판정: held-out reconstruction, token entropy, 전통 패턴 overlap에서 A/B 비교
```

확정(2026-07-06, boundary 분리 방식): boundary candle은 데이터에서 제거하지 않고 이산 구조로 분리한다.

```text
lambda_o × lambda_c의 {low boundary, interior, high boundary} 9개 조합 중
  interior × interior          -> (s1, s2) 연속 codebook 학습 대상
  나머지 boundary 조합 8개      -> 전용 discrete token
전체 vocabulary = 연속 codebook K + boundary token 8 + zero-range special token 1
```

근거: step-02 baseline에서 boundary의 비용이 reconstruction error가 아니라 codebook 용량 잠식(분봉 include 시 effective vocab 24~26, prototype 다수가 boundary에 배치)으로 확인되었다. winsorize 포함(A안)은 boundary 점유율 측정용 비교 기준으로, 제외(B안)는 interior-only fit의 ablation으로 유지한다. 상세 기록: `notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-01-shape-feature-validation/FIGURE_EXPLANATION.md`의 Next Step Decision.

zero-range candle(`high == low`)은 기존대로 special token 처리하고 빈도를 보고한다.

## Phase 0. 데이터와 Leakage 프로토콜

목표: 모든 후속 단계가 공유하는 데이터 규약을 확정한다.

### 단계적 데이터 확장

데이터는 3단계로 확장하며, 각 단계의 결과가 다음 단계의 진입 조건이 된다.

```text
Stage D1. 단일 지수
  주가 지수 1개를 선택해 단독 데이터셋으로 사용
  기본 universe: KOSPI, KOSDAQ, NASDAQ, SPX, DJI, NDX, SOX
  interval: 1d, 1m
  목적: 파이프라인 검증, feature/tokenizer 설계 확정, 초기 K sweep

Stage D2. 국가별 주요 지수 묶음
  특정 국가의 주가 지수 n개를 각각 포함해 같은 국가/시장권 데이터셋으로 사용
  기본 dataset: d2_kr-kospi-kosdaq_daily, d2_us-nasdaq-spx-dji_daily
  분봉 트랙: d2_kr-kospi-kosdaq_1m (minute split 적용, 일봉 dataset과 절대 혼합하지 않음)
  목적: 같은 시장 내 cross-index token 일반화 확인 (H1의 1차 검증)

Stage D3. 세계 주요 지수
  KIS master와 Kiwoom 코드로 검증된 주요 지수를 통합 데이터셋으로 사용
  기본 dataset: NASDAQ, SPX, DJI, KOSPI, KOSDAQ, N225, NDX, SOX, TWII
  목적: cross-market transfer와 held-out index 평가 (H1의 최종 검증)
```

확장 규칙:
- tokenizer/motif vocabulary는 각 stage에서 재학습하되, 이전 stage의 vocabulary와 매핑 비교를 기록한다(vocabulary가 확장 시 얼마나 유지되는지가 H1의 증거가 된다).
- D1은 각 단일 지수 실험을 독립 run/config로 남긴다. D2는 국가별 묶음을 독립 dataset으로 남기고, D3는 global-major 통합 dataset으로 남긴다.
- 데이터 소스는 국내 지수는 Kiwoom, 해외 지수는 KIS로 고정한다. run config에는 broker method, API용 broker symbol, KIS master 검증용 `master_symbol`, credential 전제, fallback 사용 여부를 기록한다.
- KIS 해외지수는 `brokers.kis.download_overseas_index_info()`의 master 정보로 검증한다. 기본 universe에는 master 검증을 통과한 해외지수만 포함한다.
- NYSE Composite, AMEX Composite, Russell 2000은 현재 KIS master에서 직접 검증되지 않아 기본 universe에서 제외한다. 필요하면 별도 후보 set으로 분리하고 source mapping 근거를 남긴 뒤 편입한다.
- D1에서 확정한 feature 정의, split 규약, 평가 지표는 D2/D3에서 변경하지 않는다. 변경이 필요하면 D1부터 재실행한다.
- held-out index는 D3에서만 의미가 있으므로, D1/D2의 held-out 평가는 시간 축(test 구간)으로 한정한다.

### 구현된 프로토콜 계약

현재 Phase 0의 source of truth는 `kairos/experiments/protocols/candlestick_shape_quantization.json`이다. `kairos.experiments.protocol`은 이 JSON을 로드해 dataclass, `INDEX_SYMBOLS`, `DATASET_REGISTRY`, split config, manifest scaffold를 생성한다.

구현 원칙:
- `protocol.py`에는 고정값을 직접 선언하지 않고 JSON 설정을 검증·변환하는 로직을 둔다.
- `split_name()`은 ISO date 파싱으로 train/validation/test를 판정하고, train 시작 전, split gap, test 종료 후 데이터는 `excluded`로 반환한다.
- feature 정책(`shape_core`, `eps`, `atr_period`, `include_volume`)과 filtering 정책(`zero_range_policy`, `boundary_policy`, `vix_in_main_set`)은 protocol JSON의 별도 section으로 관리한다.
- `build_manifest()`의 notebook 경로는 protocol JSON 기본값을 쓰되 호출자가 `source_notebook`으로 주입할 수 있다.
- KIS chart 호출용 symbol과 master 검증용 `master_symbol`을 분리한다. 예: NASDAQ은 API용 `.IXIC`/`IXIC`, master 검증용 `COMP`를 사용한다.

### 공통 규약

작업:
- index universe 확정. VIX는 거래 대상이 아니고 shape 의미가 다르므로 main set에서 제외하고 별도 분석 set으로 둔다. 해외지수는 KIS master 검증 통과 여부를 기록한다.
- 시간 기준 split: train / validation / test 경계 날짜를 고정하고 문서화한다. 경계에 embargo(최소 label horizon 이상)를 둔다.
- 분봉 트랙 split: provider 제약으로 분봉 데이터는 2025-07 이후만 존재하므로, 일봉 split(train 2005–2016)을 적용하면 train row가 0이 된다. 분봉 dataset에는 별도의 minute split을 적용한다 — train 2025-07-01~2026-01-31, validation 2026-02-01~2026-04-30, test 2026-05-01~ (2026-07-06 확정). 일봉 split은 변경하지 않으며, 일봉과 분봉 결과는 서로 비교 지표로 쓰지 않는다.
- 정규화 규약: ATR, volume median, z-score 통계는 train 구간 또는 t-1까지의 rolling 통계만 사용한다.
- 데이터 품질 필터: 비정상 spike, 장기 미변동 구간 제거 규칙을 정의한다(Kronos의 cleaning pipeline 방식).
- daily를 기본으로 하되 weekly 집계를 함께 생성해 multi-timeframe 평가를 준비한다.

산출물: data manifest, split 명세, cleaning 규칙 문서, 채널별 기술통계 리포트.

게이트: split 명세가 확정되기 전에는 어떤 모델도 fit하지 않는다.

## Phase 1. 단일 Candle Shape Tokenizer

목표: shape core를 안정적인 primitive token으로 변환한다. [00](00-vq-vae-tokenizer-validation-plan.md), [02](02-shape-tokenizer.md)의 실행 단계에 해당한다.

비교 tokenizer:

```text
baseline:  k-means, GMM, hand-crafted bins (모두 shape core 2D 입력)
neural:    VQ-VAE
추가:      FSQ 또는 LFQ/BSQ  # lookup-free, codebook collapse를 구조적으로 회피 (Kronos)
변형:      coarse/fine 계층 토큰  # coarse = 방향+몸통 크기 class, fine = wick 잔차 (Kronos, TimeVQVAE)
optional:  SOM
```

작업:
- Stage D1(단일 지수)에서 시작해 설계를 확정하고, D2/D3 확장 시 재학습과 vocabulary 매핑 비교를 수행한다.
- K sweep: 8, 12, 16, 24, 32 (계층형은 coarse 8 × fine 8 형태로 등가 용량 비교). baseline 단계(step-02)는 8/16/32로 구현되었고, 전 dataset에서 kmeans K=32가 reconstruction 기준 최선이라 후속 VQ 비교의 주 baseline은 `kmeans K=32`다.
- 지표: reconstruction error(shape 공간과 OHLC 복원 공간 양쪽), token entropy, effective vocab size, dead token count, seed 간 prototype 안정성, held-out index transfer(D3에서만)
- 경계 캔들 A/B 비교(위 Feature 설계 참조)
- token별 대표 shape atlas: prototype + 실제 할당 샘플의 λ 분포를 함께 시각화

산출물: candle codebook, encoder, prototype table, shape atlas, tokenizer 비교 리포트.

게이트(go/no-go):
- VQ 계열이 k-means 대비 이점(복원 오차, transfer, 안정성 중 1개 이상)을 못 보이면 k-means를 채택하고 진행한다. 연구 실패가 아니라 tokenizer 단순화 결정이다.
- effective vocab size가 K의 1/4 이하면 K 또는 feature 설계를 재검토한다.

## Phase 2. Token Sequence Corpus와 Label

목표: 지수별 token sequence와 downstream label을 정렬해 corpus를 만든다.

sequence 구성:

```text
candle_t -> (shape_token_t, rel_range_t, gap_t)
# token sequence는 shape_token만으로 만들고,
# rel_range/gap은 continuous side-channel로 병렬 정렬한다.
```

label 정의(모두 t+1 이후 정보만 사용):
- 미래 수익률: log return, horizon k ∈ {1, 5, 20}
- 방향: sign(fwd return), 거래비용 고려 임계 버전 병행
- realized volatility: OHLC 기반 Parkinson 또는 Garman-Klass, horizon 5/20
- volatility expansion: fwd RV / trailing RV > 임계 (binary)
- drawdown: horizon 내 max drawdown, 임계 초과 event (binary)
- regime: trend(가격 vs 200d MA) × vol tercile 조합 label

분석 작업:
- 지수/기간/regime별 token 빈도 분포와 KL divergence
- token unigram/bigram 통계, sequence entropy (motif 학습의 압축 여지 판단 근거)

산출물: tokenized corpus, label store, token statistics report.

게이트: token sequence의 bigram 조건부 entropy가 unigram entropy와 사실상 같다면(순차 구조 없음) Phase 3의 기대 효과를 하향 조정하고 문서화한다.

## Phase 3. 다중 Candle Motif Vocabulary

목표: token sequence에서 variable-length motif vocabulary를 학습한다.

두 경로를 병행 비교한다.

경로 A. 명시적 segmentation/vocabulary:

```text
BPE (vocab 256 / 512 / 1024)
Unigram LM (SentencePiece)
Sequitur / grammar induction
Matrix Profile  # token이 아닌 continuous shape core에 적용, discrete 경로와 교차 검증
supervised: shapelet 또는 WEASEL-style discriminative mining
```

경로 B. 암묵적 sequence model:

```text
소형 decoder-only Transformer를 shape token의 next-token 예측으로 학습 (Kronos 방식 축소판)
motif는 고빈도 n-gram의 perplexity 감소 기여와 attention 패턴으로 사후 추출
```

motif 유의성 통제(초기 계획의 "neutral motif 과다 생성 제어"의 구체화):

```text
1. min support: motif 최소 등장 횟수 하한
2. surrogate 검정: token 빈도를 보존한 Markov-1 surrogate sequence를 생성하고,
   실제 corpus의 motif 빈도가 surrogate 분포의 상위 quantile을 초과하는지 검정
3. redundancy 제거: motif 간 coverage overlap이 임계 초과면 병합
```

산출물: motif codebook(방법별), motif encoder, motif atlas(각 motif의 실제 OHLC window 샘플 grid), 방법별 길이/빈도/coverage/redundancy 비교표.

게이트: surrogate 검정을 통과하는 motif가 사실상 없으면, 단일 candle token + continuous 채널로 Phase 5를 진행하고 "multi-candle motif는 우연 수준"을 잠정 결론으로 기록한다.

## Phase 4. 해석성 평가

목표: 학습된 motif가 사람이 이해 가능한 패턴인지 정량 평가한다.

작업:
- 전통 패턴 overlap: GAF-CNN(1901.05237)의 8개 표준 패턴을 규칙 기반 라벨러로 구현하고, motif-패턴 co-occurrence를 purity/coverage로 측정한다. "재발견"과 "신규 패턴"을 이 지표로 구분한다.
- perturbation 민감도(2001.02767 방식): motif를 구성하는 개별 candle의 λ를 소폭 교란했을 때 motif 할당이 바뀌는 경계를 분석해, motif가 shape의 어떤 성분에 의존하는지 확인한다.
- regime 조건부 분석: 상승/하락/횡보 × vol regime별 motif 발생률과 label 조건부 분포
- rare but predictive motif: 발생 빈도 하위이면서 label 조건부 분포가 unconditional 대비 유의하게 다른 motif 탐색(다중검정 보정 포함)
- variable-length 관점(2201.08669): 고정 window가 아니라 motif의 시작/끝 위치가 안정적으로 잡히는지 확인

산출물: motif interpretation report, traditional pattern comparison table, regime-wise behavior table.

## Phase 5. 예측 유용성 평가

목표: motif token이 downstream task에서 실질적 정보를 갖는지 검증한다.

task: Phase 2의 label 전체 (direction, return regression, vol expansion, drawdown event, regime classification).

feature 표현 비교(동일한 경량 모델, 예: gradient boosting + logistic/linear head로 통일):

```text
1. raw OHLC 파생 feature (returns, ATR 등)
2. technical indicators
3. 전통 candlestick rule 라벨 (1901.05237 8-pattern)
4. single candle token (one-hot / embedding)
5. single token + rel_range + gap
6. motif token (BPE / Unigram / Sequitur / shapelet 각각)
7. SAX / TSAX
8. TS2Vec continuous embedding  # 양자화 정보 손실 상한 측정
9. Kronos pretrained zero-shot  # 외부 사전학습 baseline
10. candle-LM(경로 B) hidden state
```

프로토콜:
- walk-forward: 최소 5개 fold, fold마다 tokenizer/motif vocabulary를 train 구간에서 재학습(vocabulary 안정성도 여기서 측정)
- 지표: 방향 AUC/accuracy, return IC/RankIC(Kronos 지표와 정렬), vol MAE/QLIKE, event PR-AUC, Brier score
- 유의성: Diebold-Mariano 또는 stationary bootstrap으로 baseline 대비 차이 검정
- 분리 리포팅: regime별, timeframe(daily/weekly)별, 지수별
- ablation: vocab size, motif 방법, side-channel 포함 여부, 경계 캔들 A/B

산출물: predictive benchmark table, ablation report, fold별 vocabulary 안정성 리포트.

게이트: motif feature(6)가 single token(4/5) 대비 어떤 task에서도 유의한 개선이 없으면, motif의 가치는 해석성(Phase 4)에 한정된다고 결론에 기록한다.

## Phase 6. 결론 정리

핵심 질문과 판정 근거:

| 질문 | 판정 근거 |
| --- | --- |
| 단일 candle token으로 충분한가 | Phase 5의 4/5 vs 6 비교 |
| multi-candle motif가 성능을 개선하는가 | Phase 5 benchmark + 유의성 검정 |
| 빈도 기반 vs 예측 기반 motif | BPE/Unigram vs shapelet/WEASEL vs candle-LM 비교 |
| 전통 패턴 재발견 vs 신규 패턴 | Phase 4 overlap purity/coverage |
| regime 변화에도 vocabulary가 안정적인가 | fold 간 vocabulary Jaccard/coverage drift |
| 양자화로 잃는 정보는 어느 정도인가 | token 계열 vs TS2Vec/Kronos 대비 성능 격차 |

최종 산출물: tokenizer + motif codebook + benchmark + atlas + research report 초안.

## 리스크와 완화

- token sequence의 순차 구조가 약해 motif가 우연 수준일 수 있다 → Phase 2 entropy 게이트와 Phase 3 surrogate 검정으로 조기 판정
- codebook collapse → FSQ/LFQ 병행, collapse 지표 필수 기록 ([02](02-shape-tokenizer.md))
- 다중검정으로 인한 가짜 predictive motif → 보정된 유의성 기준, walk-forward 재검증
- 지수 daily 데이터의 표본 수 한계 → Stage D2/D3의 지수 pooling으로 보완하되 (H1의 transfer 검증과 겸함), D1 단일 지수에서 표본이 부족하면 timeframe 확장(분봉)을 먼저 검토
