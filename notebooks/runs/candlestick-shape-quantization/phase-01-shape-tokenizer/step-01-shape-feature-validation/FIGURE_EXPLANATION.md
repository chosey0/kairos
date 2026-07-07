# 01 Shape Feature Validation Figure 설명

이 문서는 `01_shape_feature_validation.ipynb` 실행으로 생성된 figure를 해석하기 위한 메모다. 현재 run은 D1 단계의 단일 지수별 데이터셋을 대상으로, 캔들 shape core인 `(s1, s2) = (logit(lambda_o), logit(lambda_c))` 분포를 확인한다.

## Feature 정의

- `lambda_o = (open - low) / (high - low)`
- `lambda_c = (close - low) / (high - low)`
- `s1 = logit(lambda_o)`
- `s2 = logit(lambda_c)`

`lambda_o`와 `lambda_c`는 캔들 range 안에서 시가와 종가가 어디에 위치하는지 나타낸다. 0에 가까우면 저가 근처, 1에 가까우면 고가 근처다. `s1`, `s2`는 winsorize된 lambda를 logit 변환한 값이므로, 0 근처는 range 중앙, 음수는 저가 쪽, 양수는 고가 쪽을 뜻한다.

## 생성된 Figure

### 1. 전체 shape core histogram

파일: [shape_core_histogram_overview.png](shape_core_histogram_overview.png)

모든 정상 조회 데이터셋의 `s1`, `s2` 분포를 같은 축에서 비교하는 facet histogram이다. 각 panel은 하나의 dataset을 나타내며, histogram은 density 기준으로 정규화되어 있다. 따라서 y축 높이는 절대 row 수가 아니라 해당 데이터셋 내부의 상대 밀도를 의미한다.

이 figure의 목적은 다음과 같다.

- 단일 candle token codebook을 학습하기 전에 지수별 shape core 분포가 크게 다른지 확인한다.
- 일봉과 분봉의 boundary candle 비중 차이를 확인한다.
- 국내 지수와 해외 지수의 분포 차이가 codebook 공유를 방해할 정도인지 초기 점검한다.
- zero-range row를 제외한 실제 quantization 대상의 분포를 확인한다.

현재 run에서 보이는 주요 해석은 다음과 같다.

- KOSPI/KOSDAQ 분봉은 `s1`, `s2` 양끝 tail에 질량이 크다. 이는 open 또는 close가 고가/저가에 붙는 boundary candle 비중이 매우 높다는 뜻이다.
- KOSPI/KOSDAQ 일봉은 분봉보다 tail 집중이 약하고 중앙부가 더 두껍다. 일봉 codebook과 분봉 codebook을 완전히 공유할 경우, 분봉의 boundary-heavy 분포가 prototype을 지배할 수 있다.
- NASDAQ/SPX 일봉은 국내 일봉과 유사하게 넓은 분포를 가지지만, `lambda_c` 쪽 고가 근처 질량이 상대적으로 두드러진다.
- NASDAQ/SPX 분봉은 KIS 제공 제한으로 약 102개 row만 포함되어 있으므로, 현재 histogram은 안정적인 분포 추정으로 보기 어렵다.
- DJI daily/1m은 이번 run에서 endpoint가 빈 응답을 반환해 figure에 포함되지 않았다.

### 2. Dataset별 lambda scatter

파일 패턴: `*/figures/lambda_scatter.png`

예시:

- [d1_kospi_daily lambda scatter](d1_kospi_daily/cfg-5628106a/run-20260706-134241_seed-7/figures/lambda_scatter.png)
- [d1_kosdaq_daily lambda scatter](d1_kosdaq_daily/cfg-d17b71e2/run-20260706-134259_seed-7/figures/lambda_scatter.png)
- [d1_nasdaq_daily lambda scatter](d1_nasdaq_daily/cfg-a8489a0e/run-20260706-134349_seed-7/figures/lambda_scatter.png)
- [d1_spx_daily lambda scatter](d1_spx_daily/cfg-e28b4a09/run-20260706-134417_seed-7/figures/lambda_scatter.png)

이 scatter는 x축에 `lambda_o`, y축에 `lambda_c`를 놓아 캔들 내부에서 open과 close의 상대 위치를 보여준다. 대각선 `lambda_c = lambda_o`를 기준으로 위쪽은 close가 open보다 높은 캔들, 아래쪽은 close가 open보다 낮은 캔들이다.

해석 포인트는 다음과 같다.

- 점이 좌상단에 가까우면 open은 저가 근처, close는 고가 근처인 강한 양봉 구조다.
- 점이 우하단에 가까우면 open은 고가 근처, close는 저가 근처인 강한 음봉 구조다.
- 점이 대각선 근처에 많으면 body가 짧은 doji 또는 small-body candle 비중이 높다.
- 점이 사각형 경계에 많이 붙으면 open/close가 high/low에 붙는 boundary candle 비중이 높다.

### 3. Dataset별 shape core histogram

파일 패턴: `*/figures/shape_core_hist.png`

예시:

- [d1_kospi_daily shape core histogram](d1_kospi_daily/cfg-5628106a/run-20260706-134241_seed-7/figures/shape_core_hist.png)
- [d1_kosdaq_daily shape core histogram](d1_kosdaq_daily/cfg-d17b71e2/run-20260706-134259_seed-7/figures/shape_core_hist.png)
- [d1_nasdaq_daily shape core histogram](d1_nasdaq_daily/cfg-a8489a0e/run-20260706-134349_seed-7/figures/shape_core_hist.png)
- [d1_spx_daily shape core histogram](d1_spx_daily/cfg-e28b4a09/run-20260706-134417_seed-7/figures/shape_core_hist.png)

이 histogram은 각 dataset 안에서 `s1 = logit(lambda_o)`와 `s2 = logit(lambda_c)`의 marginal distribution을 따로 보여준다. 이 figure는 tokenizer 입력 차원별로 scaling, winsorization, boundary flag 정책이 적절한지 확인하는 용도다.

해석 포인트는 다음과 같다.

- `s1`, `s2`가 모두 0 주변에 몰리면 open/close가 range 중앙에 자주 위치한다.
- 음수 tail이 길면 open 또는 close가 low 근처에 위치하는 경우가 많다.
- 양수 tail이 길면 open 또는 close가 high 근처에 위치하는 경우가 많다.
- 양끝에 spike가 있으면 winsorize boundary에 걸린 candle이 많다는 뜻이며, codebook 학습 시 boundary token 또는 boundary flag를 따로 평가해야 한다.

## Current Run Summary

| Dataset | Status | Provider | Rows | Date range | Zero-range | Boundary | Figures |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| `d1_dji_1m` | skipped | kis | 0 | - | 0 | 0 | - |
| `d1_dji_daily` | skipped | kis | 0 | - | 0 | 0 | - |
| `d1_kosdaq_1m` | ok | kiwoom | 94,236 | 2025-07-01 09:00:00 ~ 2026-07-03 15:30:00 | 248 (0.26%) | 76,082 (80.74%) | `lambda_scatter.png`, `shape_core_hist.png` |
| `d1_kosdaq_daily` | ok | kiwoom | 7,365 | 1997-01-03 ~ 2026-07-03 | 12 (0.16%) | 1,763 (23.94%) | `lambda_scatter.png`, `shape_core_hist.png` |
| `d1_kospi_1m` | ok | kiwoom | 94,149 | 2025-07-01 09:00:00 ~ 2026-07-03 15:30:00 | 248 (0.26%) | 68,635 (72.90%) | `lambda_scatter.png`, `shape_core_hist.png` |
| `d1_kospi_daily` | ok | kiwoom | 9,420 | 1990-01-03 ~ 2026-07-03 | 0 (0.00%) | 2,582 (27.41%) | `lambda_scatter.png`, `shape_core_hist.png` |
| `d1_nasdaq_1m` | ok | kis | 102 | 2026-07-02 14:33:59 ~ 2026-07-02 16:14:59 | 15 (14.71%) | 23 (22.55%) | `lambda_scatter.png`, `shape_core_hist.png` |
| `d1_nasdaq_daily` | ok | kis | 6,664 | 2000-01-03 ~ 2026-07-02 | 1 (0.02%) | 782 (11.73%) | `lambda_scatter.png`, `shape_core_hist.png` |
| `d1_spx_1m` | ok | kis | 102 | 2026-07-02 14:45:00 ~ 2026-07-06 09:34:00 | 17 (16.67%) | 30 (29.41%) | `lambda_scatter.png`, `shape_core_hist.png` |
| `d1_spx_daily` | ok | kis | 5,302 | 2005-06-02 ~ 2026-07-02 | 0 (0.00%) | 1,280 (24.14%) | `lambda_scatter.png`, `shape_core_hist.png` |

## Research Implications

현재 figure는 단일 candle codebook 설계에서 다음 결정을 우선 검증해야 함을 보여준다.

- 분봉과 일봉은 같은 `(s1, s2)` 정의를 쓰되, 하나의 codebook을 공유할지 별도 codebook을 둘지 실험으로 확인해야 한다.
- boundary candle 비중이 큰 분봉 데이터는 일반 k-means/VQ prototype을 boundary 쪽으로 끌어당길 수 있으므로, boundary flag 또는 special handling을 비교군에 포함해야 한다.
- zero-range candle은 shape core를 정의할 수 없으므로 special token으로 분리하고, codebook fit 대상에서는 제외하는 현재 정책이 타당하다.
- KIS 해외 일봉은 date-window pagination으로 충분한 표본을 확보했지만, KIS 해외 분봉은 provider 제한으로 D1 tokenizer 검증의 주된 근거로 쓰기 어렵다.

## Next Step Decision

이번 run 검토 결과에 따라, 다음 단계(tokenizer 학습·검증)는 **boundary 분리 방식**으로 진행하기로 확정한다 (2026-07-06).

- 검증 대상 dataset은 `d1_kospi_daily`, `d1_kosdaq_daily`, `d1_nasdaq_daily`, `d1_spx_daily`, `d1_kospi_1m`, `d1_kosdaq_1m` 6개이며, 각각 독립 데이터셋으로 검증한다. 합쳐서 하나의 데이터셋으로 쓰지 않는다.
- `d1_nasdaq_1m`, `d1_spx_1m`은 표본 부족(약 102 row)으로 제외한다. DJI/NDX/SOX는 미확보 상태로 두고 수집 이슈 해결 후 추가한다.
- boundary candle은 데이터에서 제외하지 않는다. 분봉의 73~81%를 차지하는 정보성 캔들(marubozu 등)이므로, 제외하면 token sequence에 구멍이 생겨 downstream 목표와 충돌한다.
- 대신 boundary 여부를 이산 구조로 분리한다. `lambda_o ∈ {low boundary, interior, high boundary} × lambda_c ∈ {low boundary, interior, high boundary}`의 9개 조합 중:
  - interior × interior 캔들만 `(s1, s2)` 연속 codebook 학습 대상으로 사용한다.
  - 나머지 8개 boundary 조합은 전용 discrete token으로 부여한다.
- 즉 boundary candle은 codebook fit에서만 제외되고, 모든 캔들은 token을 받는다.
- zero-range candle은 기존 정책대로 별도 special token으로 유지한다.
- 전체 데이터를 winsorize+logit 후 그대로 VQ 학습하는 baseline(비교군 A)은 boundary prototype 점유율 측정용 비교 기준으로 함께 유지한다.

## Caveats

- Histogram은 density 기준이므로 dataset별 row count 차이는 y축 높이로 비교하면 안 된다.
- `shape_core_histogram_overview.png`는 zero-range row를 제외하고 그린다.
- Boundary candle은 winsorize 후 logit 변환되므로, 양끝 spike는 실제 무한값이 아니라 epsilon boundary에 모인 값이다.
- DJI dataset은 이번 run에서 빈 응답으로 skip되었으므로, DJI 관련 분포 비교는 아직 유효하지 않다.
- 해외 분봉은 KIS endpoint의 최신 약 102개 row 제한 때문에 현재 run의 통계 신뢰도가 낮다.
