# 02 Tokenizer Baseline 결과 해석

> 원본 run 위치: `notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/` — 이 문서의 figure/table 링크는 gitignore된 로컬 run 산출물을 가리키므로 로컬 체크아웃에서만 열린다.

이 문서는 `02_tokenizer_baselines.ipynb`가 생성한 baseline 결과를 해석하기 위한 메모다. 해석 기준은 이전 단계의 [FIGURE_EXPLANATION.md](phase-01-step-01-shape-feature-validation.md)이다.

기준 run은 D2 병합 dataset과 minute split을 추가한 2026-07-06 15:02 UTC 실행(`run-20260706-1502*`~`run-20260706-1503*`)이다. daily D1 4개의 cfg hash와 seed 평균 MSE는 이전 run(`run-20260706-1429*`)과 동일하게 재현됐으므로 이전 daily 수치는 그대로 유효하며, 이전 run 디렉토리는 superseded 상태로 남겨 두었다. 국내 1m 2개는 이번 실행부터 minute split(`split_minute`: train 2025-07-01~2026-01-31, validation ~2026-04-30, test 2026-05-01~)이 적용되어 처음으로 fit되었고, cfg hash도 split 변경을 반영해 바뀌었다.

## 실험 목적

Phase 1의 첫 tokenizer baseline은 VQ 계열 모델을 학습하기 전에, 2D shape core `(s1, s2) = (logit(lambda_o), logit(lambda_c))`가 단순 clustering만으로 얼마나 안정적으로 quantize되는지 확인하는 단계다.

대상 dataset은 세 그룹이다.

- D1 daily 4개: `d1_kospi_daily`, `d1_kosdaq_daily`, `d1_nasdaq_daily`, `d1_spx_daily` (daily split: train 2005–2016)
- D1 1m 2개: `d1_kospi_1m`, `d1_kosdaq_1m` (minute split: train 2025-07~2026-01)
- D2 병합 2개: `d2_kr-kospi-kosdaq_daily`, `d2_kr-kospi-kosdaq_1m` — 성분 D1 shape row를 timestamp 정렬로 concatenate. 일봉과 분봉은 절대 섞지 않는다. step-01의 [병합 분포 검증](phase-01-step-01-merge-distribution-check.md)이 이 구성의 근거다.

비교한 baseline은 다음과 같다.

- `kmeans`: train split에서 scaler와 centroid를 fit한 뒤 전체 split에 할당
- `gmm`: train split에서 Gaussian mixture를 fit한 뒤 posterior argmax로 할당
- `handcrafted_lambda_bins`: `lambda_o`, `lambda_c`를 균등 bin으로 나눈 규칙 기반 tokenizer

boundary candle은 두 정책으로 비교했다.

- `include_boundary`: winsorize된 boundary row를 포함한다. 이 run 시점의 기본 정책이었다.
- `exclude_boundary`: open/close가 high/low에 붙은 boundary row를 제거한다. boundary tail이 codebook을 지배하는지 확인하기 위한 비교군이다.

이 run 이후 step-01 [Next Step Decision](phase-01-step-01-shape-feature-validation.md#next-step-decision)에서 boundary 분리 방식(비교군 B: boundary 조합 8개는 전용 discrete token, interior × interior만 연속 codebook 학습)이 확정되었다. 아래 결과는 그 결정 이전 설계의 기록이며, 비교군 B 관점에서 다음처럼 재해석한다.

- `include_boundary` = 비교군 A. boundary가 codebook을 얼마나 점유하는지 측정하는 baseline으로 유지한다.
- `exclude_boundary` = 비교군 B의 interior-only codebook fit에 대한 근사. 단, 비교군 B와 달리 이 run에서는 제거된 boundary row에 token을 부여하지 않았다.

zero-range candle은 모든 정책에서 tokenizer fit/evaluation 대상에서 제외하고 special token 후보로만 기록한다.

## Dataset 처리 결과

| Dataset | Status | Split | Source rows | Include-boundary train rows | Exclude-boundary train rows | Warning |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `d1_kosdaq_1m` | ok | minute | 94,236 | 55,185 | 11,147 | `boundary_heavy_distribution` |
| `d1_kosdaq_daily` | ok | daily | 7,365 | 2,976 | 2,363 | - |
| `d1_kospi_1m` | ok | minute | 94,149 | 55,185 | 16,304 | `boundary_heavy_distribution` |
| `d1_kospi_daily` | ok | daily | 9,420 | 2,976 | 2,379 | - |
| `d1_nasdaq_daily` | ok | daily | 6,664 | 3,020 | 2,605 | - |
| `d1_spx_daily` | ok | daily | 5,302 | 2,915 | 2,004 | - |
| `d2_kr-kospi-kosdaq_1m` | ok | minute | 188,385 | 110,370 | 27,451 | `boundary_heavy_distribution` |
| `d2_kr-kospi-kosdaq_daily` | ok | daily | 16,785 | 5,952 | 4,742 | - |

해석:

- 8개 dataset 전부 fit되었다. 국내 1분봉은 이전 run에서 daily split 때문에 train row 0으로 skip되었으나, minute split 도입으로 train 55,185 row를 확보했다.
- 해외 1분봉(`d1_nasdaq_1m`, `d1_spx_1m`)과 DJI daily/1m은 step-01 Next Step Decision에 따라 input universe(`FEATURE_INPUTS`)에서 제외했다. 이전 run에서는 각각 KIS endpoint 표본 부족(약 102 row)과 빈 응답으로 skipped였다.
- D2 병합의 source rows는 성분 D1 합과 정확히 일치한다 (예: 1m 94,149+94,236=188,385).

## Reconstruction 기준 Best Baseline

아래 표는 seed `7`, `17`, `37`의 최신 run을 평균낸 결과다. MSE는 `(s1, s2)` 공간의 reconstruction error다.

| Dataset | Boundary policy | Best model | K | Mean MSE | MSE std | Effective vocab | Dead tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `d1_kosdaq_daily` | include_boundary | kmeans | 32 | 0.643314 | 0.001755 | 28.54 | 0.00 |
| `d1_kosdaq_daily` | exclude_boundary | kmeans | 32 | 0.508771 | 0.002698 | 26.46 | 0.00 |
| `d1_kospi_daily` | include_boundary | kmeans | 32 | 0.641517 | 0.008802 | 28.68 | 0.00 |
| `d1_kospi_daily` | exclude_boundary | kmeans | 32 | 0.472221 | 0.002208 | 27.22 | 0.00 |
| `d1_nasdaq_daily` | include_boundary | kmeans | 32 | 0.608110 | 0.002505 | 27.63 | 0.00 |
| `d1_nasdaq_daily` | exclude_boundary | kmeans | 32 | 0.452202 | 0.000970 | 28.55 | 0.00 |
| `d1_spx_daily` | include_boundary | kmeans | 32 | 0.556864 | 0.002712 | 27.08 | 0.00 |
| `d1_spx_daily` | exclude_boundary | kmeans | 32 | 0.401378 | 0.001635 | 27.45 | 0.00 |
| `d1_kospi_1m` | include_boundary | kmeans | 32 | 0.377890 | 0.001003 | 26.52 | 0.00 |
| `d1_kospi_1m` | exclude_boundary | kmeans | 32 | 0.339140 | 0.000936 | 28.10 | 0.00 |
| `d1_kosdaq_1m` | include_boundary | kmeans | 32 | 0.243703 | 0.000615 | 24.20 | 0.00 |
| `d1_kosdaq_1m` | exclude_boundary | kmeans | 32 | 0.244359 | 0.002332 | 30.12 | 0.00 |
| `d2_kr-kospi-kosdaq_daily` | include_boundary | kmeans | 32 | 0.644516 | 0.004913 | 28.89 | 0.00 |
| `d2_kr-kospi-kosdaq_daily` | exclude_boundary | kmeans | 32 | 0.486715 | 0.005247 | 26.67 | 0.00 |
| `d2_kr-kospi-kosdaq_1m` | include_boundary | kmeans | 32 | 0.319937 | 0.000513 | 24.99 | 0.00 |
| `d2_kr-kospi-kosdaq_1m` | exclude_boundary | kmeans | 32 | 0.310992 | 0.000532 | 28.39 | 0.00 |

핵심 결론:

- 8개 dataset, 두 boundary policy 전부에서 `kmeans K=32`가 reconstruction MSE 기준 최선이다.
- dead token은 모든 조합에서 0이고 effective vocab은 24~30 수준이다. codebook collapse 징후는 없다. 단, boundary 비중이 큰 1m의 `include_boundary`에서 effective vocab이 24~26으로 낮아지는데, 이는 boundary point mass에 질량이 몰리기 때문이다.
- seed 간 MSE 표준편차가 작다. daily와 minute 트랙 모두 안정적으로 반복된다.

## Model별 비교

| Dataset | Boundary policy | kmeans K=32 MSE | gmm K=32 MSE | handcrafted K=16 MSE |
| --- | --- | ---: | ---: | ---: |
| `d1_kosdaq_daily` | include_boundary | 0.643314 | 1.284534 | 8.734297 |
| `d1_kosdaq_daily` | exclude_boundary | 0.508771 | 0.580506 | 2.641022 |
| `d1_kospi_daily` | include_boundary | 0.641517 | 1.521082 | 9.083159 |
| `d1_kospi_daily` | exclude_boundary | 0.472221 | 0.552359 | 2.002041 |
| `d1_nasdaq_daily` | include_boundary | 0.608110 | 0.973902 | 5.155207 |
| `d1_nasdaq_daily` | exclude_boundary | 0.452202 | 0.526529 | 2.243917 |
| `d1_spx_daily` | include_boundary | 0.556864 | 0.918476 | 7.640190 |
| `d1_spx_daily` | exclude_boundary | 0.401378 | 0.482266 | 1.551601 |
| `d1_kospi_1m` | include_boundary | 0.377890 | 0.503322 | 22.863393 |
| `d1_kospi_1m` | exclude_boundary | 0.339140 | 0.374814 | 0.977376 |
| `d1_kosdaq_1m` | include_boundary | 0.243703 | 0.286658 | 26.692801 |
| `d1_kosdaq_1m` | exclude_boundary | 0.244359 | 0.281231 | 0.547421 |
| `d2_kr-kospi-kosdaq_daily` | include_boundary | 0.644516 | 1.582688 | 8.930224 |
| `d2_kr-kospi-kosdaq_daily` | exclude_boundary | 0.486715 | 0.534200 | 2.289449 |
| `d2_kr-kospi-kosdaq_1m` | include_boundary | 0.319937 | 0.442302 | 24.778984 |
| `d2_kr-kospi-kosdaq_1m` | exclude_boundary | 0.310992 | 0.354004 | 0.799048 |

해석:

- `kmeans`가 가장 강한 baseline이다. 입력이 2D이고 Euclidean geometry에 맞춰 logit transform을 적용했기 때문에 자연스러운 결과다.
- `gmm`은 `exclude_boundary`에서는 k-means에 가까워지지만, `include_boundary`에서는 boundary tail과 비정규 분포의 영향을 더 크게 받는다.
- `handcrafted_lambda_bins`는 가장 약하다. lambda 공간에서 균등 bin을 자르면 logit 공간의 tail을 제대로 복원하지 못한다. 특히 boundary 포함 조건에서 MSE가 크게 나빠진다.
- 1m `include_boundary`에서 handcrafted MSE가 22~27로 폭발하는 것은 boundary 질량(73~81%)이 bin 가장자리(logit ±6.9)에 있는데 복원값은 bin 중심의 logit이라 오차가 극대화되기 때문이다. boundary 비중이 큰 데이터에서 균등 bin 방식이 왜 성립하지 않는지 보여주는 정량 근거다.

## Boundary A/B 해석

| Dataset | Include best MSE | Exclude best MSE | MSE reduction from exclude | Train boundary rows removed |
| --- | ---: | ---: | ---: | ---: |
| `d1_kosdaq_daily` | 0.643314 | 0.508771 | 20.91% | 613 |
| `d1_kospi_daily` | 0.641517 | 0.472221 | 26.39% | 597 |
| `d1_nasdaq_daily` | 0.608110 | 0.452202 | 25.64% | 415 |
| `d1_spx_daily` | 0.556864 | 0.401378 | 27.92% | 911 |
| `d1_kospi_1m` | 0.377890 | 0.339140 | 10.25% | 38,881 |
| `d1_kosdaq_1m` | 0.243703 | 0.244359 | -0.27% | 44,038 |
| `d2_kr-kospi-kosdaq_daily` | 0.644516 | 0.486715 | 24.48% | 1,210 |
| `d2_kr-kospi-kosdaq_1m` | 0.319937 | 0.310992 | 2.80% | 82,919 |

daily에서 boundary를 제외하면 MSE는 20-28% 정도 낮아진다. 이는 모델이 더 좋아졌다는 뜻만은 아니다. boundary row가 logit 공간의 양끝 tail에 있고, 그 tail을 제거하면 더 쉬운 중앙부 분포만 남기 때문에 reconstruction error가 낮아지는 효과가 있다.

반면 1m에서는 exclude 효과가 거의 없다(-0.3~10%). boundary가 73~81%인 데이터에서는 k-means가 prototype 다수를 boundary point mass에 직접 배치해 해당 row들의 오차를 사실상 0으로 만들기 때문이다. 즉 분봉에서 boundary의 비용은 MSE가 아니라 **codebook 용량 잠식**이다 — `include_boundary`의 effective vocab이 24~26으로 떨어지고, atlas에서 32개 token 중 20개 이상이 boundary 좌표(`lambda` 0.00/1.00)를 가진 prototype이 된다. 이것이 비교군 B(boundary discrete token 분리)가 해결하려는 문제의 정량적 형태다.

따라서 현재 결론은 다음처럼 잡는 편이 안전하다.

- reconstruction MSE만 보면 `exclude_boundary`가 유리하다.
- 하지만 boundary candle은 반복적으로 나타나는 실제 shape이므로 단순 제거하면 vocabulary가 중요한 극단 shape을 잃을 수 있다.
- 이 두 관찰이 비교군 B(boundary 분리)를 지지한다. boundary를 discrete token으로 분리하면 codebook은 `exclude_boundary` 수준의 쉬운 interior 분포만 학습하면서도, boundary shape 정보는 전용 token으로 보존된다.
- `include_boundary`에서도 k-means K=32는 dead token이 0이므로, boundary 포함이 codebook collapse를 즉시 유발한다고 보기는 어렵다.

## D2 병합 해석 (H1 1차 검증)

같은 시장(KR)·같은 interval의 두 지수를 하나의 dataset으로 병합했을 때 codebook 품질이 유지되는지 확인한다. `kmeans K=32`, seed 평균 기준:

| Track | Boundary policy | `d1_kospi` | `d1_kosdaq` | `d2_kr` 병합 |
| --- | --- | ---: | ---: | ---: |
| daily | include_boundary | 0.641517 | 0.643314 | 0.644516 |
| daily | exclude_boundary | 0.472221 | 0.508771 | 0.486715 |
| 1m | include_boundary | 0.377890 | 0.243703 | 0.319937 |
| 1m | exclude_boundary | 0.339140 | 0.244359 | 0.310992 |

해석:

- daily 병합 MSE(0.6445)는 최악 성분(KOSDAQ 0.6433)과 0.2% 차이다. 두 지수를 하나의 codebook으로 표현해도 reconstruction 저하가 사실상 없다. exclude에서도 병합값(0.4867)이 성분 범위(0.4722~0.5088) 안에 있다.
- 1m 병합(0.3199)도 성분 가중평균(약 0.311)에 근접하고 성분 범위(0.2437~0.3779) 안이다.
- 병합 dataset에서도 dead token 0, effective vocab 25~29로 collapse 징후가 없다.
- 이 결과는 step-01 [병합 분포 검증](phase-01-step-01-merge-distribution-check.md)의 JSD 결론(같은 interval 병합은 분포 유지)과 일관되며, **같은 시장·같은 interval의 cross-index token 일반화(H1)의 1차 근거**가 된다.
- 남은 확인 사항: 병합 codebook에서 token 점유율을 지수별로 분해해 한 지수가 특정 token을 독점하는지 보는 것. 현재 metrics에는 source별 분해가 없으므로 후속 단계에서 추가한다.

## Figure 읽는 법

### Baseline reconstruction MSE

각 run에는 `figures/baseline_reconstruction_mse.png`가 있다. 최신 seed 7 기준 예시는 다음과 같다.

- [KOSPI daily baseline figure](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kospi_daily/cfg-d35d74f4/run-20260706-150235_seed-7/figures/baseline_reconstruction_mse.png)
- [KOSDAQ daily baseline figure](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kosdaq_daily/cfg-e0bd2eb7/run-20260706-150303_seed-7/figures/baseline_reconstruction_mse.png)
- [NASDAQ daily baseline figure](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_nasdaq_daily/cfg-960d7a53/run-20260706-150330_seed-7/figures/baseline_reconstruction_mse.png)
- [SPX daily baseline figure](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_spx_daily/cfg-fc473cf1/run-20260706-150336_seed-7/figures/baseline_reconstruction_mse.png)
- [KOSPI 1m baseline figure](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kospi_1m/cfg-e03165b4/run-20260706-150241_seed-7/figures/baseline_reconstruction_mse.png)
- [KOSDAQ 1m baseline figure](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kosdaq_1m/cfg-7fff2966/run-20260706-150309_seed-7/figures/baseline_reconstruction_mse.png)
- [KR daily 병합 baseline figure](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d2_kr-kospi-kosdaq_daily/cfg-f4c4fcdf/run-20260706-150340_seed-7/figures/baseline_reconstruction_mse.png)
- [KR 1m 병합 baseline figure](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d2_kr-kospi-kosdaq_1m/cfg-cced087a/run-20260706-150350_seed-7/figures/baseline_reconstruction_mse.png)

이 figure는 boundary policy, model, K 조합별 reconstruction MSE를 막대로 보여준다. 낮을수록 같은 shape point를 더 가까운 prototype으로 복원했다는 뜻이다. 단, 이 figure는 reconstruction 품질만 보여주므로 token 해석성, regime별 안정성, downstream 유용성까지 판단하지는 않는다.

### Cluster shape atlas

각 dataset의 seed 7 run에는 `kmeans K=32`가 찾은 cluster를 캔들 형태로 그린 atlas 2장(`token-shapes__<dataset>__kmeans__k32-<boundary-policy>__seed-7.png`)이 있다.

- [KOSPI daily atlas (include_boundary)](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kospi_daily/cfg-d35d74f4/run-20260706-150235_seed-7/figures/token-shapes__d1_kospi_daily__kmeans__k32-include_boundary__seed-7.png) / [exclude_boundary](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kospi_daily/cfg-d35d74f4/run-20260706-150235_seed-7/figures/token-shapes__d1_kospi_daily__kmeans__k32-exclude_boundary__seed-7.png)
- [KOSDAQ daily atlas (include_boundary)](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kosdaq_daily/cfg-e0bd2eb7/run-20260706-150303_seed-7/figures/token-shapes__d1_kosdaq_daily__kmeans__k32-include_boundary__seed-7.png) / [exclude_boundary](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kosdaq_daily/cfg-e0bd2eb7/run-20260706-150303_seed-7/figures/token-shapes__d1_kosdaq_daily__kmeans__k32-exclude_boundary__seed-7.png)
- [NASDAQ daily atlas (include_boundary)](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_nasdaq_daily/cfg-960d7a53/run-20260706-150330_seed-7/figures/token-shapes__d1_nasdaq_daily__kmeans__k32-include_boundary__seed-7.png) / [exclude_boundary](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_nasdaq_daily/cfg-960d7a53/run-20260706-150330_seed-7/figures/token-shapes__d1_nasdaq_daily__kmeans__k32-exclude_boundary__seed-7.png)
- [SPX daily atlas (include_boundary)](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_spx_daily/cfg-fc473cf1/run-20260706-150336_seed-7/figures/token-shapes__d1_spx_daily__kmeans__k32-include_boundary__seed-7.png) / [exclude_boundary](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_spx_daily/cfg-fc473cf1/run-20260706-150336_seed-7/figures/token-shapes__d1_spx_daily__kmeans__k32-exclude_boundary__seed-7.png)
- [KOSPI 1m atlas (include_boundary)](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kospi_1m/cfg-e03165b4/run-20260706-150241_seed-7/figures/token-shapes__d1_kospi_1m__kmeans__k32-include_boundary__seed-7.png) / [exclude_boundary](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kospi_1m/cfg-e03165b4/run-20260706-150241_seed-7/figures/token-shapes__d1_kospi_1m__kmeans__k32-exclude_boundary__seed-7.png)
- [KOSDAQ 1m atlas (include_boundary)](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kosdaq_1m/cfg-7fff2966/run-20260706-150309_seed-7/figures/token-shapes__d1_kosdaq_1m__kmeans__k32-include_boundary__seed-7.png) / [exclude_boundary](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d1_kosdaq_1m/cfg-7fff2966/run-20260706-150309_seed-7/figures/token-shapes__d1_kosdaq_1m__kmeans__k32-exclude_boundary__seed-7.png)
- [KR daily 병합 atlas (include_boundary)](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d2_kr-kospi-kosdaq_daily/cfg-f4c4fcdf/run-20260706-150340_seed-7/figures/token-shapes__d2_kr-kospi-kosdaq_daily__kmeans__k32-include_boundary__seed-7.png) / [exclude_boundary](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d2_kr-kospi-kosdaq_daily/cfg-f4c4fcdf/run-20260706-150340_seed-7/figures/token-shapes__d2_kr-kospi-kosdaq_daily__kmeans__k32-exclude_boundary__seed-7.png)
- [KR 1m 병합 atlas (include_boundary)](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d2_kr-kospi-kosdaq_1m/cfg-cced087a/run-20260706-150350_seed-7/figures/token-shapes__d2_kr-kospi-kosdaq_1m__kmeans__k32-include_boundary__seed-7.png) / [exclude_boundary](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-02-tokenizer-baselines/d2_kr-kospi-kosdaq_1m/cfg-cced087a/run-20260706-150350_seed-7/figures/token-shapes__d2_kr-kospi-kosdaq_1m__kmeans__k32-exclude_boundary__seed-7.png)

읽는 법은 다음과 같다.

- 각 칸은 하나의 token(cluster prototype)이다. prototype `(s1, s2)`를 sigmoid로 되돌린 `(lambda_o, lambda_c)`를 low=0, high=1로 정규화한 캔들 glyph로 그렸다. 세로선이 range(low~high), body가 open~close이며 양봉은 red, 음봉은 blue다.
- 칸 제목은 token id, 전체 row 대비 token 점유율, prototype의 `(lambda_o, lambda_c)`이고, 점유율 내림차순으로 배치된다.
- `lambda`가 0.00 또는 1.00에 붙은 prototype이 boundary candle을 흡수한 token이다. `include_boundary` atlas에서 이런 token이 codebook 32개 중 몇 개를 차지하는지가 비교군 B(boundary discrete token 분리)의 근거 자료다.
- `exclude_boundary` atlas는 interior 분포만 학습했을 때 codebook이 중앙부 shape을 얼마나 세밀하게 나누는지 확인하는 용도다.
- shape core만 사용하므로 캔들의 절대 크기(`rel_range` 등)는 표현되지 않는다. 같은 glyph라도 실제 캔들의 range는 다를 수 있다.
- k-means fit 절차와 seed가 baseline 표와 동일하므로, atlas의 token id와 점유율은 `metrics.json`의 histogram과 일치한다.

## 현재 단계의 결정

현재 D1 daily baseline 기준으로 다음 결정을 기록한다.

1. 다음 VQ/FSQ/LFQ tokenizer 비교의 주 baseline은 `kmeans K=32`로 둔다.
2. boundary 정책은 step-01에서 확정한 비교군 B를 기본으로 한다. boundary candle을 데이터에서 제거하지 않고, `lambda_o × lambda_c`의 boundary 조합 8개에 전용 discrete token을 부여하며, interior × interior 캔들만 연속 codebook 학습에 사용한다. 따라서 다음 tokenizer run의 주 지표는 interior-only fit의 reconstruction MSE(이 run의 `exclude_boundary`에 근사)이고, `include_boundary`는 비교군 A baseline으로 함께 기록한다. 전체 vocabulary는 연속 codebook K개 + boundary token 8개 + zero-range special token으로 구성된다.
3. `handcrafted_lambda_bins`는 해석 가능한 약한 baseline으로만 유지한다. reconstruction 경쟁 baseline으로는 부적합하다.
4. `gmm`은 k-means보다 일관되게 약하므로, 후속 단계에서는 보조 baseline으로만 둔다.
5. 분봉 트랙은 minute split(`split_minute`, 2026-07-06 확정)으로 baseline이 확보되었다. 분봉에서도 주 baseline은 `kmeans K=32`이며, 분봉의 boundary 비용은 MSE가 아니라 codebook 용량 잠식이므로 비교군 B 평가 지표에 boundary token 점유율을 반드시 포함한다.
6. D2 KR 병합(daily, 1m)은 성분 대비 reconstruction 저하가 사실상 없어 채택한다. 다음 VQ/FSQ/LFQ 비교는 D1 daily 4개 + D1 1m 2개 + D2 KR 2개를 대상으로 진행하고, 병합 codebook의 지수별 token 점유율 분해를 지표로 추가한다.

## 다음 검증 단계

다음 단계의 핵심 질문은 다음으로 고정한다.

> VQ-VAE Latent Clustering이 `kmeans K=32 include_boundary`보다 boundary를 보존하면서도 reconstruction, stability, token usage 중 하나 이상에서 이기는가?

검증 기준:

- 비교 기준선은 `include_boundary` 조건의 `kmeans K=32`다. boundary를 제거한 `exclude_boundary` 결과는 쉬운 interior-only 분포의 참고선으로만 사용한다.
- VQ-VAE는 encoder-decoder를 train split에서만 학습하고, encoder latent에 clustering을 적용한다.
- boundary 보존은 단순히 MSE가 낮은지가 아니라, boundary candle이 별도 token 또는 안정적인 latent region으로 표현되는지로 판단한다.
- 통과 조건은 `kmeans K=32 include_boundary` 대비 다음 중 최소 하나 이상에서 개선을 보이는 것이다: reconstruction error, seed 간 stability, effective vocab/token usage.
- boundary 보존이 악화되면 reconstruction MSE가 낮아도 채택하지 않는다.

## Caveats

- 이 결과는 D1 단일 지수와 D2 KR 병합까지의 baseline이다. D2 US 병합과 D3 cross-market vocabulary 안정성은 아직 검증하지 않았다.
- daily와 1m의 MSE 절대값은 서로 비교하면 안 된다. split, 기간(29~36년 vs 1년), boundary 비중이 모두 다르며, 1m의 낮은 MSE는 boundary point mass가 쉽게 복원되는 효과를 포함한다.
- 1m 트랙은 2025-07~2026-07 약 1년 커버리지라 regime 다양성이 제한적이다. minute split 경계는 이후 데이터가 추가되어도 고정한다.
- MSE는 `(s1, s2)` shape space 기준이다. OHLC 복원 공간, token-pattern overlap, downstream 성능은 별도 평가가 필요하다.
- daily D1 4개의 이전 run(`run-20260706-1429*`)은 동일 cfg의 superseded run으로 남아 있다. 이 문서는 최신 run(`run-20260706-1502*`~`1503*`)을 기준으로 한다.
- `exclude_boundary`의 낮은 MSE는 tail 제거 효과를 포함하므로, 연구 정책 변경의 직접 근거로 쓰면 안 된다.
- 이 run은 비교군 B 확정 이전 설계라 boundary discrete token을 구현하지 않았다. 표의 K, effective vocab, dead token 수치는 연속 codebook만 센 값이며, 비교군 B 기준 전체 vocabulary(K + boundary 8 + zero-range 1)와 다르다. 비교군 B 구현 후의 run이 이 baseline을 대체한다.
- VQ 계열 모델을 채택하려면 이 k-means baseline 대비 reconstruction, stability, transfer 중 최소 하나에서 이점을 보여야 한다.
