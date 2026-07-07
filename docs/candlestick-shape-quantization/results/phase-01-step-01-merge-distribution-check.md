# KR Dataset 병합 분포 유지 검증 (미니스텝)

> 원본 run 위치: `notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-01-shape-feature-validation/merge-distribution-check/` — 이 문서의 figure/table 링크는 gitignore된 로컬 run 산출물을 가리키므로 로컬 체크아웃에서만 열린다.

D2 단계로 넘어가기 전, KR dataset을 병합했을 때 shape core `(s1, s2)` 분포가 유지되는지 확인한 미니스텝 기록이다 (2026-07-06). 입력은 step-01 feature run의 `shape_sample.csv`이며, zero-range row는 pipeline 규약대로 제외했다. 재현 스크립트는 [make_figures.py](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-01-shape-feature-validation/merge-distribution-check/make_figures.py)다.

## 방법

- 세 가지 병합을 단순 concatenation으로 구성했다: `kr_daily`(KOSPI+KOSDAQ daily), `kr_1m`(KOSPI+KOSDAQ 1m), `kr_all`(4개 전부).
- 각 병합에 대해 성분 dataset과 병합 dataset의 `s1`, `s2` marginal density를 겹쳐 그렸다 (figure 3장).
- 유사도는 Jensen-Shannon distance(base 2, 0=동일, 1=완전 분리)로 정량화했다. marginal(80 bins)과 2D histogram(60×60) 기준을 함께 계산했고, 2D는 boundary row를 포함한 full과 interior-only 두 버전을 계산했다. interior-only가 비교군 B의 연속 codebook 학습 대상 분포에 해당한다.

## Figure

- [kr_daily 병합](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-01-shape-feature-validation/merge-distribution-check/merge-check__kr_daily__s1-s2-density.png)
- [kr_1m 병합](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-01-shape-feature-validation/merge-distribution-check/merge-check__kr_1m__s1-s2-density.png)
- [kr_all 병합](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-01-shape-feature-validation/merge-distribution-check/merge-check__kr_all__s1-s2-density.png)

검은 점선이 병합 분포, 색 실선이 성분 dataset이다. y축은 density이므로 row 수 차이는 반영되지 않는다.

## 결과

JSD(성분, 병합):

| 병합 | 성분 | Row share | Boundary | s1 | s2 | 2D full | 2D interior |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kr_daily` | `d1_kospi_daily` | 56.2% | 27.4% | 0.054 | 0.042 | 0.175 | 0.197 |
| `kr_daily` | `d1_kosdaq_daily` | 43.8% | 24.0% | 0.063 | 0.049 | 0.201 | 0.218 |
| `kr_1m` | `d1_kospi_1m` | 50.0% | 73.1% | 0.046 | 0.044 | 0.084 | 0.096 |
| `kr_1m` | `d1_kosdaq_1m` | 50.0% | 81.0% | 0.054 | 0.052 | 0.105 | 0.172 |
| `kr_all` | `d1_kospi_daily` | 4.6% | 27.4% | 0.305 | 0.311 | 0.461 | 0.265 |
| `kr_all` | `d1_kosdaq_daily` | 3.6% | 24.0% | 0.358 | 0.341 | 0.522 | 0.342 |
| `kr_all` | `d1_kospi_1m` | 45.9% | 73.1% | 0.028 | 0.030 | 0.086 | 0.132 |
| `kr_all` | `d1_kosdaq_1m` | 45.9% | 81.0% | 0.080 | 0.084 | 0.143 | 0.214 |

성분 간 pairwise 2D JSD: KOSPI↔KOSDAQ daily 0.350(full)/0.386(interior), KOSPI↔KOSDAQ 1m 0.183/0.257, daily↔1m 조합은 0.47~0.60(full).

## 해석

1. **같은 interval끼리 병합은 분포를 유지한다.** `kr_daily`와 `kr_1m` 모두 marginal JSD가 0.03~0.06 수준이고 figure에서도 병합 곡선이 두 성분 사이를 그대로 지나간다. 어느 한쪽 지수의 분포가 병합에서 사라지지 않는다. D2의 `d2_kr-kospi-kosdaq_daily` 구성을 지지하는 결과다 (H1 1차 근거).
2. **daily+1m 전체 병합은 분포를 유지하지 못한다.** daily의 row share가 8.2%뿐이라 병합 분포가 사실상 1m 분포다. daily 성분의 JSD가 0.30~0.52로 치솟고, figure에서 병합 곡선(boundary spike density 약 1.3)이 daily의 중앙 볼록 분포와 완전히 다른 모양이 된다. daily와 1m은 boundary 비중부터 25% vs 77%로 근본적으로 다른 분포이므로, 이 병합에서 학습한 codebook은 daily 캔들을 제대로 대표하지 못한다. 로드맵의 "일봉과 분봉은 같은 실험 run 안에서 섞지 않는다" 규칙이 데이터로 재확인됐다.
3. **주의점**: KOSPI↔KOSDAQ daily의 interior 2D JSD 0.386은 marginal이 보여주는 것보다 큰 차이다(KOSDAQ의 s1 우측 치우침 등 2D 구조 차이). daily 병합 codebook을 학습할 때는 지수별 token 점유율을 나눠 확인해 한쪽 지수가 특정 token을 독점하는지 추적해야 한다. 1m 쪽은 boundary 비율 8%p 차이(73% vs 81%)가 병합 시 boundary token 빈도 해석에 영향을 준다.
4. **한계**: 1m은 약 1년치, daily는 29~36년치라 병합 비교가 regime coverage 차이를 포함한다. JSD 절대값은 bin 설정에 의존하므로 병합 간 상대 비교 용도로만 사용한다.

## 결론

- D2 방향(같은 interval, 같은 시장권 병합)은 진행해도 된다: `kr_daily` 병합, `kr_1m` 병합 모두 분포 유지.
- daily+1m 통합 codebook은 배제한다. interval별 codebook을 유지하고, 공유 여부는 interval 안에서만 실험한다.
