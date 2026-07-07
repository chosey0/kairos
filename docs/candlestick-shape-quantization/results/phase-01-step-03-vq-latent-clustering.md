# 03 VQ-VAE Latent Clustering 결과 해석

> 원본 run 위치: `notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets/cfg-9b7586c3/` — 이 문서의 figure/table 링크는 gitignore된 로컬 run 산출물을 가리키므로 로컬 체크아웃에서만 열린다.

이 문서는 `03_vq_tokenizer.ipynb`가 생성한 `cfg-9b7586c3` 결과를 해석한다. 기준 run은 figure 생성까지 포함된 최신 실행인 `run-20260707-025655_seed-multi`다.

관련 문서:

- [02 Tokenizer Baseline 결과 해석](phase-01-step-02-tokenizer-baselines.md)
- [Shape Feature Validation Figure 해석](phase-01-step-01-shape-feature-validation.md)

## 실험 목적

step-02 이후 확정한 boundary 분리 방식이 실제 tokenizer 비교에서 작동하는지 확인했다. 비교 질문은 다음이다.

> VQ-VAE Latent Clustering이 `kmeans K=32 include_boundary`보다 boundary를 보존하면서도 reconstruction, stability, token usage 중 하나 이상에서 이기는가?

이번 run에서는 이 질문을 더 엄격하게 보기 위해 `include_boundary`가 아니라 새 비교군 B를 직접 구현했다.

- `kmeans_boundary_aware`: interior x interior row만 KMeans K=32로 fit하고, boundary 8개 조합과 zero-range는 discrete token으로 직접 배정한다.
- `vqvae_latent_kmeans`: autoencoder를 train split의 interior x interior row에만 학습하고, encoder latent에 KMeans K=32를 적용한다. boundary와 zero-range token 배정은 비교군 B와 동일하다.

전체 vocabulary는 두 모델 모두 같다.

```text
continuous codebook K=32
+ boundary discrete token 8
+ zero-range special token 1
= total vocabulary size 41
```

따라서 boundary token 점유율은 두 모델에서 동일해야 한다. 차이는 interior row를 2D shape space에서 직접 clustering하느냐, autoencoder latent로 보낸 뒤 clustering하느냐에 있다.

## Dataset 처리 결과

대상은 결정 #6의 8개 dataset이다.

| Dataset | Rows | Interior rows | Train interior rows | Boundary ratio | Zero ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d1_kosdaq_1m` | 94,236 | 17,906 | 11,147 | 80.74% | 0.26% |
| `d1_kosdaq_daily` | 7,365 | 5,590 | 2,363 | 23.94% | 0.16% |
| `d1_kospi_1m` | 94,149 | 25,266 | 16,304 | 72.90% | 0.26% |
| `d1_kospi_daily` | 9,420 | 6,838 | 2,379 | 27.41% | 0.00% |
| `d1_nasdaq_daily` | 6,664 | 5,881 | 2,605 | 11.73% | 0.02% |
| `d1_spx_daily` | 5,302 | 4,022 | 2,004 | 24.14% | 0.00% |
| `d2_kr-kospi-kosdaq_1m` | 188,385 | 43,172 | 27,451 | 76.82% | 0.26% |
| `d2_kr-kospi-kosdaq_daily` | 16,785 | 12,428 | 4,742 | 25.89% | 0.07% |

해석:

- 8개 dataset 모두 실행됐다.
- 1m dataset은 boundary ratio가 72.9~80.7%로 매우 높다. step-02에서 확인한 codebook 잠식 문제가 그대로 재확인된다.
- 이번 run에서는 boundary를 제거하지 않고 discrete token으로 보존한다. 따라서 1m에서도 모든 row가 token을 받으며, continuous codebook은 interior 분포만 학습한다.
- zero-range는 전체적으로 낮지만 special token으로 분리되어 누락되지 않는다.

## Reconstruction 비교

아래 표는 seeds `7`, `17`, `37` 평균이다. MSE는 interior-only `(s1, s2)` reconstruction 기준이다.

| Dataset | KMeans-B MSE | VQ latent MSE | VQ/KMeans | KMeans eff vocab | VQ eff vocab | Boundary ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d1_kosdaq_1m` | 0.244359 | 0.263963 | 1.080x | 13.87 | 13.83 | 80.74% |
| `d1_kosdaq_daily` | 0.508771 | 0.532782 | 1.047x | 30.96 | 30.33 | 23.94% |
| `d1_kospi_1m` | 0.339140 | 0.363502 | 1.072x | 16.28 | 16.20 | 72.90% |
| `d1_kospi_daily` | 0.472221 | 0.501368 | 1.062x | 30.86 | 30.93 | 27.41% |
| `d1_nasdaq_daily` | 0.452202 | 0.477233 | 1.055x | 33.02 | 32.39 | 11.73% |
| `d1_spx_daily` | 0.401378 | 0.421799 | 1.051x | 28.85 | 28.39 | 24.14% |
| `d2_kr-kospi-kosdaq_1m` | 0.310992 | 0.334947 | 1.077x | 15.10 | 15.03 | 76.82% |
| `d2_kr-kospi-kosdaq_daily` | 0.486715 | 0.509660 | 1.047x | 30.79 | 30.19 | 25.89% |

핵심 결론:

- VQ latent clustering은 8개 dataset 전부에서 `kmeans_boundary_aware`보다 reconstruction MSE가 높다.
- VQ/KMeans MSE 비율은 1.047~1.080x이며, 평균은 약 1.061x다. 즉 현재 설정에서는 VQ latent가 KMeans-B보다 4.7~8.0% 나쁘다.
- effective vocab size도 개선되지 않았다. `d1_kospi_daily`만 아주 작게 높고(+0.07), 나머지는 같거나 낮다.
- dead token count는 두 모델이 동일하다. daily 일부에서 boundary/zero token 중 미사용 token이 생기지만, 모델 간 차이는 아니다.

따라서 현재 최소 VQ-VAE latent clustering은 reconstruction 또는 token usage 기준으로 KMeans-B를 이기지 못한다.

## Seed 안정성

| Dataset | KMeans MSE std | VQ MSE std | Std ratio | KMeans dead | VQ dead |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d1_kosdaq_1m` | 0.001904 | 0.010123 | 5.3x | 0.0 | 0.0 |
| `d1_kosdaq_daily` | 0.002698 | 0.016952 | 6.3x | 0.0 | 0.0 |
| `d1_kospi_1m` | 0.000764 | 0.016206 | 21.2x | 0.0 | 0.0 |
| `d1_kospi_daily` | 0.002208 | 0.018195 | 8.2x | 3.0 | 3.0 |
| `d1_nasdaq_daily` | 0.000970 | 0.007226 | 7.4x | 2.0 | 2.0 |
| `d1_spx_daily` | 0.001635 | 0.005846 | 3.6x | 3.0 | 3.0 |
| `d2_kr-kospi-kosdaq_1m` | 0.000435 | 0.017853 | 41.1x | 0.0 | 0.0 |
| `d2_kr-kospi-kosdaq_daily` | 0.004284 | 0.005247 | 1.2x | 0.0 | 0.0 |

해석:

- KMeans-B는 seed 간 MSE 변동이 매우 작다.
- VQ latent는 모든 dataset에서 seed 표준편차가 더 크다. 특히 1m 병합 dataset은 VQ std가 KMeans의 41배 수준이다.
- 안정성 기준에서도 현재 VQ 설정은 KMeans-B보다 약하다.

## Boundary 보존 해석

이번 run의 boundary 보존은 두 모델이 같은 rule-based discrete token layer를 공유하기 때문에 동일하다.

```text
token 0..31  : interior continuous codebook
token 32..39 : boundary 8-cell discrete token
token 40     : zero-range special token
```

따라서 VQ latent가 boundary를 더 잘 보존했다고 볼 근거는 없다. boundary는 VQ가 학습한 것이 아니라 tokenizer wrapper가 보존한 것이다.

이 설계는 step-02의 문제를 해결한다는 점에서 중요하다.

- 이전 `exclude_boundary` 근사는 boundary row가 token을 받지 못했다.
- 이번 KMeans-B/VQ-B는 boundary row도 모두 token을 받는다.
- continuous K=32는 interior shape에만 사용되므로, 1m boundary point mass가 연속 codebook을 잠식하지 않는다.

즉 boundary 문제의 해결책은 VQ latent가 아니라 **boundary-aware vocabulary 조립**이다.

## D2 지수별 token 점유율

아래는 seed 7 기준 D2 dataset의 symbol별 token 점유율 요약이다. `Top tokens`는 각 symbol에서 점유율이 높은 token 상위 5개다.

| Dataset | Model | Symbol | Boundary share | Zero share | Top tokens seed 7 |
| --- | --- | --- | ---: | ---: | --- |
| `d2_kr-kospi-kosdaq_daily` | `kmeans_boundary_aware` | `KOSPI` | 27.41% | 0.00% | 33:6.8%, 36:6.7%, 38:6.4%, 3:5.6%, 35:5.1% |
| `d2_kr-kospi-kosdaq_daily` | `kmeans_boundary_aware` | `KOSDAQ` | 23.94% | 0.16% | 33:6.2%, 35:5.6%, 36:5.5%, 38:4.7%, 5:4.7% |
| `d2_kr-kospi-kosdaq_daily` | `vqvae_latent_kmeans` | `KOSPI` | 27.41% | 0.00% | 30:7.0%, 33:6.8%, 0:6.8%, 36:6.7%, 38:6.4% |
| `d2_kr-kospi-kosdaq_daily` | `vqvae_latent_kmeans` | `KOSDAQ` | 23.94% | 0.16% | 33:6.2%, 35:5.6%, 36:5.5%, 30:5.3%, 0:5.2% |
| `d2_kr-kospi-kosdaq_1m` | `kmeans_boundary_aware` | `KOSDAQ` | 80.74% | 0.26% | 33:14.8%, 38:13.7%, 35:13.3%, 37:12.9%, 36:12.7% |
| `d2_kr-kospi-kosdaq_1m` | `kmeans_boundary_aware` | `KOSPI` | 72.90% | 0.26% | 33:14.7%, 38:14.0%, 35:13.6%, 36:13.1%, 37:8.8% |
| `d2_kr-kospi-kosdaq_1m` | `vqvae_latent_kmeans` | `KOSDAQ` | 80.74% | 0.26% | 33:14.8%, 38:13.7%, 35:13.3%, 37:12.9%, 36:12.7% |
| `d2_kr-kospi-kosdaq_1m` | `vqvae_latent_kmeans` | `KOSPI` | 72.90% | 0.26% | 33:14.7%, 38:14.0%, 35:13.6%, 36:13.1%, 37:8.8% |

해석:

- D2 daily에서 KOSPI와 KOSDAQ의 boundary share는 27.4% vs 23.9%로 비슷하다.
- D2 1m에서는 KOSDAQ의 boundary share가 KOSPI보다 높다(80.7% vs 72.9%). 이 차이는 token usage 해석에서 계속 분리해서 봐야 한다.
- 1m의 top token은 대부분 boundary token 33, 35, 36, 37, 38이다. 이는 boundary 분리 없이는 연속 codebook이 극단점에 잠식될 수밖에 없다는 step-02 결론을 강화한다.
- VQ와 KMeans의 D2 1m top boundary token은 동일하다. boundary layer가 rule-based로 공유되기 때문이다.

## Figure 읽는 법

기준 run figure:

- [Interior reconstruction MSE](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets/cfg-9b7586c3/run-20260707-025655_seed-multi/figures/interior_reconstruction_mse_by_dataset.png)
- [Boundary token ratio](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets/cfg-9b7586c3/run-20260707-025655_seed-multi/figures/boundary_token_ratio_by_dataset.png)
- [Effective vocab size](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets/cfg-9b7586c3/run-20260707-025655_seed-multi/figures/effective_vocab_size_by_dataset.png)
- [D2 token share by symbol](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets/cfg-9b7586c3/run-20260707-025655_seed-multi/figures/d2_token_share_by_symbol_seed-7.png)

읽는 법:

- `interior_reconstruction_mse_by_dataset.png`: 낮을수록 좋다. 모든 dataset에서 KMeans-B가 VQ latent보다 낮다.
- `boundary_token_ratio_by_dataset.png`: 두 모델 막대가 같아야 정상이다. boundary 배정은 모델 학습 결과가 아니라 fixed vocabulary rule이기 때문이다.
- `effective_vocab_size_by_dataset.png`: VQ가 token usage를 개선하는지 보는 figure다. 이번 run에서는 의미 있는 개선이 없다.
- `d2_token_share_by_symbol_seed-7.png`: D2 병합 dataset에서 KOSPI/KOSDAQ이 token을 얼마나 다르게 쓰는지 보는 heatmap이다. 흰 vertical guide 기준으로 `0..31`은 continuous token, `32..39`는 boundary token, `40`은 zero-range token이다.

## 결정

현재 최소 VQ-VAE latent clustering 설정은 채택하지 않는다.

근거:

1. Reconstruction: 8개 dataset 전부에서 KMeans-B보다 MSE가 높다.
2. Stability: seed 간 MSE 변동이 KMeans-B보다 크다.
3. Token usage: effective vocab size 개선이 없다.
4. Boundary: boundary 보존은 VQ의 성과가 아니라 shared boundary-aware vocabulary wrapper의 성과다.

따라서 Phase 1의 현재 주 baseline은 `kmeans_boundary_aware K=32`로 둔다.

## 다음 단계

VQ 계열을 완전히 폐기하기 전에 한 번만 더 검증할 수 있는 최소 변형은 다음이다.

- autoencoder 학습 epoch/latent_dim sweep을 작게 수행한다. 예: latent_dim 2/4/8, epochs 80/200.
- VQ 자체 quantizer를 사용하는 모델과 latent KMeans를 분리해서 비교한다.
- reconstruction만 보지 말고 latent prototype stability 또는 downstream motif stability가 개선되는지 확인한다.

단, 이번 run 기준으로는 VQ latent가 KMeans-B를 이긴 지표가 없으므로, 다음 연구 진행은 KMeans-B를 기본 tokenizer로 두고 motif vocabulary 단계로 넘어가는 것이 더 합리적이다.

## Caveats

- 이 run의 VQ는 최소 구현이다. hyperparameter tuning을 충분히 하지 않았다.
- autoencoder는 `(s1, s2)`만 입력으로 사용한다. side channel은 넣지 않았다.
- figure와 metrics는 최신 run `run-20260707-025655_seed-multi` 기준이다. 같은 cfg 아래 이전 run 중 `run-20260707-025058_seed-multi`는 figure 생성 전 실행이다.
- run artifacts(config/metrics/figures)는 `notebooks/runs/` 아래 로컬 산출물이며 gitignore 대상이다. 이 해석 문서만 저장소에 커밋된다.
