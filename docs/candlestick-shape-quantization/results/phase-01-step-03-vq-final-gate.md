# 03 VQ Tokenizer Final Gate 결과 해석

> 원본 run 위치: `notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets-multi/cfg-d59bafed/` — 이 문서의 figure/table 링크는 gitignore된 로컬 run 산출물을 가리키므로 로컬 체크아웃에서만 열린다.

이 문서는 `03_vq_tokenizer.ipynb`가 생성한 `cfg-d59bafed` 결과를 해석한다. 기준 run은 `run-20260707-032508_seed-multi`다.

관련 문서:

- [이전 VQ latent 결과](phase-01-step-03-vq-latent-clustering.md)
- [02 Tokenizer Baseline 결과 해석](phase-01-step-02-tokenizer-baselines.md)
- [Shape Feature Validation Figure 해석](phase-01-step-01-shape-feature-validation.md)

## 실험 목적

Phase 1 gate를 닫기 전에 roadmap에 남아 있던 VQ 계열 후보를 같은 boundary-aware wrapper에서 한 번 더 검증했다. 비교 질문은 다음이다.

> FSQ, BSQ, coarse-fine 중 하나라도 `kmeans_boundary_aware K=32` 대비 reconstruction, seed stability, effective vocabulary/token usage 중 하나 이상에서 개선을 보이는가?

비교 모델은 5개다.

- `kmeans_boundary_aware`: train split의 interior x interior row만 KMeans K=32로 fit하고, boundary 8개 조합과 zero-range는 rule-based token으로 배정한다.
- `vqvae_latent_kmeans`: autoencoder를 train interior row에 학습하고 encoder latent에 KMeans K=32를 적용한다.
- `fsq`: autoencoder latent_dim=2, tanh bound, levels `[6, 5]`, continuous capacity 30.
- `bsq`: autoencoder latent_dim=5, L2 normalize + sign quantization, continuous capacity 32.
- `coarse_fine`: direction 2 x body quartile 4 coarse class에 class별 KMeans K=4 fine token을 붙인 non-neural 후보, continuous capacity 32.

모든 후보는 boundary-aware vocabulary를 공유한다.

```text
continuous codebook K
+ boundary discrete token 8
+ zero-range special token 1
= total vocabulary size K + 9
```

## Dataset 처리

대상은 D1 daily 4개, D1 KR 1m 2개, D2 KR 2개다. 모든 dataset은 seeds `7`, `17`, `37`에서 5개 모델이 실행됐고, 모든 row가 token을 받았다.

| Dataset | Rows | Interior | Train interior | Boundary | Zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d1_kosdaq_1m` | 94,236 | 17,906 | 11,147 | 80.74% | 0.26% |
| `d1_kosdaq_daily` | 7,365 | 5,590 | 2,363 | 23.94% | 0.16% |
| `d1_kospi_1m` | 94,149 | 25,266 | 16,304 | 72.90% | 0.26% |
| `d1_kospi_daily` | 9,420 | 6,838 | 2,379 | 27.41% | 0.00% |
| `d1_nasdaq_daily` | 6,664 | 5,881 | 2,605 | 11.73% | 0.02% |
| `d1_spx_daily` | 5,302 | 4,022 | 2,004 | 24.14% | 0.00% |
| `d2_kr-kospi-kosdaq_1m` | 188,385 | 43,172 | 27,451 | 76.82% | 0.26% |
| `d2_kr-kospi-kosdaq_daily` | 16,785 | 12,428 | 4,742 | 25.89% | 0.07% |

1m dataset은 boundary ratio가 72.9~80.7%다. 따라서 boundary row를 continuous codebook에 넣으면 codebook 잠식이 발생한다는 step-02 결론은 유지된다.

## Reconstruction 비교

MSE는 interior-only `(s1, s2)` reconstruction 기준이며, 표는 seeds 평균이다. 낮을수록 좋다.

| Dataset | KMeans-B | VQ latent | FSQ | BSQ | Coarse-fine | Best candidate / KMeans |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d1_kosdaq_1m` | 0.244359 | 0.263963 | 0.305420 | 0.531253 | 0.325540 | VQ latent 1.080x |
| `d1_kosdaq_daily` | 0.508771 | 0.532782 | 0.707982 | 1.391458 | 0.652024 | VQ latent 1.047x |
| `d1_kospi_1m` | 0.339140 | 0.363502 | 0.448573 | 0.719813 | 0.510821 | VQ latent 1.072x |
| `d1_kospi_daily` | 0.472221 | 0.501368 | 0.687150 | 1.337003 | 0.703939 | VQ latent 1.062x |
| `d1_nasdaq_daily` | 0.452202 | 0.477233 | 0.658028 | 1.372050 | 0.619733 | VQ latent 1.055x |
| `d1_spx_daily` | 0.401378 | 0.421799 | 0.567238 | 1.101332 | 0.549213 | VQ latent 1.051x |
| `d2_kr-kospi-kosdaq_1m` | 0.310992 | 0.334947 | 0.418822 | 0.651688 | 0.457343 | VQ latent 1.077x |
| `d2_kr-kospi-kosdaq_daily` | 0.486715 | 0.509660 | 0.654215 | 1.233288 | 0.666557 | VQ latent 1.047x |

해석:

- reconstruction은 8개 dataset 전부에서 `kmeans_boundary_aware`가 최선이다.
- 신경망 후보 중에서는 `vqvae_latent_kmeans`가 가장 덜 나쁘지만, KMeans-B보다 4.7~8.0% 높다.
- `fsq`와 `bsq`는 capacity를 맞췄음에도 reconstruction 손실이 크다. 특히 BSQ는 auxiliary loss 없는 최소 구현에서 code usage와 reconstruction 모두 약하다.
- `coarse_fine`은 안정적이지만 coarse rule 때문에 reconstruction 해상도가 낮다.

## Seed 안정성

표는 seeds `7`, `17`, `37`의 MSE 표준편차다. 낮을수록 안정적이다.

| Dataset | KMeans-B std | VQ latent std | FSQ std | BSQ std | Coarse-fine std | Lowest std |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `d1_kosdaq_1m` | 0.001904 | 0.010123 | 0.002451 | 0.027665 | 0.000221 | Coarse-fine |
| `d1_kosdaq_daily` | 0.002698 | 0.016952 | 0.029466 | 0.082037 | 0.000466 | Coarse-fine |
| `d1_kospi_1m` | 0.000764 | 0.016206 | 0.004843 | 0.023902 | 0.000031 | Coarse-fine |
| `d1_kospi_daily` | 0.002208 | 0.018195 | 0.044839 | 0.056247 | 0.000363 | Coarse-fine |
| `d1_nasdaq_daily` | 0.000970 | 0.007226 | 0.052403 | 0.077096 | 0.000282 | Coarse-fine |
| `d1_spx_daily` | 0.001635 | 0.005846 | 0.039635 | 0.046512 | 0.000345 | Coarse-fine |
| `d2_kr-kospi-kosdaq_1m` | 0.000435 | 0.017853 | 0.004704 | 0.037985 | 0.000098 | Coarse-fine |
| `d2_kr-kospi-kosdaq_daily` | 0.004284 | 0.005247 | 0.016604 | 0.029994 | 0.000013 | Coarse-fine |

`coarse_fine`은 rule-based coarse split이 대부분의 구조를 고정하기 때문에 seed std가 가장 낮다. 다만 이 안정성은 reconstruction 비용을 지불하고 얻은 안정성이다.

## Token usage

표는 effective vocabulary size의 seeds 평균이다. 높을수록 token 분산이 넓다.

| Dataset | KMeans-B eff | VQ latent eff | FSQ eff | BSQ eff | Coarse-fine eff | Highest eff |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `d1_kosdaq_1m` | 13.87 | 13.83 | 13.70 | 12.93 | 13.92 | Coarse-fine |
| `d1_kosdaq_daily` | 30.96 | 30.33 | 26.75 | 18.93 | 31.15 | Coarse-fine |
| `d1_kospi_1m` | 16.28 | 16.20 | 16.02 | 14.74 | 16.53 | Coarse-fine |
| `d1_kospi_daily` | 30.86 | 30.93 | 27.19 | 20.25 | 32.24 | Coarse-fine |
| `d1_nasdaq_daily` | 33.02 | 32.39 | 27.88 | 19.24 | 33.76 | Coarse-fine |
| `d1_spx_daily` | 28.85 | 28.39 | 25.11 | 18.03 | 29.54 | Coarse-fine |
| `d2_kr-kospi-kosdaq_1m` | 15.10 | 15.03 | 14.83 | 14.15 | 15.29 | Coarse-fine |
| `d2_kr-kospi-kosdaq_daily` | 30.79 | 30.19 | 27.97 | 22.38 | 31.16 | Coarse-fine |

전체 평균:

| Model | MSE mean | Seed std mean | Effective vocab mean | Dead tokens mean | Boundary ratio | Zero ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kmeans_boundary_aware` | 0.401972 | 0.001862 | 24.97 | 1.00 | 42.95% | 0.13% |
| `vqvae_latent_kmeans` | 0.425657 | 0.012206 | 24.66 | 1.00 | 42.95% | 0.13% |
| `fsq` | 0.555928 | 0.024368 | 22.43 | 1.00 | 42.95% | 0.13% |
| `bsq` | 1.042236 | 0.047680 | 17.58 | 10.29 | 42.95% | 0.13% |
| `coarse_fine` | 0.560646 | 0.000227 | 25.45 | 1.00 | 42.95% | 0.13% |

`coarse_fine`은 effective vocab 평균이 가장 높지만, reconstruction 평균은 KMeans-B보다 약 39.5% 높다. 따라서 token usage 개선은 주 tokenizer 교체 근거로는 약하고, low-resolution sequence 후보로만 의미가 있다.

## Boundary 해석

모든 모델은 같은 boundary-aware wrapper를 공유한다.

```text
KMeans/VQ/BSQ/coarse_fine:
  token 0..31  : interior continuous codebook
  token 32..39 : boundary 8-cell discrete token
  token 40     : zero-range special token

FSQ:
  token 0..29  : interior continuous codebook
  token 30..37 : boundary 8-cell discrete token
  token 38     : zero-range special token
```

따라서 boundary/zero ratio는 모델 간 동일해야 정상이다. 이번 run에서 boundary 평균은 42.95%, zero 평균은 0.13%로 모든 모델에서 동일했다. boundary 보존은 VQ 계열의 성과가 아니라 wrapper의 성과다.

## D2 symbol 분해

seed 7 기준 D2 top token 요약이다.

| Dataset | Model | Symbol | Boundary share | Zero share | Top tokens seed 7 |
| --- | --- | --- | ---: | ---: | --- |
| `d2_kr-kospi-kosdaq_daily` | `kmeans_boundary_aware` | `KOSDAQ` | 23.94% | 0.16% | 33:6.2%, 35:5.6%, 36:5.5%, 38:4.7%, 5:4.7% |
| `d2_kr-kospi-kosdaq_daily` | `kmeans_boundary_aware` | `KOSPI` | 27.41% | 0.00% | 33:6.8%, 36:6.7%, 38:6.4%, 3:5.6%, 35:5.1% |
| `d2_kr-kospi-kosdaq_daily` | `coarse_fine` | `KOSDAQ` | 23.94% | 0.16% | 33:6.2%, 35:5.6%, 36:5.5%, 10:4.9%, 38:4.7% |
| `d2_kr-kospi-kosdaq_daily` | `coarse_fine` | `KOSPI` | 27.41% | 0.00% | 33:6.8%, 36:6.7%, 38:6.4%, 6:5.1%, 35:5.1% |
| `d2_kr-kospi-kosdaq_1m` | `kmeans_boundary_aware` | `KOSDAQ` | 80.74% | 0.26% | 33:14.8%, 38:13.7%, 35:13.3%, 37:12.9%, 36:12.7% |
| `d2_kr-kospi-kosdaq_1m` | `kmeans_boundary_aware` | `KOSPI` | 72.90% | 0.26% | 33:14.7%, 38:14.0%, 35:13.6%, 36:13.1%, 37:8.8% |
| `d2_kr-kospi-kosdaq_1m` | `coarse_fine` | `KOSDAQ` | 80.74% | 0.26% | 33:14.8%, 38:13.7%, 35:13.3%, 37:12.9%, 36:12.7% |
| `d2_kr-kospi-kosdaq_1m` | `coarse_fine` | `KOSPI` | 72.90% | 0.26% | 33:14.7%, 38:14.0%, 35:13.6%, 36:13.1%, 37:8.8% |

D2 1m은 top token 대부분이 boundary token이다. D2 daily는 continuous token 차이가 일부 보이지만, KOSPI/KOSDAQ 간 boundary share 차이도 여전히 해석에 중요하다.

## Figure 읽는 법

기준 run figure:

- [Reconstruction MSE](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets-multi/cfg-d59bafed/run-20260707-032508_seed-multi/figures/reconstruction_mse_by_dataset.png)
- [Effective vocab](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets-multi/cfg-d59bafed/run-20260707-032508_seed-multi/figures/effective_vocab_by_dataset.png)
- [Boundary/zero ratio](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets-multi/cfg-d59bafed/run-20260707-032508_seed-multi/figures/boundary_zero_ratio_by_dataset.png)
- [D2 symbol token share heatmap](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets-multi/cfg-d59bafed/run-20260707-032508_seed-multi/figures/d2_symbol_token_share_heatmap_seed7.png)
- [Seed MSE std](../../../notebooks/runs/candlestick-shape-quantization/phase-01-shape-tokenizer/step-03-vq-tokenizer/all-datasets-multi/cfg-d59bafed/run-20260707-032508_seed-multi/figures/seed_mse_std_by_dataset.png)

읽는 법:

- `reconstruction_mse_by_dataset.png`: 낮을수록 좋다. 모든 dataset에서 KMeans-B가 최선이다.
- `effective_vocab_by_dataset.png`: token usage 확산을 본다. coarse-fine이 가장 높지만 reconstruction 비용이 크다.
- `boundary_zero_ratio_by_dataset.png`: 모델별 막대가 같아야 정상이다. boundary/zero는 shared wrapper가 배정한다.
- `d2_symbol_token_share_heatmap_seed7.png`: D2의 symbol별 token 점유율을 본다. 1m은 boundary token 열이 지배적이다.
- `seed_mse_std_by_dataset.png`: seeds 간 reconstruction 변동을 본다. coarse-fine이 가장 안정적이고, 신경망 계열은 KMeans-B보다 흔들린다.

## Gate 판정

고정 스펙 timebox 기준 결론은 다음과 같다.

1. `vqvae_latent_kmeans`, `fsq`, `bsq`는 reconstruction, seed stability, token usage 중 의미 있는 개선을 보이지 못했다.
2. `coarse_fine`은 reconstruction에서는 KMeans-B에 크게 지지만, seed stability와 effective vocab에서는 모든 dataset에서 KMeans-B보다 좋다.
3. 따라서 "FSQ/BSQ/coarse-fine 중 어느 것도 한 지표도 개선하지 못하면 Phase 1 gate를 닫고 KMeans-B를 최종 채택한다"는 실패 조건은 충족되지 않는다.
4. 다만 shape reconstruction용 주 tokenizer는 여전히 `kmeans_boundary_aware K=32`가 가장 방어 가능하다.
5. `coarse_fine`은 최종 shape tokenizer로 채택하지 않고, Phase 3 motif 단계에서 low-resolution coarse token sequence 후보로 재평가한다.

실무 결정:

- Phase 1의 main tokenizer: `kmeans_boundary_aware K=32`.
- Phase 1의 rejected neural candidates: `vqvae_latent_kmeans`, `fsq`, `bsq`.
- Phase 3 carry-forward candidate: `coarse_fine` low-resolution sequence ablation.
- timebox 원칙에 따라 FSQ/BSQ/coarse-fine 조건을 바꾼 재도전은 하지 않는다.

## Caveats

- FSQ/BSQ는 최소 구현이며 entropy/commitment 보조 loss를 넣지 않았다.
- `coarse_fine`의 stability 우위는 신경망 학습 안정성이 아니라 rule-based coarse class 고정에서 온다.
- `coarse_fine`은 reconstruction MSE가 나쁘므로 shape atlas용 최종 tokenizer로 쓰기 어렵다.
- run artifacts(config/metrics/figures)는 `notebooks/runs/` 아래 로컬 산출물이며 gitignore 대상이다. 이 해석 문서만 저장소에 커밋된다.
