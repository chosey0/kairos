# 07. 근거와 참고 문헌

상위 문서:
- [Candlestick Shape Quantization](README.md)

연계 문서:
- [02. Shape Tokenizer 설계](02-shape-tokenizer.md)
- [08. 연구 로드맵](08-research-roadmap.md)

## 문서 역할

이 문서는 설계 판단의 근거를 정리한다. 각 문헌은 설계의 특정 결정과 연결된다.

## VQ-VAE와 Discrete Representation

참고:
- [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

연결된 설계:
- [02. Shape Tokenizer 설계](02-shape-tokenizer.md)

근거:
- VQ-VAE는 continuous encoder output을 discrete codebook entry로 양자화해 latent discrete representation을 학습한다.
- 이 연구의 `shape_token`은 VQ-VAE의 discrete latent code와 같은 역할을 한다.

설계상 해석:
- VQ-VAE를 쓰는 목적은 가격을 예측하기 위해서가 아니라, 반복되는 candle shape를 token vocabulary로 압축하기 위해서다.
- 다만 입력이 저차원이므로 k-means, GMM baseline과 비교해야 한다.

## Codebook Usage와 Collapse Risk

참고:
- [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/abs/2111.14822)
- [Representation Learning with Vector Quantized Variational Autoencoders](https://arxiv.org/abs/1711.00937)

연결된 설계:
- [02. Shape Tokenizer 설계](02-shape-tokenizer.md)
- [08. 연구 로드맵](08-research-roadmap.md)

근거:
- VQ 계열 모델은 codebook을 사용하는 구조이므로 실제 token 사용률을 확인해야 한다.
- 일부 token만 사용되면 discrete vocabulary가 충분히 학습되었다고 보기 어렵다.

설계상 해석:
- `dead_token_count`, `token_entropy`, `effective_vocab_size`를 필수 지표로 둔다.
- K를 고정하지 않고 `8, 12, 16, 24`를 비교한다.

## Time-Series Split과 Leakage 방지

참고:
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

연결된 설계:
- [08. 연구 로드맵](08-research-roadmap.md) Phase 0

근거:
- 시계열 데이터는 시간 순서가 있으며, 일반 random split은 미래 정보가 학습에 들어가는 leakage를 만들 수 있다.
- TimeSeriesSplit은 시간 순서를 보존하면서 train/test index를 나누는 cross-validator다.

설계상 해석:
- random split을 사용하지 않는다.
- tokenizer, scaler, prototype은 train split에서만 fit한다.
- validation/test는 시간상 train 이후 구간으로 둔다.

## Reference 논문과 설계 연결

`references/` 폴더의 논문별 설계 연결은 [08. 연구 로드맵](08-research-roadmap.md)의 Reference 논문 매핑을 따른다. 핵심 근거:

- pseudo-PCA(2103.16908): OHLC의 제약을 제거하는 bijective feature 표현. shape core `(logit λ_o, logit λ_c)`의 직접 근거.
- Kronos(2508.02739): K-line 전용 tokenizer(BSQ), coarse/fine 계층 토큰, autoregressive candle-LM, 공개 pretrained baseline.
- TimeVQVAE(2303.04743): coarse/fine codebook 분리와 VQ 기반 시계열 생성.
- TS2Vec(2106.10466): 양자화 전 continuous representation baseline.
- GAF-CNN 계열(1901.05237, 2001.02767, 2201.08669, 1903.12258): 전통 패턴 ground truth, 해석성 평가 방법론, 이미지 기반 baseline.

## 설계 판단 요약

| 설계 판단 | 근거 | 연결 문서 |
| --- | --- | --- |
| Shape token을 discrete latent로 둔다 | VQ-VAE의 discrete representation 학습 구조 | [02](02-shape-tokenizer.md) |
| 입력은 `(logit λ_o, logit λ_c)`를 기본으로 둔다 | 기존 4D feature가 독립적이지 않음, pseudo-PCA의 constraint-free 표현 | [02](02-shape-tokenizer.md), [08](08-research-roadmap.md) |
| k-means/GMM baseline을 둔다 | 입력이 저차원이므로 VQ-VAE 복잡도 검증 필요 | [02](02-shape-tokenizer.md), [00](00-vq-vae-tokenizer-validation-plan.md) |
| position/range는 continuous side-channel로 둔다 | transition matrix만으로 연속 정보를 표현하면 정보 손실이 큼 | [08](08-research-roadmap.md) |
| random split을 금지한다 | 시계열 leakage 방지 | [08](08-research-roadmap.md) Phase 0 |
| token utilization을 필수 평가한다 | codebook collapse 감시 | [02](02-shape-tokenizer.md), [00](00-vq-vae-tokenizer-validation-plan.md) |
| motif는 surrogate 검정을 통과해야 한다 | 빈도 기반 motif의 우연 생성 방지 | [08](08-research-roadmap.md) Phase 3 |

