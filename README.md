# kairos

OHLC 캔들을 해석 가능한 discrete shape token으로 양자화하고, token sequence에서 다중 캔들 motif vocabulary를 학습하는 연구 프로젝트다.

핵심 질문:

> 절대 가격, 변동폭, 시장 단위가 달라도 캔들 내부 모양이 비슷하면 같은 shape token으로 묶을 수 있는가?
> 그리고 그 token sequence에는 우연 수준을 넘어 반복되고 예측에 유용한 multi-candle motif가 존재하는가?

## 연구 개요

캔들 하나를 가격 수준·변동폭과 분리된 2D shape core로 표현하고, 이를 discrete token으로 양자화한 뒤, token sequence에서 motif vocabulary를 학습한다.

```text
OHLC
  -> shape core (logit λ_o, logit λ_c)         # scale-invariant 캔들 내부 모양
  -> shape_token                                # k-means/GMM/VQ-VAE/FSQ 비교로 선택
  -> token sequence + side channels             # rel_range, gap은 별도 연속 채널
  -> multi-candle motif vocabulary              # BPE/Unigram/Sequitur + surrogate 검정
  -> 해석성 평가 + downstream 예측 유용성 평가
```

전체 계획은 Phase 0(데이터/leakage 프로토콜)부터 Phase 6(결론)까지 결정 게이트가 있는 로드맵으로 관리한다: [docs/candlestick-shape-quantization/08-research-roadmap.md](docs/candlestick-shape-quantization/08-research-roadmap.md)

데이터는 3단계로 확장한다.

```text
D1. 단일 지수 (KS11/KQ11/IXIC/GSPC/DJI 중 1개)   파이프라인·설계 확정
D2. 국가별 주요 지수 (미국 또는 한국)              같은 시장 내 일반화 검증
D3. 세계 주요 지수 (11개 universe)                cross-market transfer 검증
```

## 저장소 구조

```text
kairos/       연구 파이프라인 패키지
                data.py              캔들 필터링, 시간 기준 split
                features.py          shape core와 side-channel 추출
                model.py             VQ-VAE tokenizer
                train.py             tokenizer 학습
                shape_metrics.py     token utilization/consistency 지표
                sequence_metrics.py  token sequence 지표
tests/        leakage-sensitive 순수 함수 단위 테스트
notebooks/    실험 노트북과 run 산출물
                candlestick-shape-quantization/   노트북 소스 (00 -> 05 순서 실행)
                runs/                             실험 config/metrics/figures
docs/         연구 설계 문서 (아래 문서 안내 참조)
references/   참조 논문 PDF
AGENTS.md     에이전트/기여자 작업 규칙 (연구 규칙, leakage 규칙 포함)
```

## 시작하기

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요하다.

```bash
uv sync                          # 의존성 설치
uv run --with pytest pytest -q   # 단위 테스트
```

feature 추출 예시:

```python
from kairos.features import extract_features, extract_shape

shape = extract_shape(candle)          # 단일 캔들 -> ShapeFeatures (s1, s2, 경계 flag)
rows = extract_features(candles)       # 시계열 -> shape core + rel_range/gap 채널
```

실험 노트북은 `notebooks/candlestick-shape-quantization/`의 번호 순서대로 실행한다. run 산출물 규약은 [notebooks/AGENTS.md](notebooks/AGENTS.md)를 따른다.

## 문서 안내

연구 설계 문서는 `docs/candlestick-shape-quantization/`에 있다. 권장 읽기 순서:

1. [01. 연구 목적과 범위](docs/candlestick-shape-quantization/01-problem-and-research-scope.md)
2. [08. 연구 로드맵](docs/candlestick-shape-quantization/08-research-roadmap.md) — source of truth
3. [02. Shape Tokenizer 설계](docs/candlestick-shape-quantization/02-shape-tokenizer.md)
4. [00. VQ-VAE Tokenizer 전제 검증 계획](docs/candlestick-shape-quantization/00-vq-vae-tokenizer-validation-plan.md)
5. [07. 근거와 참고 문헌](docs/candlestick-shape-quantization/07-evidence-and-references.md)

## 연구 규칙 요약

자세한 내용은 [AGENTS.md](AGENTS.md)와 로드맵을 따른다.

- **Leakage 방지 최우선.** 시간 기준 split 확정 전에는 어떤 모델·scaler·tokenizer도 fit하지 않는다. trailing 통계(ATR, volume median)는 t-1까지만 사용한다.
- **Shape core는 `(logit λ_o, logit λ_c)` 2D.** body/wick 4D feature는 파생 리포팅 전용이다. `rel_range`/`gap`은 token에 섞지 않는 별도 채널이다.
- **결정 게이트.** VQ 계열이 k-means를 이기지 못하면 k-means를 채택하고, motif가 Markov-1 surrogate 검정을 통과하지 못하면 단일 token 경로로 진행한다.
- **실험 기록.** 모든 run은 `experiment_config.json`/`metrics.json`을 남기고, seed 3개 이상, 지수별·regime별 분리 보고를 기본으로 한다.

## 참조 논문

PDF는 `references/`에 있으며, 논문별 설계 연결은 [로드맵의 Reference 논문 매핑](docs/candlestick-shape-quantization/08-research-roadmap.md#reference-논문-매핑)을 따른다.

### 표현 학습과 양자화

| 논문 | 링크 | 역할 |
| --- | --- | --- |
| Dimension reduction of OHLC data based on pseudo-PCA | [arXiv:2103.16908](https://arxiv.org/abs/2103.16908) | 제약 없는 bijective OHLC 표현. shape core의 직접 근거 |
| Kronos: A Foundation Model for the Language of Financial Markets | [arXiv:2508.02739](https://arxiv.org/abs/2508.02739) · [code](https://github.com/shiyu-coder/Kronos) | K-line 전용 tokenizer(BSQ), coarse/fine 계층 토큰, 공개 pretrained baseline |
| Neural Discrete Representation Learning (VQ-VAE) | [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) | discrete codebook 학습의 원형 |
| TimeVQVAE | [arXiv:2303.04743](https://arxiv.org/abs/2303.04743) | coarse/fine codebook 분리, VQ 기반 시계열 생성 |
| TSAX: A Novel Trend Symbolic Aggregate Approximation | [arXiv:1905.00421](https://arxiv.org/abs/1905.00421) | 심볼화에서 방향/trend 보존의 근거, Phase 5 baseline |
| TS2Vec | [arXiv:2106.10466](https://arxiv.org/abs/2106.10466) | 양자화 전 continuous representation baseline |
| VQ-AR | [arXiv:2205.15894](https://arxiv.org/abs/2205.15894) | discrete representation 기반 forecasting 비교 프레임 |

### 캔들 패턴 인식과 해석성

| 논문 | 링크 | 역할 |
| --- | --- | --- |
| Encoding Candlesticks as Images (GAF-CNN) | [arXiv:1901.05237](https://arxiv.org/abs/1901.05237) | 8개 표준 패턴 분류 체계. 해석성 평가의 ground truth |
| Explainable Deep Convolutional Candlestick Learner | [arXiv:2001.02767](https://arxiv.org/abs/2001.02767) | perturbation 기반 shape 민감도 분석 방법론 |
| Dynamic Deep Convolutional Candlestick Learner | [arXiv:2201.08669](https://arxiv.org/abs/2201.08669) | variable-length 패턴 위치 탐지 관점 |
| DL + Candlestick Chart Representation | [arXiv:1903.12258](https://arxiv.org/abs/1903.12258) | 캔들 이미지 CNN 예측 baseline, window/volume ablation |
| Deep Stock Representation Learning | [arXiv:1709.03803](https://arxiv.org/abs/1709.03803) | 비지도 캔들 representation을 downstream 의사결정으로 연결 |

주의: arXiv `2508.02739`는 Kronos 논문이다. VLM candlestick 벤치마크로 오인하지 않는다.
