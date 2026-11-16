# 내 목소리를 구해줘! — 챌린지 제출 코드

- 팀명: `조도계`  / 제출 회차: `2회차` / 커밋: `(제출 커밋 해시 기입)` (1회차: `3a37108`)
- 방법: **전통적 신호 처리만 사용** (딥러닝·머신러닝 없음)
  — SPP 기반 잡음 PSD 추정 + 양방향 decision-directed SNR + MMSE-LSA 이득 + 하모닉 재생(HRNR)
  + **음성 바닥 보정** (2회차 추가: 잡음 추정에 섞인 연속 음성 성분 제거)

## 설치

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   /  macOS·Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12에서 테스트했습니다. `enhance.py` 실행에는 numpy, scipy, soundfile만 필요합니다
(pandas·pesq는 `eval_validation.py` 채점용).

## 실행

```bash
python enhance.py                                   # ./input/*.wav  ->  ./output/*.wav
python enhance.py --input <입력폴더> --output <출력폴더>
```

- `input` 폴더의 모든 `.wav`를 처리해 **같은 파일명**으로 `output`에 저장합니다.
- 출력은 입력과 **샘플링 주파수, 채널 수, 샘플 수, 샘플 형식(PCM_16 등)이 같습니다** (저장 후 assert로 확인).
- 다채널은 채널별로 처리하고, 16 kHz가 아닌 입력은 창/hop 길이를 같은 시간(64 ms / 8 ms)으로 맞춥니다.
- 절대 경로에 의존하지 않습니다 (기본 경로는 `enhance.py` 위치 기준).

## 알고리즘 (`denoiser.py`)

| 단계 | 내용 |
| --- | --- |
| 1. STFT | Hann 1024 샘플(64 ms), hop 128(8 ms) |
| 2. 잡음 초기 추정 | 파일에서 에너지가 가장 낮은 10% 프레임의 평균 파워 (**주파수 평활 없음**: 톤성 잡음 선을 보존) |
| 3. 오프라인 보정 | 음성 부재 확률 `1−P(H1)`로 가중한 파일 전체 평균 파워로 잡음 PSD 재추정 |
| 4. 잡음 추적 | SPP 기반 MMSE 잡음 추정 (Gerkmann & Hendriks 2012), 평활 계수 0.99 |
| 5. 사전 SNR | Decision-directed (α = 0.98)를 시간 순방향·역방향으로 각각 계산해 기하평균, ξ_min = −22 dB |
| 6. 1단계 이득 | MMSE log-spectral amplitude (Ephraim & Malah 1985) |
| 7. 하모닉 재생 (HRNR) | 1단계 출력을 반파 정류해 사라진 고조파를 되살리고, 사전 SNR을 다시 추정해 LSA 이득 재계산 (Plapous et al. 2006, ρ = 0.5) |
| 8. 무음 프레임 감쇠 | 추정 프레임 SNR이 낮은 프레임을 추가로 −3 dB 감쇠 |
| 9. 음성 바닥 보정 (2회차) | 3초짜리 연속 발화는 가장 조용한 순간에도 낮은 음성 성분이 남아 있어서, 잡음이 약한 대역에서는 이것이 잡음으로 추정됩니다. 1차 결과로 파일의 평균 음성 스펙트럼 S(f)를 구하고, 바닥 F = S·10^(r/10) (r = −19 dB, 300 Hz 아래 −15 dB)를 잡음 추정에서 뺍니다 (`N' = max(N − F, 0.1·N)`). 그다음 5–8단계를 다시 계산합니다. 잡음이 큰 대역에서는 N ≫ F라서 거의 변하지 않습니다 |
| 10. ISTFT | overlap-add, 입력과 같은 길이로 자름 |

Clean 음성은 잡음 제거에 전혀 쓰지 않고, 평가(`eval_validation.py`)에만 씁니다.
파라미터는 **Train 320개 부분집합**에서 골랐고, Validation은 결과 확인에만 썼습니다.

## Validation 결과 (164개 파일, PESQ-WB / SNR)

| 방법 | PESQ | SNR (dB) |
| --- | ---: | ---: |
| Noisy input | 1.395 | 9.63 |
| Basic filter (baseline) | 1.449 | 9.07 |
| Spectral subtraction (baseline) | 1.538 | 12.57 |
| Wiener filter (baseline) | 1.598 | 13.12 |
| 초기 버전 (SPP + MMSE-LSA, 미제출) | 1.834 | 14.46 |
| 1회차 제출본 (+ 톤 보존 잡음 추정, HRNR, 무음 감쇠) | 1.939 | 14.49 |
| **2회차 제출본 (+ 음성 바닥 보정)** | **1.981** | **14.68** |

목표 SNR별 PESQ (2회차): 0 dB 1.294 · 5 dB 1.755 · 10 dB 1.968 · 15 dB 2.330
(Train 전체 1296개: PESQ 1.920 → **1.976** / SNR 13.96 → 14.14)

2회차 변경 근거: 오라클 진단에서 잡음 추정의 **과대 추정**(특히 잡음이 약한 대역)이 가장 큰 손실이었고,
그 원인이 clean 음성 자체의 "조용한 순간에도 남는 음성 성분"임을 확인했습니다.
파라미터(r = −19 / −15 dB, 0.1)는 Train 320개에서 골랐고, 주변 값(±2 dB)에서도 Validation +0.033~+0.047로 안정적이었습니다.

재현:

```bash
python eval_validation.py --data <train·validation 폴더가 있는 데이터 경로>
```

결과 파일: `results/` (파일별 CSV, 잡음 종류별·SNR별 PESQ, 스펙트로그램 그림)

## 참고 문헌

- S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," IEEE TASSP 27(2), 1979.
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator," IEEE TASSP 32(6), 1984.
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," IEEE TASSP 33(2), 1985.
- T. Gerkmann, R. C. Hendriks, "Unbiased MMSE-based noise power estimation with low complexity and low tracking delay," IEEE TASLP 20(4), 2012.
- C. Plapous, C. Marro, P. Scalart, "Improved signal-to-noise ratio estimation for speech enhancement," IEEE TASLP 14(6), 2006.
