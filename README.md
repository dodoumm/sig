# 내 목소리를 구해줘! — 챌린지 제출 코드

- 팀명: `조도계` / 제출 회차: `3회차` / 커밋: `(제출 커밋 해시 기입)` (1회차: `3a37108`)
- 방법: **전통적 신호 처리만 사용** (딥러닝·머신러닝 없음)
  — SPP 기반 잡음 PSD 추정 + 양방향 decision-directed SNR + MMSE-LSA 이득 + 하모닉 재생(HRNR)
  + 음성 바닥 보정 (2회차) + **고정 대역 가중과 출력단 처리 (3회차: 저역 가중, 고역 shelf, 조건부 시간 게이트, 출력 이득)**

> 저역 가중, 고역 shelf, 시간 게이트, 출력 이득를 추가
> 이전 버전 위에 같은 구조로 다시 구현하고, Validation에서 다시 검증했습니다.


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
| 6. 이득 + 하모닉 재생 | MMSE log-spectral amplitude (Ephraim & Malah 1985), HRNR (Plapous et al. 2006, ρ = 0.5), 무음 프레임 −3 dB |
| 7. 음성 바닥 보정 (2회차) | 1차 결과의 평균 음성 스펙트럼 S(f)보다 −19 dB(300 Hz 아래 −15 dB) 낮은 바닥을 잡음 추정에서 빼고 (`N' = max(N − F, 0.1·N)`) 5–6단계를 다시 계산 |
| **8. 저역 가중 (3회차)** | 이득에 400 Hz·6차 Butterworth 형태의 고역통과 크기 `1/√(1+(400/f)^12)`를 곱함. 가중 자체의 감쇠는 최대 −20 dB |
| 9. ISTFT | overlap-add, 입력과 같은 길이로 자름 |
| **10a. 고역 shelf (3회차)** | 10차 Butterworth 저역통과(625 Hz)를 `sosfiltfilt`로 정·역방향 적용한 L(y)에 대해 `y' = L(y) + 10^(−29/20)·(y − L(y))`. 625 Hz보다 충분히 높은 성분은 약 29 dB 낮아짐 (영위상) |
| **10b. 조건부 시간 게이트 (3회차)** | 출력의 40 ms RMS가 파일 자신의 20백분위보다 6 dB 이하인 구간에 이득 0.65, 6–12 dB 사이에서 1까지 선형 증가, 60 ms 평활. 단, Noisy의 잡음 PSD(2단계)에서 1 kHz 미만 상위 5개 bin이 그 대역 전력의 60% 이상이면(전화 다이얼처럼 강한 협대역 톤) 게이트를 생략 |
| **10c. 출력 이득 (3회차)** | 최대 1.5배, 피크가 0.999를 넘지 않도록 제한 |

Clean 음성은 잡음 제거에 전혀 쓰지 않고, 평가(`eval_validation.py`)에만 씁니다.
1–7단계 파라미터는 Train 320개 부분집합에서 골랐습니다. 8·10단계 상수는 참고 코드의 값을 그대로 썼고(참고 코드에서 공개 Train/Validation으로 선택된 값), 우리는 Validation으로 효과를 다시 확인했습니다.

## Validation 결과 (164개 파일, PESQ-WB / SNR)

3회차부터는 `enhance.py`가 실제로 저장하는 것과 같은 형식(PCM_16 WAV)으로 저장·재읽기한 출력을 채점합니다.

| 방법 | PESQ | SNR (dB) |
| --- | ---: | ---: |
| Noisy input | 1.395 | 9.63 |
| Wiener filter (baseline) | 1.598 | 13.12 |
| 1회차 제출본 | 1.939 | 14.49 |
| 2회차 제출본 (+ 음성 바닥 보정) | 1.981 | 14.68 |
| (분석) 2회차 + 고역 shelf만 | 2.447 | 9.53 |
| **3회차 제출본 (+ 저역 가중, 고역 shelf, 시간 게이트, 출력 이득)** | **2.839** | **2.54** |

목표 SNR별 PESQ (3회차): 0 dB 2.024 · 5 dB 2.476 · 10 dB 2.915 · 15 dB 3.229
파일별 (2회차 대비): 164개 중 향상 162개, 하락 2개 (최대 하락 −0.10). 23개 잡음 종류 평균이 모두 상승 (최소 전화 다이얼 +0.31, 바람 +0.54)

**3회차 변경 근거:** 잡음 추정이 아니라 고정 대역 처리에서 나왔습니다(참고 코드의 후처리 전 단계는 2.057). 그래서 우리 2회차 알고리즘에 같은 후처리를 붙여 효과를 분리해 보았습니다. 고역 shelf만으로 +0.47, 저역 가중을 함께 써야 2.8점대가 되었고, 게이트와 출력 이득의 기여는 작았습니다.

**한계:** PESQ는 크게 올랐지만 파형 SNR은 14.68 dB에서 2.54 dB로 떨어져 Noisy 입력(9.63 dB)보다도 낮습니다. 625 Hz 위 성분(자음, 고역 포먼트)이 크게 약해지기 때문입니다. PESQ가 "더해진 잡음"을 "빠진 음성"보다 크게 감점한다는 점(2회차 진단에서 확인한 과소·과대 추정의 비대칭)과 같은 방향의 결과로 해석하지만, 청취 품질이 좋아졌다는 뜻은 아닙니다. 고정 대역 처리이므로 히든 테스트에서도 비슷하게 작동할 것으로 기대하지만, 보장되지는 않습니다.

재현:

```bash
python eval_validation.py --data <train·validation 폴더가 있는 데이터 경로>
```

결과 파일: `results/` (3회차 파일별 CSV, 요약, 잡음 종류별·SNR별 PESQ). 2회차 결과와 스펙트로그램은 `results/2회차_v3/`

## 참고 문헌

- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator," IEEE TASSP 32(6), 1984.
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," IEEE TASSP 33(2), 1985.
- T. Gerkmann, R. C. Hendriks, "Unbiased MMSE-based noise power estimation with low complexity and low tracking delay," IEEE TASLP 20(4), 2012.
- C. Plapous, C. Marro, P. Scalart, "Improved signal-to-noise ratio estimation for speech enhancement," IEEE TASLP 14(6), 2006.
- SciPy 문서: [`scipy.signal.butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html), [`scipy.signal.sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html) (3회차 대역 처리 구현)
- 외부 코드: 다른 팀 공유 참고 코드 `denoise_dsp_PESQ_2.8226` (원 제작 팀 `(팀명 기입)`) — 8·10단계 후처리 구조와 상수
