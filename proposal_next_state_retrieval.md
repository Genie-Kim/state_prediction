# 공정 이해 기반 Next-State 이미지 리트리벌 모델
### Process-Understanding via Next-State Image Retrieval — 연구 제안서

---

## 요약 (Abstract)

- **문제**: 공정 데이터에서 "공정을 이해했는가"를 정의·측정할 수 있는 학습 목표가 없음.
- **정의**: *공정 이해 = 공정에 의해 변하는 상태(state)의 전이를 예측하는 능력.* 좋은 state 표현은 다음 상태 예측에 **충분(sufficient)** 해야 함.
- **관찰**: 공정 이미지는 자연어로 표현이 어려움 → 텍스트 기반 state는 (1) public 도메인과 gap이 커 매칭이 불분명, (2) 텍스트-매칭 metric의 평가가 불확실.
- **제안**: state를 **latent image-feature** 로 추상화하고, MLLM(Qwen-VL)이 재사용 special token `<emb>` 로 임의의 중간 상태 latent를 예측 → 리트리벌.
- **검증**: **frozen domain-DINOv3** 공간에서 InfoNCE로 정렬, **Recall@K + anchor-ablation** 으로 전이 이해를 정량 평가.
- **근거 논문**: 임베딩-as-토큰(CIR-CoT), 다중 기능 토큰 interleaving(LIRA), 구조화 멀티모달 출력(M2IO-R1).

---

## 1. 배경 및 동기 (Motivation)

### 1.1 논리 전개 (claim → 근거 → 함의)

| # | 명제 | 근거 | 함의 |
|---|------|------|------|
| P1 | 공정 이해 = 공정에 의해 변하는 **상태**를 예측 | 공정은 상태 전이의 연쇄 | 학습·평가 대상은 **state** |
| P2 | state는 다음 상태 예측에 **충분한 정보**를 담아야 함 | Markov sufficiency | 표현의 정보 손실 최소화가 목표 |
| P3 | 공정 이미지는 **자연어로 표현이 어려움** | 미세 결함·텍스처·형상/색 변화 | 텍스트 state는 정보 병목 |
| P4 | 텍스트 state는 **도메인 gap + 평가 모호** | 공정 데이터 ≠ public 코퍼스 | 텍스트-매칭 metric 신뢰 저하 |
| C | ⇒ state를 **latent feature로 추상화** (이미지 인코더 벡터) | P2–P4의 결론 | **retrieval**로 학습·평가 가능 |

### 1.2 텍스트 state vs latent(image-feature) state

| 축 | 텍스트 state | latent state (본 제안) |
|----|--------------|------------------------|
| 미세 변화 표현력 | 낮음 (언어화 손실) | 높음 (연속 벡터) |
| 도메인 적합성 | 낮음 (public 편향) | 높음 (도메인 SSL 파인튜닝) |
| 매칭 명확성 | 모호 (동의어/서술 다양) | 명확 (벡터 유사도) |
| 평가 가능성 | 불확실 (텍스트 metric) | 정량적 (Recall@K, mAP) |
| 필요 라벨 | 상태별 정밀 캡션 | 상태별 이미지(존재하는 데이터) |

### 1.3 형식적 정의 — State Sufficiency

공정 궤적:
$$ s_0 \xrightarrow{a_1} s_1 \xrightarrow{a_2} \cdots \xrightarrow{a_T} s_T $$

- $s_k$: $k$-단계 상태 (실제로는 관측 이미지 $x_k$ 로만 접근 가능)
- $a_k$: $k$-번째 공정 스텝 (텍스트/조건)

**Markov 충분성** — 좋은 state 표현이 만족해야 할 조건:
$$ p(s_{k+1}\mid s_k, a_{k+1}) \;=\; p(s_{k+1}\mid s_{\le k}, a_{\le k}) $$

**본 제안의 추상화** — state를 이미지 인코더 latent로 정의:
$$ z_k \;:=\; \mathrm{DINOv3}(x_k)\in\mathbb{R}^d \quad(\text{latent state}) $$

**Retrieval = 충분성의 대리 검증 (proxy test):**

- 모델은 $(x_T,\; a_{1:n},\; \text{slot at } k)$ 로부터 $\hat z_k$ 를 예측 (= `<emb>` hidden → projection).
- $\hat z_k \approx z_k$ (= 정답 중간 이미지 검색 성공) ⇒ 모델이 전이를 이해한다는 **관측 가능한 증거**.
- 검증이 **텍스트가 아닌 latent 공간**에서 이뤄지므로 P4의 평가 모호성을 회피.

> 비고: 본 과제는 최종 상태 $x_T$ 로부터 **임의 중간 상태**를 복원하므로 forward 전이 이해와 inverse(과거 복원) 이해를 함께 요구한다. 이는 "next-state" 이해의 일반화된 형태다.

---

## 2. 목표 · 가설 · 연구질문

**Objectives**
- **O1.** 공정 이해를 latent next/intermediate-state **retrieval** 문제로 정식화.
- **O2.** 텍스트 state의 도메인 gap·평가 모호성을 image-feature latent state로 제거.
- **O3.** 단일 재사용 `<emb>` 기반 **멀티슬롯 embedding-as-token** 을 Qwen-VL에 구현·검증.

**Hypotheses**
- **H1.** 도메인 SSL로 파인튜닝한 DINOv3 공간은 공정 상태를 미세하게 분리한다 (kNN/t-SNE로 사전 검증 가능).
- **H2.** MLLM이 앵커 $x_T$ + 스텝 문맥으로 특정 인스턴스의 중간 상태 latent를 예측할 수 있다.
- **H3.** "같은 스텝 다른 인스턴스"를 하드 네거티브로 두면, 모델은 step-type가 아니라 **인스턴스**를 구분하며 이 판별 정보는 앵커에서 온다.

**Research Questions**
- **RQ1.** latent state 공간으로 DINOv3(frozen)와 Qwen 비전 인코더 중 무엇이 우수한가?
- **RQ2.** 다중 슬롯에서 표현 붕괴/순서 무시를 무엇이 막는가?
- **RQ3.** 앵커 $x_T$ 제거 시 성능 저하 폭이 전이 이해의 척도가 되는가?

---

## 3. 과제 정의 (Task Formulation)

### 3.1 표기 (Notation)

| 기호 | 의미 |
|------|------|
| $x_T$ | timestep $T$ 의 공정 이미지 (앵커, 입력) |
| $a_{1:n}$ | 거쳐온 $n$ 개 공정 스텝 설명 ($T>n$) |
| slot $k$ | 검색을 원하는 지점 표시(입력에 `<slot>`) |
| $x^{(k)}$ | slot $k$ 에 대응하는 GT 중간 상태 이미지 |
| $z^{(k)}=\mathrm{DINOv3}(x^{(k)})$ | slot $k$ 의 정답 latent (key) |
| `<emb>` | 출력에서 각 slot의 쿼리를 담는 재사용 토큰 |
| $q_k$ | slot $k$ 쿼리 벡터 (`<emb>` hidden → proj) |

### 3.2 입출력 정의

- **입력**: 앵커 $x_T$ + 지시문 + `a_{1:n}` 에 `<slot>` 이 삽입된 시퀀스.
- **출력**: 스텝 나열을 복사하되 `<slot>` 자리에 `<emb>` 배치(= teacher-structured). 각 `<emb>` 의 hidden state = 해당 slot의 쿼리.
- **리트리벌**: 갤러리(전체 중간 상태 이미지의 DINOv3 latent)에서 $q_k$ 의 top-$K$ 이웃.

### 3.3 리트리벌 구성요소

| 구성 | 정의 |
|------|------|
| Gallery | 전 데이터 중간 상태 이미지의 DINOv3(frozen) latent 집합(사전 인덱싱) |
| Positive | 해당 slot의 GT 중간 이미지 $x^{(k)}$ |
| Query | 해당 `<emb>` 의 last-layer hidden → projection |

---

## 4. 제안 방법 (Method)

### 4.1 아키텍처 개요

```
              ┌──────────────── QUERY TOWER (학습) ───────────────┐
 x_T, a_1:n → │ Qwen-VL(prompt + response w/ <emb>)                │
 (+<slot>)    │   → hidden@<emb>  → q_proj(MLP) → L2norm → q_k     │
              └────────────────────────────┬──────────────────────┘
                                            │  InfoNCE (τ)
              ┌────────────── GALLERY TOWER (frozen) ─────────────┐
   x^(k) ───→ │ DINOv3(도메인 SSL, frozen) → L2norm → k_k         │  ← 사전 인덱싱
              └───────────────────────────────────────────────────┘
```

- **학습**: `q_proj`, `<emb>` 임베딩 행, Qwen-VL(LoRA).
- **동결**: DINOv3 → 갤러리 인덱스를 한 번만 프리컴퓨트(안정적).

### 4.2 State latent 공간 = frozen domain-DINOv3 (RQ1 결론)

- 공정 데이터는 public과 gap이 크므로, **도메인 SSL 파인튜닝 DINOv3** 가 일반 Qwen 비전 인코더보다 상태 차이를 잘 분리 → 검색 품질 상한을 결정.
- DINOv3 **freeze** ⇒ 타깃 공간 고정 ⇒ 인덱스 프리컴퓨트 가능, 학습 안정.
- 검색 표현을 LLM의 (상대적으로 약한) 비전 특징과 **분리**.
- 대안(단일 모델화): CIR-CoT식 shared-encoder(타깃도 Qwen 인코딩) → cross-space gap 제거, 대신 도메인 표현력 저하.

$$ q_k=\mathrm{norm}(W_q\, h^{(\mathrm{emb})}_k),\qquad k_k=\mathrm{norm}\big(\mathrm{DINOv3}(x^{(k)})\big)\ (\text{stop-grad}) $$

### 4.3 Special Token 설계

| 항목 | 결정 | 근거 |
|------|------|------|
| 토큰 형태 | **단일 재사용** `<emb>` (번호 X) | 임의 개수 슬롯 일반화, 임베딩 낭비 방지 — LIRA의 `<seg>` 방식 |
| 입력 마커 | 전용 토큰 `<slot>` (실단어 X) | 실단어는 서브워드 분해·의미 누수 |
| 초기화 | 기존 임베딩 평균 + 소량 노이즈 | cold-start 안정화 |
| 쿼리 추출 | `<emb>` **last-layer hidden** | CIR-CoT `<emb>`, LIRA `<seg>` 와 동일 패턴 |
| $k$-번째 쿼리 | $k$-번째 `<emb>` 등장 | 순서로 슬롯 구분(번호 토큰 불필요) |

### 4.4 시퀀스 배치 & 인과성

- **문제**: 스텝을 시계열로 나열하면 causal attention 하에서 `<emb>` 가 뒤쪽(=T에 가까운) 스텝을 못 봄.
- **해결**: `<emb>` 를 **응답부(프롬프트 전체 소비 후)** 에 배치 ⇒ 모든 슬롯이 앵커 + 전체 스텝을 attend.
- 응답에 스텝 나열을 복사(간이 CoT)해 각 `<emb>` 에 **국소 문맥**을 부여(M2IO-R1/CIR-CoT의 reasoning-then-embed 효과).
- **추론**: 슬롯 위치를 이미 알므로 응답 구조를 결정적으로 구성 → **샘플링 없이 single forward** 로 hidden 추출 → ANN 검색.

### 4.5 손실 함수 (Loss)

**전체 목적함수:**
$$ \mathcal{L} = \lambda_{\mathrm{lm}}\,\mathcal{L}_{\mathrm{CE}}(\text{narration}) \;+\; \lambda_{\mathrm{ret}}\,\mathcal{L}_{\mathrm{InfoNCE}} $$

**InfoNCE (false-negative 마스킹 포함):**
$$ \mathcal{L}_{\mathrm{InfoNCE}} = -\frac{1}{S}\sum_{i=1}^{S}\log
\frac{\exp(\langle q_i, k_i^{+}\rangle/\tau)}
{\sum_{j}\mathbb{1}[\,j\notin \mathcal{F}_i\,]\exp(\langle q_i, k_j\rangle/\tau)} $$

- $S$: 배치 내 총 `<emb>` 슬롯 수, $k_i^{+}$: slot $i$ 의 정답 key, $\mathcal{F}_i$: false-negative 마스크.
- `<emb>` emission에는 CE 미적용(위치는 입력이 지정) → LM CE는 **narration 텍스트에만**.

**네거티브 설계:**

| 종류 | 정의 | 목적 | 확보 방법 |
|------|------|------|-----------|
| 같은 시퀀스 다른 슬롯 | 동일 예시 내 다른 상태 이미지 | 슬롯 붕괴·순서 무시 방지 | 멀티슬롯 예시면 **자동 동일 배치** |
| 같은 스텝 다른 인스턴스 | 다른 궤적의 동일 공정 단계 | step-type 아닌 **instance** 판별(앵커 활용) 강제 | step-index 뱅크 / 배치 믹싱 |
| in-batch 일반 | 배치 내 나머지 key | 대조 신호 확대 | 자동 |
| **마스킹**: false neg | 중복/인접(near-identical) 상태 | 노이즈 제거 | image-id 동일 or $|\Delta\text{step}|\le1$ 제외 |

### 4.6 학습 절차 (CIR-CoT식 2-stage)

| 단계 | 동결 | 학습 | 목적 |
|------|------|------|------|
| **Stage A (워밍업)** | Qwen-VL 전체 | `q_proj` + `<emb>` 행 | LLM drift 없이 DINOv3 공간으로 **정렬 cold-start** 해소 |
| **Stage B (SFT)** | DINOv3 | LoRA + `q_proj` + `<emb>` 행 | 멀티슬롯 interleaved 데이터로 **본 학습** (LM CE + InfoNCE) |

---

## 5. 모델 입출력 예시 (Running Example — 반도체 포토 공정)

```
[INPUT]  (prompt; 모든 <emb> 앞에서 소비되어 전 스텝을 attend 가능)
  <image: x_T = 최종 식각 완료 웨이퍼>
  지시문: "최종 상태 이미지와 아래 공정 순서를 보고, 표시된 지점의 중간 상태를 복원하라."
  공정:
    1) 산화막 증착
    2) 감광액 코팅        <slot>      ← 이 시점 상태 검색 요청
    3) 노광
    4) 현상               <slot>      ← 이 시점 상태 검색 요청
    5) 식각
    6) 감광액 제거

[OUTPUT] (response; <slot> 자리에 <emb> 배치)
  "1) 산화막 증착  2) 감광액 코팅 <emb>  3) 노광  4) 현상 <emb>  5) 식각  6) 감광액 제거"
      └ <emb>#1 hidden → q1 → 검색: '코팅 직후 웨이퍼' latent
      └ <emb>#2 hidden → q2 → 검색: '현상 직후 패턴 웨이퍼' latent
```

**슬롯–정답–네거티브 대응:**

| 슬롯 | 대응 상태 | Positive (정답) | 하드 네거티브(예) |
|------|-----------|-----------------|-------------------|
| #1 | 코팅 직후 (S2) | 이 시퀀스 S2 실제 이미지 | 같은 시퀀스 S4; **다른 웨이퍼의 '코팅 직후'** |
| #2 | 현상 직후 (S4) | 이 시퀀스 S4 실제 이미지 | 같은 시퀀스 S2; **다른 웨이퍼의 '현상 직후'** |

> 핵심: "다른 웨이퍼의 동일 단계"가 네거티브이므로, 모델은 단순히 "코팅 단계 이미지"가 아니라 **이 앵커에서 유래한 그 인스턴스**를 골라야 한다(H3).

---

## 6. Related Work

### 6.1 Embedding-as-token & LLM → Retrieval
- **CIR-CoT** (arXiv:2510.08003): MLLM이 CoT 추론 후 `<emb>` 토큰의 last hidden을 검색 임베딩으로 사용, InfoNCE + 2-stage 학습. → **본 제안의 단일 슬롯 원형**.
- **FROMAGe** (Koh et al., 2023): frozen LLM에 `[RET]` 토큰, hidden→CLIP 공간 매핑으로 이미지 검색. → embedding-as-token을 **외부 고정 공간**에 정렬하는 대표.
- **GILL** (Koh et al., 2023): LLM hidden을 이미지 검색/생성 공간으로 매핑(생성까지 확장).
- **E5-V** (Jiang et al., 2024): MLLM 기반 universal multimodal embedding, "한 단어로 요약: `<emb>`" 프롬프트.

### 6.2 Reasoning / Composed Image Retrieval
- **SPRC** (ICLR'24): sentence-level prompt로 CIR 성능 향상(CIR-CoT의 baseline).
- **CIR-LVLM** (AAAI'25): MLLM을 CIR user-intent 인코더로 파인튜닝.

### 6.3 MLLM의 기능성 Special Token (segmentation)
- **LISA** (CVPR'24): embedding-as-mask, `<seg>` 토큰으로 세그멘테이션 디코더 트리거. → seg special token의 원조.
- **LIRA** (arXiv:2507.06272): `<seg>` 를 여러 번 interleaved 생성, 각 hidden을 pixel decoder로. → **다중 기능 토큰 처리의 원형**(본 제안의 멀티슬롯 근거).
- **PixelLM**: 경량 pixel decoder + segmentation codebook로 다중 객체 처리.

### 6.4 Multimodal RAG & Interleaved Multimodal Output
- **M2IO-R1** (arXiv:2508.06328): 텍스트 답 생성 후 문장 사이에 이미지 삽입(리트리벌 기반 멀티모달 출력), GRPO·구조화 출력. → interleave/구조화 관점의 참조.
- **MRAMG-Bench / M2RAG**: 멀티모달 입력–출력 벤치·평가 지표.

### 6.5 Self-Supervised 시각 표현 (state 공간)
- **CLIP** (Radford et al., 2021): 이미지-텍스트 대조학습 → 텍스트 매칭 도메인 gap의 근거.
- **DINOv2 / DINOv3** (Meta): self-supervised 이미지 표현, 도메인 파인튜닝 시 미세 상태 구분에 강함 → **본 제안의 latent state 공간**.

### 6.6 World Models / Latent Next-State 예측 (개념적 근거)
- **I-JEPA** (Assran et al., 2023): 픽셀이 아닌 **표현(latent) 예측** → latent state 추상화 근거.
- **PlaNet / Dreamer** (Hafner et al.): latent state로 dynamics 예측 → "state=충분통계" 근거.

### 6.7 비교표 (본 연구와의 차이)

| 계열 | 대표 | special token | 검색/출력 대상 공간 | 다중 처리 | 본 연구와 차이 |
|------|------|---------------|---------------------|-----------|----------------|
| 임베딩-as-토큰 | CIR-CoT | `<emb>`×1 | MLLM 자기표현 | 단일 | **멀티슬롯** + **외부 domain-DINOv3** 공간 |
| 기능성 토큰 | LIRA | `<seg>`×N | 픽셀 디코더 | interleaved 다중 | 디코더 대신 **retrieval**, state 예측 목적 |
| 멀티모달 RAG | M2IO-R1 | 포맷 마커 | (문장-이미지 매핑) | dict 출력 | 토큰-표현 수준의 **latent 정렬** |
| LLM→검색 | FROMAGe/GILL | `[RET]` | CLIP 고정공간 | 단일 | **공정 상태**·다중·domain SSL 공간 |
| **본 제안** | — | `<emb>`×N (재사용) | **frozen domain-DINOv3** | interleaved 다중 | 공정 이해=next-state retrieval **정식화** |

---

## 7. 구현 계획 (Implementation Plan)

### 7.1 데이터 파이프라인
- 궤적 단위 수집: `(x_T, a_{1:n}, {slot 위치}, {x^{(k)}}, step_id, seq_id, image_id)`.
- 갤러리 사전 인덱싱: 전 중간 이미지 → DINOv3(frozen) latent 캐시.
- 하드 네거티브 뱅크: step_id별 인덱스로 "같은 스텝 다른 인스턴스" 샘플링.
- 마스킹 테이블: 동일/인접 상태(near-duplicate) image-id 목록.

### 7.2 하이퍼파라미터(초기값)

| 항목 | 값 |
|------|----|
| query backbone | Qwen-VL(2.5/3) + LoRA `r=128` |
| state encoder | DINOv3(도메인 SSL) **frozen** |
| 공유 차원 $d$ | $= d_{\mathrm{DINOv3}}$ (예: 1024), gallery linear 없음 |
| temperature $\tau$ | 0.05 |
| $\lambda_{\mathrm{lm}} : \lambda_{\mathrm{ret}}$ | 0.5 : 1.0 |
| Stage A lr | 3e-4 (`q_proj`+`<emb>` 행) |
| Stage B lr | 2e-5 (LoRA) |
| special tokens | `<\|emb\|>`, `<\|slot\|>` (재사용) |

### 7.3 스택
- Transformers(Qwen-VL) + PEFT(LoRA, `modules_to_save=[embed_tokens, lm_head]`).
- DINOv3 + AutoImageProcessor(도메인 ckpt).
- FAISS 등 ANN 인덱스(추론 검색).

---

## 8. 평가 프로토콜 (Evaluation)

### 8.1 지표

| 지표 | 정의 | 역할 |
|------|------|------|
| Recall@K (per-slot) | 슬롯별 정답 이미지가 top-K에 포함 | **주지표** |
| mAP@K | 복수 정답 대응 | 다중정답 상황 |
| Order/Consistency | 한 시퀀스의 슬롯들이 **서로 다른** 정답을 top-1로 뽑는 비율 | **붕괴 진단** |
| Prev-state Acc | "$i$-step 이전 상태" 정확 검색율 | 전이 이해 척도 |

### 8.2 베이스라인
- (a) DINOv3 kNN only (앵커→최근접, 문맥 무시).
- (b) 텍스트 state 매칭(스텝 캡션→텍스트 검색) — P4 대비.
- (c) Qwen 자기표현 shared-encoder(CIR-CoT식) — RQ1 대비.
- (d) 단일 `<emb>`(전 슬롯 concat) vs 멀티 `<emb>`.

### 8.3 Ablation

| Ablation | 검증 대상 |
|----------|-----------|
| 앵커 $x_T$ 제거 | instance disambiguation이 앵커에서 오는가 (**핵심 H3**) |
| gallery: DINOv3 vs Qwen-vision | 도메인 state 공간의 이점 (RQ1) |
| single `<emb>` vs numbered `<emb_i>` | 재사용 토큰 일반화 |
| in-sequence neg on/off | 슬롯 붕괴 방지 기여 (RQ2) |
| Stage A 유무 | cold-start 워밍업 효과 |
| narration(CoT) 유무 | interleaved 근거의 검색 기여(CIR-CoT Full vs Fast) |

---

## 9. 위험 요소 & 완화 (Risks)

| 위험 | 원인 | 완화 |
|------|------|------|
| 표현 붕괴 | 슬롯 임베딩 동질화 | in-sequence 하드 네거티브, 다양성 정규화 |
| false negative | 인접 상태 유사 | image-id / $\Delta$step 마스킹, $\tau$ 조정 |
| 인과 가림 | `<emb>` 가 뒤 스텝 못 봄 | `<emb>` 를 **출력부**(프롬프트 후)에 배치 |
| cold start | LLM이 DINO공간 못 맞춤 | **Stage A** 정렬 워밍업 |
| 공정 확률성 | 같은 스텝→다른 상태 | 앵커 $x_T$ 로 disambiguation + same-step 하드 네거티브 |
| 도메인 표현 부족 | DINOv3 미분리 상태 | SSL 파인튜닝 강화, **사전 t-SNE/kNN 점검**(H1) |
| 인덱스 드리프트 | gallery 학습 시 | gallery **freeze**(프리컴퓨트) |

---

## 10. 마일스톤 (Milestones)

| 단계 | 산출물 | 기간(예) |
|------|--------|----------|
| M1 | DINOv3 도메인 SSL + 상태분리 검증(t-SNE, kNN) | 3주 |
| M2 | 데이터 스키마·collator·단일 슬롯 파일럿(Stage A) | 3주 |
| M3 | 멀티슬롯 Stage B + 붕괴/앵커 ablation | 4주 |
| M4 | 전체 벤치·베이스라인·리포트 | 4주 |

---

## 11. 기대 기여 (Contributions)

- **C1.** 공정 이해를 **latent next/intermediate-state retrieval** 로 정식화한 최초 프레임.
- **C2.** 텍스트-state의 도메인 gap·평가 모호성을 **image-feature latent state** 로 회피.
- **C3.** 단일 재사용 `<emb>` 의 **멀티슬롯 embedding-as-token** 을 Qwen-VL에 구현(LIRA × CIR-CoT).
- **C4.** **frozen domain-DINOv3** 를 state 공간으로 사용, retrieval + **anchor-ablation** 으로 전이 이해를 정량 평가.
- **C5.** (데이터 확보 시) 상태-속성 평가 세트(AttrEval류)로 공정 이해의 세분화 진단.

---

## 부록 A. 핵심 수식 요약

$$
q_k=\mathrm{norm}(W_q h^{(\mathrm{emb})}_k),\quad
k_j=\mathrm{norm}(\mathrm{DINOv3}(x^{(j)})),\quad
\mathcal{L}=\lambda_{\mathrm{lm}}\mathcal{L}_{\mathrm{CE}}+\lambda_{\mathrm{ret}}\mathcal{L}_{\mathrm{InfoNCE}}
$$

$$
\mathcal{L}_{\mathrm{InfoNCE}}=-\frac1S\sum_i\log
\frac{\exp(\langle q_i,k_i^{+}\rangle/\tau)}{\sum_{j\notin\mathcal{F}_i}\exp(\langle q_i,k_j\rangle/\tau)}
$$
