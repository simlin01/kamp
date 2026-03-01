# Risk-Aware Integrated Framework for Production Planning  
### Demand Forecasting × Constrained Optimization × Monte Carlo Risk Control

---

## Authors

| Name | GitHub |
|------|--------|
| 심승현 | https://github.com/simlin01 |
| 채소연 | https://github.com/Chaesoyeon |

---

## Abstract

본 연구는 수요 변동성과 설비 제약이 동시에 존재하는 제조 환경에서의 생산계획 문제를 확률적 리스크 관점에서 재정의한다. 단일 예측값 기반 최적화는 평균 성능에는 적합할 수 있으나, 실제 운영 환경에서 발생하는 tail risk를 충분히 반영하지 못한다.  

이에 본 프로젝트는 **수요예측–제약최적화–Monte Carlo 기반 리스크 평가–정책조정**을 통합한 리스크 민감형 생산계획 프레임워크를 제안한다. VaR 및 CVaR 기반 위험 지표를 통해 평균 손실이 아닌 상위 위험 구간을 통제하며, 정책 파라미터를 반복적으로 조정하는 피드백 구조를 구축한다.

---

## 1. Problem Setting

제조 산업에서 생산계획은 다음의 불확실성을 동시에 가진다.

- 수요 예측 오차  
- 생산능력(capacity) 제약  
- 최소 생산 단위(min-lot) 제약  
- 재고 커버리지 및 결품 리스크  

평균 수요 기준의 결정은 변동성이 큰 환경에서 다음과 같은 구조적 문제를 야기한다.

- Backlog 급증  
- 특정 SKU의 재고 커버리지 붕괴  
- 손실(Loss)의 tail risk 확대  

따라서 본 연구는 생산계획을 단일 최적화 문제가 아닌 **확률적 리스크 최소화 문제**로 재정의한다.

---

## 2. Methodology

본 프레임워크는 다음의 4단계 구조로 구성된다.

### (1) Demand Forecasting  
SKU 단위 시계열 수요 예측 수행  
- 평균(mean) 수요 시나리오  
- 보수적(p90) 수요 시나리오  

### (2) Constrained Optimization  
OR-Tools CP-SAT 기반 생산계획 최적화  
- 일일 생산능력 제약  
- 최소 생산 lot 제약  
- 재고 밸런싱  
- 비용 기반 목적함수 최소화  

### (3) Monte Carlo Simulation  
다중 수요 시나리오 생성 후 다음 지표를 추정한다.

- ShortageRate  
- InventoryRate  
- Loss  
- VaR (p90)  
- CVaR  
- Worst-case loss  

### (4) Policy Adjustment  
Monte Carlo 결과를 기반으로 다음 정책 파라미터를 반복 조정한다.

- capacity  
- λ_smooth  
- λ_CVaR  
- min_lot_map  

---

## 3. Risk-Oriented Evaluation

기존 접근이 평균 손실(mean Loss)에 초점을 둔다면, 본 연구는 다음을 중심으로 평가한다.

- **VaR (Value at Risk)** : 상위 10% 위험 구간  
- **CVaR (Conditional VaR)** : tail 평균 손실  
- Worst-case scenario  

이를 통해 단순 효율성(efficiency)이 아닌 **안정성(stability)** 중심의 의사결정 구조를 구현한다.

---

## 4. System Integration

본 시스템은 다음과 같은 통합 파이프라인으로 구성된다.

```
Demand Forecasting  
→ CP-SAT Optimization  
→ Monte Carlo Risk Evaluation  
→ Traffic Light Diagnosis  
→ LLM-based Executive Report  
```

- Traffic Light 기반 경영진 진단 체계  
- 자동 리포트 생성 및 정책 피드백 루프 구현  

---

## 5. Contributions

1. 생산계획 문제를 확률적 tail risk 최소화 관점으로 재정의  
2. VaR / CVaR 기반 정책 튜닝 구조 제안  
3. 예측–최적화–리스크–리포트를 연결한 통합 의사결정 시스템 구축  
4. 다양한 제조 공정으로 확장 가능한 일반화 구조 제시  

---

## 6. Conclusion

본 프레임워크는 평균 성능 최적화에 머무르지 않고, 불확실성 하에서의 위험 통제까지 고려한 생산계획 체계를 제안한다. Monte Carlo 기반 분포 추정과 CVaR 중심 정책 조정을 통해, 변동성이 높은 제조 환경에서도 안정적인 운영 전략 수립이 가능함을 보인다.
