# Factory Maintenance Digital Twin 🏭

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/meta-pytorch/OpenEnv)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A high-fidelity **Reinforcement Learning environment** simulating industrial machine health, stochastic decay, and strategic budget management. Designed for the **Meta OpenEnv Hackathon**.

![Factory Dashboard](https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/docs/assets/banner.png)

## 📌 Motivation
In modern manufacturing, **Predictive Maintenance** (PdM) is a critical real-world problem. Traditional "run-to-fail" strategies are costly, while "preventive" strategies can be wasteful. This environment models the complex trade-offs an AI agent must make between operational revenue and maintenance costs, simulating the stochastic nature of industrial wear-and-tear.

---

## 🛠 Action & Observation Spaces

### Action Space (Typed)
Agents interact using the `FactoryAction` model.

| Action | Argument | Cost | Description |
| :--- | :--- | :--- | :--- |
| `wait` | - | $10 | Observe machine decay without intervening. |
| `inspect` | `machine_id` | $30 | Get exact health status of a specific machine. |
| `repair` | `machine_id` | $150 | Partial restoration of health (prevents failure). |
| `replace` | `machine_id` | $600 | Reset machine to 100% health (operational). |

### Observation Space (Typed)
The `FactoryObservation` provides a rich digital twin state.

| Field | Type | Description |
| :--- | :--- | :--- |
| `machines` | `List[MachineState]` | ID, Status (operational/warning/broken), and Health %. |
| `production_rate` | `float` | Current efficiency (0-100%). |
| `budget` | `float` | Financial resources (failure if < 0). |
| `last_event` | `str` | NL description of the previous step's events. |
| `task_id` | `str` | Current task difficulty (easy, medium, hard). |
| `score` | `float` | Normalized performance score (0.0 - 1.0).|

---

## 🎯 Tasks & Programmatic Grader

The environment includes **3 built-in tasks** that test different strategy horizons.

| Task ID | Difficulty | Budget | Decay | Objective |
| :--- | :--- | :--- | :--- | :--- |
| **`easy`** | **Easy** | $3,000 | Slow | Survive 50 steps with minimal downtime. |
| **`medium`** | **Medium** | $2,000 | Standard | Balance revenue vs costs in a volatile market. |
| **`hard`** | **Hard** | $1,000 | Aggressive | Prevent cascaded machine failure under crisis. |

### The Grader Logic
Your performance is measured by a **deterministic programmatic grader**:
$$\text{Score} = \max\left(0.0, \min\left(1.0, \frac{\text{Budget}_{\text{final}}}{\text{Budget}_{\text{initial}}}\right)\right)$$
*   **Success**: Score $\ge 0.5$ (Survived with healthy profit).
*   **Failure**: Score $< 0.5$ or Bankruptcy.

---

## 🚀 Quick Start (Local)

### 1. Installation
```bash
pip install -e .
```

### 2. Run the Digital Twin (Browser)
Observe the agent or play manually via the Gradio dashboard:
```bash
python -m server.app
# Open http://localhost:8000/dashboard
```

### 3. Run Inference
Connect a local LLM or API model to the environment:
```bash
export HF_TOKEN="your_token"
python inference.py
```

---

## 🐳 Docker & Submission

### Build & Run
```bash
docker build -t factory-pdm -f server/Dockerfile .
docker run -p 8000:8000 factory-pdm
```

### Validate Submission
Use the built-in validator to ensure all specs are met:
```bash
./scripts/validate-submission.sh http://localhost:8000
```

---

## 📊 Baseline Performance
| Model | Task | Score | Result |
| :--- | :--- | :--- | :--- |
| **Qwen2.5-72B** | `easy` | 0.92 | ✅ PASS |
| **Qwen2.5-72B** | `medium` | 0.81 | ✅ PASS |
| **Qwen2.5-72B** | `hard` | 0.44 | ❌ FAIL |

---

## ⚖️ License & Compliance
This project is open-source under the **BSD-3 License** and strictly follows the **OpenEnv v0.2.2 Specification**.
