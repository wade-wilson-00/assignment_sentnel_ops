# SentnelOps AI/ML Internship Assignment
## Infrastructure Anomaly Detection & Reasoning System

### Overview
I chose to implement this with a **Hybrid AI/ML approach** it is designed to detect infrastructure anomalies and provide human-readable reasoning for each detection. It combines predictive Machine Learning for pattern recognition and LLM as a reasoning system.

---

### My Approach: The Hybrid Pipeline
I chose a **Hybrid Architecture** because infrastructure data is quantitative (metrics) which alone ML outputs cannot be helpful for understanding in depth, LLMs can recieve the context of the metrics and predictions and based on that it can generate valuable insights.
Took LLM help for better understanding of features and logic to build the entire pipeline, which helped me to understand and make better decisions.

#### 1. Synthetic Data Generation (`synthetic_data.py`)
- I took the help of LLM and asked it to Create a dataset of **1000 resources** across 5 distinct scenarios: `normal`, `cpu_saturated`, `memory_pressure`, `over_provisioned`, and `traffic_cpu_mismatch`. as asked in the assignment.
- **Realism**: Added noiseto simulate real-world "dirty" infrastructure logs.
- **Feature Engineering**: Engineered 3 high-signal features:
    - `cpu_spike_ratio`: Detects micro-bursting.
    - `resource_saturation`: Combined pressure of CPU and RAM.
    - `load_efficiency`: Disambiguates traffic surges from runaway processes.

#### 2. ML Detection Layer (`model_train.py`)
- Used a **Random Forest Classifier** to perform multi-class classification.
- **Reasoning**: Random Forests are highly interpretable and handle tabular data better than deep learning for this scale. 
- **Performance**: initially Model overfitted because of the perfect synthetic data, so altered it with by adding some noise and Achieved **~82% accuracy**. This is a realistic score that accounts for the intentional noise and ambiguity added to the dataset.

#### 3. LLM Reasoning Layer (`llm_reasoning.py`)
- Integrated **Meta-Llama-3** (via HuggingFace Inference API).
- **Process**: The ML model's prediction and confidence score are "packaged" with raw metrics and sent to the LLM.
- **Persona**: The LLM acts as a "Smart Resource Analyst," transforming raw numbers into actionable advice and security notes.

---

### How to Run
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set up Environment**: Create a `.env` file with your HuggingFace API key:
    ```env
    HUGGING_FACE_API_KEY=your_key_here
    META_LLAMA_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
    ```
3.  **Generate Data**:
    ```bash
    python scripts/synthetic_data.py
    ```
4.  **Train & Analyze**:
    ```bash
    python scripts/model_train.py
    ```

---

### Tradeoffs & Decisions
- **Rule-based vs. ML**: I avoided a purely rule-based approach (e.g., `if cpu > 80`) because it's not an efficient way to detect. Using ML allows the system to find complex relationships (like CPU vs. Network divergence) and if added an LLM for reasoning, it enhances the approach to detect and analyze the anomaly and it's cause.
- **Single Script vs. Modular**: I chose a modular approach (`synthetic_data` -> `model_train` -> `llm_reasoning`) to ensure the pipeline is maintainable and testable.
- **Confidence Scores**: I used `predict_proba` from the Random Forest to provide a mathematical confidence score, which is passed to the LLM to help it decide how "sure" it should sound.

---

---

### Sample Outputs
*(The system generates valid JSON as requested)*

```Complete Output with 5 Sample analysis
Shape of X:(1000, 9)
Shape of Y:(1000,)
Shape of X_train - (800, 9)
Shape of X_test - (200, 9)
Shape of y_train - (800,)
Shape of y_test - (200,)
Model is Trained !

--- Model Performance ---
Accuracy : 82.00%
Precision: 82.44%
Recall   : 82.00%
GENERATING 5 SAMPLE ANALYSES

--- [SAMPLE 6] ---
{
    "resource_id": "i-0605",
    "is_anomalous": true,
    "anomaly_type": "memory_pressure",
    "reason": "High resource utilization (CPU avg: 42.0%) combined with memory pressure prediction score of 0.7878974355510664 indicates anomalous behavior.",
    "suggested_action": "Investigate and adjust system configuration to address high memory utilization and prevent potential system crashes.",
    "confidence": 0.7878974355510664,
    "security_note": "High CPU and memory utilization may lead to security vulnerabilities if not addressed promptly."
}

--- [SAMPLE 7] ---
{
  "resource_id": "i-0478",
  "is_anomalous": true,
  "anomaly_type": "cpu_saturated",
  "reason": "High CPU average usage of 74.8% indicates a potential bottleneck",
  "suggested_action": "Monitor and investigate further for any system crashes or performance degradation",
  "confidence": 0.919,
  "security_note": "This anomaly may have a moderate impact on system reliability and performance if left unaddressed"
}

--- [SAMPLE 8] ---
{
  "resource_id": "i-0056",
  "is_anomalous": false,
  "anomaly_type": "normal",
  "reason": "All metrics are within normal ranges and the prediction model confirms a normal state with high confidence.",
  "suggested_action": "No action needed at this time.",
  "confidence": 0.7561581183833557,
  "security_note": null
}

--- [SAMPLE 9] ---
{
  "resource_id": "i-0182",
  "is_anomalous": true,
  "anomaly_type": "cpu_saturated",
  "reason": "CPU average utilization is higher than expected (64.8%) indicating potential resource overload.",
  "suggested_action": "Scale up the resource to increase CPU capacity or monitor for potential issues before saturation.",
}

--- [SAMPLE 10] ---
{
  "resource_id": "i-0344",
  "is_anomalous": false,
  "anomaly_type": "over_provisioned",
  "reason": "Cpu usage is below 25% and the resource is not internet-facing, but network usage is above average, indicating possible oversubscription.", 
  "suggested_action": "Monitor the resource's performance and adjust the allocation as necessary to avoid potential underutilization.",
  "confidence": 0.79,
  "security_note": ""
}
```