# AI Security: Key Risks and Mitigations for Security Teams

## Core AI Security Challenges

AI/ML Observability & Safety teams are crucial in mitigating emerging AI-specific risks in a compliant way. Current market solutions fail to effectively manage cyber and legal exposure, while state-backed actors actively target market leaders. Additionally, AI systems face threats including regulatory exposure, reputational damage from negative customer experiences and hallucinations, data loss, external IP theft, and hidden intrusions.

## Key AI-Specific Vulnerabilities

All Generative AI introduces durable net-new risk specific to auto-regressive decoding mechanisms. These risks may not be fully mitigated even in next-generation AI architectures that generate abstractions rather than token-level representations.

Key vulnerability categories include:

1. Decompilation of binaries and decoding of encrypted data
2. Universal and transferable susceptibility to automated attacks
3. Durable and scalable exploitation via malicious data

A recently discovered novel attack method called "ShadowLogic" allows attackers to create codeless, surreptitious backdoors in AI models by manipulating their computational graph. These backdoors persist through fine-tuning and can hijack models to trigger attacker-defined behavior when specific inputs are received.

## Mitigation Strategy

Affecting Enterprise AI/ML Observability & Safety requires development and integration of four product categories:

1. **Per-Application Baselines**: Using known good/bad examples for real-time monitoring
2. **Real-Time Monitoring**: Leveraging efficient evaluation against examples with configurable thresholds
3. **Automated Playbooks**: Implementing test criteria and validation of baselines
4. **Customizable Instrumentation**: Using customized transformer heads to understand model activations

## Core Competencies for Security Teams

Three core competencies are essential:

* **External Observability**: Managing alerts at scale by exception
* **Internal Explainability**: Detailed audit of AI activations & decisions
* **Per-Component Securability**: Minimizing residual risk

## 'AI Passport' Approach

Implementation requires a role-based approach with an API control plane that provides:

* Schema evolution support
* Role-based access control
* Complete audit trail
* Backwards compatibility

AI security is fundamentally about observability with context. Teams should track inputs, outputs, and model behavior patterns, comparing against established baselines to detect anomalies. This should be logged to a per-integration API with deployment-specific details and end-to-end configuration records.

## Documentation

* [1 of 3 Emerging AI Risks](https://github.com/ec-council-learning/Hands-on-AI-Red-Teaming-with-PyRIT/blob/main/1%20of%203%20Emerging%20AI%20Risks.pdf)
* [2 of 3 AI Vulnerability](https://github.com/ec-council-learning/Hands-on-AI-Red-Teaming-with-PyRIT/blob/main/2%20of%203%20AI%20Vulnerability.pdf)
* [3 of 3 AI-Specific Risk Mitigation](https://github.com/ec-council-learning/Hands-on-AI-Red-Teaming-with-PyRIT/blob/main/3%20of%203%20AI-Specific%20Risk%20Mitigation.pdf)
* [ShadowLogic Research](https://hiddenlayer.com/innovation-hub/shadowlogic/#Backdooring%20Phi-3)

## External Tools & Frameworks

* [PyRIT (Python Risk Identification Tool for AI)](https://azure.github.io/PyRIT/index.html) - Microsoft's open-source tool for AI risk assessment
* [GhidraMCP (Machine Learning Model Analysis)](https://github.com/LaurieWired/GhidraMCP) - Ghidra extension for ML model analysis and reverse engineering
* [MITRE ATLAS Framework](https://atlas.mitre.org/) - Adversarial Threat Landscape for AI Systems
* [AI Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf) (NIST)
