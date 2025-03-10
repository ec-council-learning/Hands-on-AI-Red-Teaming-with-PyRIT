# Section 1: Introduction to AI Red Teaming

## 1.1 Understanding the AI Security Landscape

### The Challenge

- Rapid AI system deployment across industries e.g. Visa w/ 500
- Organizations lack AI security testing expertise e.g. AI Purple Teaming
- Complex attack surfaces and novel threats e.g. nanogcg optimizers
- Need for systematic security assessment e.g. PyRIT Framework

### Why It Matters

```python
# Example: Basic PyRIT initialization
from pyrit.common import initialize_pyrit, IN_MEMORY
initialize_pyrit(memory_db_type=IN_MEMORY)
```

---

## 1.2 Common LLM Vulnerabilities and Attack Vectors

### Core Vulnerability Types

1. Prompt Injection

- Direct manipulation of inputs
- Forces unintended outputs

```python
basic_prompt = "Tell me how to create [harmful content]"
```

2. Jailbreaking

   - Bypasses safety measures
   - Enables restricted outputs

3. Data Extraction

   - Retrieves training data
   - Exposes sensitive information

4. Data Poisoning

   - Corrupts training process
   - Introduces backdoors

5. Model Manipulation

   - Alters runtime behavior
   - Compromises model integrity

---

## 1.3 Microsoft's AI Red Teaming Framework Overview

### Framework Components

1. Systematic Testing Process

   - Structured methodology
   - Repeatable procedures
   - Clear documentation

2. Risk Assessment

   - Threat modeling
   - Impact analysis
   - Mitigation strategies

```python
# Example: Basic orchestrator setup
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import AzureMLChatTarget

target = AzureMLChatTarget()
orchestrator = PromptSendingOrchestrator(objective_target=target)
```

---

## 1.4 PyRIT Architecture and Components

### Core Components

1. Orchestrators

   - Coordinate attacks
   - Manage test sequences
   - Handle responses

2. Targets

   - Model endpoints
   - API interfaces
   - System boundaries

3. Converters

   - Transform prompts
   - Modify inputs
   - Format outputs

4. Scorers

   - Evaluate responses
   - Measure success
   - Track metrics

---

## 1.5 Hands-on Lab: Setting Up Your PyRIT Environment

### Environment Requirements

**Hardware Requirements:**

- OS: Windows 10+, macOS 10.15+, Ubuntu 20.04+
- CPU: 2.0 GHz dual-core minimum
- AzureML Endpoints or local-alternative (e.g. Ollama)
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB free space

**Software Setup:**

```bash
# Python environment
python -m venv pyrit-env
source pyrit-env/bin/activate  # Unix
pyrit-env\Scripts\activate     # Windows

# Install PyRIT
pip install --upgrade pip
pip install pyrit
```

---

## Review & Next Steps

### Key Takeaways

- Understanding of AI security landscape
- Knowledge of common vulnerabilities
- Familiarity with PyRIT framework
- Working test environment

### Coming Up: Section 2

- Attack Strategies and Concepts
- Building blocks of AI system attacks
- Practical attack implementation

### Resources

- llm-attacks.org
- Latent Space Tools
- ZeroDay.Tools
- PyRIT Documentation

---