# AI Red Teaming Course: Section Quizzes

## Section 1: Introduction to AI Red Teaming

1. **What is the primary purpose of the PyRIT framework?**
   - A) To create AI art and images
   - B) To systematically test AI system security
   - C) To optimize AI model performance
   - D) To develop new language models

2. **Which of these is a core vulnerability type mentioned in the course materials?**
   - A) Electrical power surges
   - B) Prompt injection
   - C) Physical hardware damage
   - D) Network infrastructure failure

3. **What is the correct initialization method for PyRIT's memory component?**
   - A) `pyrit.initialize_memory(IN_MEMORY)`
   - B) `initialize_pyrit(memory_db_type=IN_MEMORY)`
   - C) `pyrit.memory.set_type(IN_MEMORY)`
   - D) `pyrit.config(memory="IN_MEMORY")`

4. **In PyRIT's architecture, what is the role of orchestrators?**
   - A) They handle API authentication only
   - B) They coordinate attacks and manage test sequences
   - C) They train AI models on adversarial examples
   - D) They repair vulnerable systems

5. **What is a core component of Microsoft's AI Red Teaming Framework?**
   - A) Systematic testing process
   - B) Automatic code generation
   - C) Physical security measures
   - D) Cloud infrastructure optimization

## Section 2: Attack Strategies and Concepts

1. **What is a primary goal of "input manipulation" as an attack primitive?**
   - A) To optimize model training
   - B) To enhance model performance
   - C) To strategically modify prompts to bypass safeguards
   - D) To improve tokenization algorithms

2. **Which of these is an example of a character-level transformation technique?**
   - A) Sentiment analysis
   - B) ROT13 encoding
   - C) Content summarization
   - D) Semantic similarity matching

3. **In the context of LLM manipulation psychology, what does "role-playing tactics" refer to?**
   - A) Having the model perform in video games
   - B) Encouraging the model to adopt specific personas that might bypass restrictions
   - C) Creating fictional scenarios for entertainment purposes
   - D) Testing the model's acting capabilities

4. **What is a key component of an "attack chain" in AI red teaming?**
   - A) Physical access to server hardware
   - B) Sequence of techniques applied to achieve an objective
   - C) Chain of command in security teams
   - D) Cryptographic key exchange

5. **Which PyRIT component is responsible for transforming prompts before sending them to the target?**
   - A) Orchestrators
   - B) Converters
   - C) Scorers
   - D) Memory Components

## Section 3: PyRIT Orchestrator Deep Dive

1. **What is the primary responsibility of orchestrators in PyRIT's architecture?**
   - A) Processing model outputs only
   - B) Connecting targets, converters, and scorers in an attack flow
   - C) Managing database connections only
   - D) Providing user interfaces

2. **Which database option in PyRIT provides persistent state management for orchestrators?**
   - A) MongoDB
   - B) SQLite
   - C) DuckDB
   - D) PostgreSQL

3. **In a stateful orchestrator, what does the `track_state` method typically do?**
   - A) Monitors system CPU usage
   - B) Records attack stages and data in memory and/or database
   - C) Tracks user login information
   - D) Manages network connections

4. **What is a benefit of implementing orchestrators as classes rather than functions?**
   - A) They run faster
   - B) They require less memory
   - C) They can maintain state across multiple operations
   - D) They only work with specific models

5. **Which technique allows an orchestrator to handle multiple targets with fallback capabilities?**
   - A) Multi-threading
   - B) Target integration with fallback strategies
   - C) Memory scaling
   - D) GPU acceleration

## Section 4: Advanced Attack Techniques

1. **What does the acronym PAIR stand for in the context of AI red teaming?**
   - A) Performance Analysis of Intelligent Response
   - B) Prompt Automatic Iterative Refinement
   - C) Primary Attack Injection Routine
   - D) Programmatic AI Resource Integration

2. **What is the primary function of the `refine_prompt` method in a PAIR orchestrator?**
   - A) To clean the prompt of spelling errors
   - B) To improve the prompt based on previous responses
   - C) To shorten the prompt length
   - D) To translate the prompt to another language

3. **What technique does GCG (Greedy Coordinate Gradient) optimization use to bypass model safeguards?**
   - A) Word frequency analysis
   - B) Targeted token optimization using model gradients
   - C) Black-box random text generation
   - D) Hardware-level instruction manipulation

4. **In a TAP (Tree of Attacks with Pruning) implementation, what does the "advisor model" do?**
   - A) Provides legal advice on attack legality
   - B) Suggests optimal next prompts to achieve the objective
   - C) Creates a visual representation of the attack tree
   - D) Manages infrastructure resources

5. **What is a key benefit of using a hybrid attack orchestrator that combines multiple techniques?**
   - A) It always produces faster attacks
   - B) It uses less memory
   - C) It can adaptively try different strategies if initial ones fail
   - D) It requires no database storage

## Section 5: Security Assessment and Reporting

1. **Which of these is a quantitative measure for attack success?**
   - A) Subjective assessment of model behavior
   - B) Success rate (% of attempts achieving objective)
   - C) Theoretical vulnerability analysis
   - D) User satisfaction with responses

2. **What does CVSS stand for in the context of security scoring?**
   - A) Common Vulnerability Scoring System
   - B) Comprehensive Validation Security Standard
   - C) Complete Vulnerability Surveillance Service
   - D) Critical Verification System Specification

3. **Which component should be included in the executive summary of a red team report?**
   - A) Detailed code snippets of all attacks
   - B) Complete technical logs
   - C) Overview of findings categorized by severity
   - D) Full attack reproduction steps

4. **What does a "Defense-in-Depth Assessment" evaluate?**
   - A) The physical security of data centers
   - B) The multiple layers of security controls and their effectiveness
   - C) The depth of encryption algorithms
   - D) The server rack configuration

5. **What is an effective way to present technical findings to non-technical stakeholders?**
   - A) Providing raw log files
   - B) Using visual elements like risk heat maps and charts
   - C) Including all code samples
   - D) Focusing exclusively on technical implementation details

---

# Answer Key

### Section 1: Introduction to AI Red Teaming
1. B) To systematically test AI system security
2. B) Prompt injection
3. B) `initialize_pyrit(memory_db_type=IN_MEMORY)`
4. B) They coordinate attacks and manage test sequences
5. A) Systematic testing process

### Section 2: Attack Strategies and Concepts
1. C) To strategically modify prompts to bypass safeguards
2. B) ROT13 encoding
3. B) Encouraging the model to adopt specific personas that might bypass restrictions
4. B) Sequence of techniques applied to achieve an objective
5. B) Converters

### Section 3: PyRIT Orchestrator Deep Dive
1. B) Connecting targets, converters, and scorers in an attack flow
2. C) DuckDB
3. B) Records attack stages and data in memory and/or database
4. C) They can maintain state across multiple operations
5. B) Target integration with fallback strategies

### Section 4: Advanced Attack Techniques
1. B) Prompt Automatic Iterative Refinement
2. B) To improve the prompt based on previous responses
3. B) Targeted token optimization using model gradients
4. B) Suggests optimal next prompts to achieve the objective
5. C) It can adaptively try different strategies if initial ones fail

### Section 5: Security Assessment and Reporting
1. B) Success rate (% of attempts achieving objective)
2. A) Common Vulnerability Scoring System
3. C) Overview of findings categorized by severity
4. B) The multiple layers of security controls and their effectiveness
5. B) Using visual elements like risk heat maps and charts