You are the architecture agent for this repository.

Read:
1. ASSIGNMENT.md in full.
2. The existing starter repository structure and code.

Your job is NOT to implement the assignment. Your job is to define the architecture and interface contracts that will allow multiple independent coding agents to implement different components in parallel.

The implementation will be delegated across these components:

1. download.py
2. preprocess.py
3. word_tokenizer.py + build_word_tokenizer.py
4. data.py
5. model.py
6. utils.py
7. train.py
8. eval.py

Your task:

1. Inspect the existing repository before proposing anything.
   - Identify starter code that should be preserved or extended.
   - Identify any existing classes/functions/interfaces that later agents should reuse.
   - Do not redesign something that is already appropriately provided.

2. Extract the functional requirements from ASSIGNMENT.md and map each requirement to the component responsible for it.

3. Define the PUBLIC INTERFACE of each component.
   For every public class/function specify:
   - name
   - purpose
   - arguments and types
   - return value and type
   - important tensor/data shapes
   - exceptions or assumptions where relevant

4. Define all cross-component contracts explicitly.

   In particular specify:
   - raw dataset representation expected by preprocess.py
   - JSONL schema produced by preprocess.py
   - tokenizer interface used by data.py
   - tokenizer artifact format
   - dataset/dataloader output format
   - context and target tensor shapes
   - model constructor interface
   - model.forward() input/output shapes
   - config schema
   - checkpoint schema
   - functions train.py expects from utils.py
   - functions eval.py expects from data.py/model.py/utils.py
   - metrics.json schema

5. Construct a dependency graph showing which components may depend on which other components.

6. Minimize coupling.
   Agents implementing individual components should only need:
   - ASSIGNMENT.md
   - their own target file(s)
   - the public interface contracts of their dependencies

7. Do not add unnecessary abstractions. This is a small educational implementation of a feed-forward Neural Probabilistic Language Model, not a production ML framework.

8. Resolve ambiguities in ASSIGNMENT.md by choosing the simplest implementation that satisfies the assignment. Clearly label these choices as architectural decisions rather than assignment requirements.

9. Do NOT:
   - implement the components
   - substantially modify source files
   - introduce features not required by the assignment
   - create complex inheritance/plugin systems
   - assume an interface without first checking whether the starter repository already establishes one

OUTPUT:

Create ARCHITECTURE.md containing:

# Architecture

## 1. Assignment Requirements
A concise requirement -> owning component mapping.

## 2. Dependency Graph
Show component dependencies.

## 3. Shared Data Contracts
JSONL schema, tokenizer artifacts, configs, checkpoints, metrics, etc.

## 4. Component Interfaces

### download.py
...

### preprocess.py
...

### word_tokenizer.py / build_word_tokenizer.py
...

### data.py
...

### model.py
...

### utils.py
...

### train.py
...

### eval.py
...

For each, clearly distinguish:
- Public API
- Inputs
- Outputs
- Dependencies
- Assignment requirements satisfied

## 5. Tensor Shape Contracts
Explicitly trace shapes through the pipeline, e.g.

token IDs -> context window -> embeddings -> concatenation -> hidden -> logits -> loss

Use symbolic dimensions such as:
B = batch size
C = context size
E = embedding dimension
H = hidden dimension
V = vocabulary size

## 6. Architectural Decisions
List choices made where ASSIGNMENT.md allows multiple approaches and explain the simplest choice.

## 7. Agent Context Map
For each implementation agent, specify exactly which other component interfaces it needs to see.

Do not implement code. The goal is to freeze contracts before parallel implementation begins.