## About the Project — Agentic ICD-10 Coding System

This project implements an **agentic AI system** for automated **ICD-10 medical coding**, designed to map clinical diagnoses and procedures to standardized ICD-10 codes.

The system is built around a **hierarchical agent architecture**, where a primary agent delegates tasks to specialized **sub-agents**. These sub-agents process structured and unstructured medical case inputs, reason over clinical context, and output the most appropriate ICD-10 codes.

---

## Core Capabilities

### Agentic Medical Coding
- **Primary agent** orchestrates the workflow and delegates subtasks  
- **Specialized sub-agents** receive diagnoses and procedures and infer corresponding ICD-10 codes  
- Designed to handle complex, multi-condition medical cases  

### Clinical RAG
- **Retrieval-Augmented Generation (RAG)** pipeline indexing:
  - ICD-10 official guidelines  
  - Tabular lists  
  - Coding rules and exclusions  
- Ensures outputs are grounded in authoritative medical documentation  

### Evaluation & Metrics
- Integrated **LangSmith evaluations** to assess system performance:
  - **F1-score**
  - **Accuracy**
  - **Latency**
  - Error analysis per medical case  
- Enables iterative improvement and objective comparison across agent versions  

---

## AI Architecture

- **Deep agent framework** with hierarchical agent–subagent delegation  
- **Tool-based reasoning** for guideline retrieval and code validation  
- **Object-Oriented Design (OOP)** for modularity and extensibility  

Implemented using:
- **LangChain** for agent abstraction  
- **LangGraph** for structured multi-step workflows  
- **DeepAgents** for hierarchical agent coordination  

---

## Tech Stack

- **Backend**: FastAPI  
- **Agents / Orchestration**: LangChain, LangGraph, DeepAgents  
- **Evaluation & Tracing**: LangSmith  
- **Retrieval**: RAG over ICD-10 guidelines and tabular codes  
- **Architecture**: Object-Oriented Python  


This project explores the application of **agentic AI systems in clinical decision support**, focusing on **accuracy, explainability, and evaluation-driven development** for medical coding workflows.
