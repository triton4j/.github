# Triton4J

> Bringing Python Triton AI kernel development to the Java ecosystem

## Overview

Triton4J is an innovative project that bridges the gap between Python's [Triton](https://github.com/openai/triton) language for GPU programming and the Java ecosystem. This project enables dynamically handling, translating, and executing Python Triton AI kernel code within a Java-based environment using reflection and bytecode manipulation.

## 🎯 Mission

Our goal is to make GPU kernel development accessible to Java developers by providing tools to work with Triton-style kernels directly in Java, leveraging the power of Project Babylon and modern Java capabilities.

## 📦 Projects

### [triton4j-parser](https://github.com/triton4j/triton4j-parser)

The core parser and code generator that converts Triton-style Python kernels into Java source code.

**Key Features:**
- Parses Triton-style Python kernels
- Generates Java source code compatible with Babylon/Triton APIs
- Implements a complete Python parser in Java
- Provides a robust code generation pipeline for GPU-kernel workflows

**Language:** Java  
**License:** Apache License 2.0

## 🚀 Getting Started

To start using Triton4J in your projects, check out the [triton4j-parser](https://github.com/triton4j/triton4j-parser) repository for installation instructions and usage examples.

## 🛠️ Technology Stack

- **Java** - Primary implementation language
- **Project Babylon** - Java runtime for GPU kernels
- **Triton API** - GPU kernel programming model
- **Bytecode Manipulation** - Dynamic code generation and execution

## 📄 License

All Triton4J projects are licensed under the Apache License 2.0.

## 🤝 Contributing

We welcome contributions! Please check individual repository contribution guidelines for more details.

## 📧 Contact

For questions, issues, or discussions, please use the issue trackers in the respective repositories.

---

**Note:** This is an experimental project exploring the integration of Python Triton kernels with Java. APIs and features are subject to change as the project evolves.
