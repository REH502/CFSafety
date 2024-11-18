# CFSafety: Comprehensive Fine-grained Safety Assessment for LLMs

[CFSafety](https://github.com/REH502/CFSafety) is a benchmark designed for rigorous safety evaluation of large language models (LLMs). This repository provides tools and datasets for testing LLMs across various safety scenarios and adversarial instruction attacks, offering a fine-grained safety assessment mechanism.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
As large language models become increasingly integrated into daily life, their safety and ethical implications have garnered significant attention. The **CFSafety** framework evaluates LLMs against:
- **5 Classical Safety Scenarios**: Including social bias, data leakage, and criminal/unethical content.
- **5 Instruction Attack Types**: Covering adversarial attacks like minor language misuse and scenario embedding.

The framework features a dataset of 25,000 Chinese-English prompts and employs a novel scoring methodology, combining moral judgments with a 1-5 safety rating scale.

---

## Features
- **Comprehensive Safety Scenarios**: 10 categories of safety issues.
- **Bilingual Dataset**: Chinese and English prompts.
- **Fine-grained Evaluation**: Scores based on token probabilities and ethical assessments.
- **Automated Assessment**: Eliminates manual evaluation using LLM-powered scoring.

---

## Installation
To set up and run the CFSafety framework:

1. Clone the repository:
   ```bash
   git clone https://github.com/REH502/CFSafety.git
   cd CFSafety
