# 📄 Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning

![PaperCoder Overview](./assets/papercoder_overview.png)

📄 [Read the paper on arXiv](https://arxiv.org/abs/2504.17192)

**PaperCoder** is a multi-agent LLM system that transforms paper into a code repository.
It follows a three-stage pipeline: planning, analysis, and code generation, each handled by specialized agents with hierarchical memory support.  
Our method outperforms strong baselines on both Paper2Code and PaperBench and produces faithful, high-quality implementations.

## ✨ New: Hierarchical Memory System

Paper2Code now includes an advanced memory system that learns from previous papers and code implementations:

- 🧠 **Node Memory**: Per-agent context tracking for real-time decision-making
- 💾 **Short-Term Memory**: Session-level context for current paper processing
- 🗄️ **Long-Term Memory**: Persistent learning across sessions with vector-based retrieval
- 🔍 **Semantic Search**: Retrieve similar papers and code patterns using embeddings

This memory system integrates insights from [CodeGen](https://arxiv.org/abs/2203.13474), [AlphaCode](https://arxiv.org/abs/2203.07814), and our PaperCoder research.

[📖 Read the Memory System Documentation](./MEMORY_SYSTEM.md)

---

## 🗺️ Table of Contents

- [⚡ Quick Start](#-quick-start)
- [🏗️ Architecture](#-architecture)
- [🧠 Memory System](#-memory-system)
- [📚 Detailed Setup Instructions](#-detailed-setup-instructions)
- [📦 Paper2Code Benchmark Datasets](#-paper2code-benchmark-datasets)
- [📊 Model-based Evaluation of Repositories](#-model-based-evaluation-of-repositories-generated-by-papercoder)
- [🤝 Contributing](#-contributing)

---

## ⚡ Quick Start
- Note: The following command runs example paper ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).  

### Using OpenAI API
- 💵 Estimated cost for using o3-mini: $0.50–$0.70

```bash
pip install openai

export OPENAI_API_KEY="<OPENAI_API_KEY>"

cd scripts
bash run.sh
```

### Using Open Source Models with vLLM
- If you encounter any issues installing vLLM, please refer to the [official vLLM repository](https://github.com/vllm-project/vllm).
- The default model is `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`.

```bash
pip install vllm

cd scripts
bash run_llm.sh
```

### Output Folder Structure (Only Important Files)
```bash
outputs
├── Transformer
│   ├── analyzing_artifacts
│   ├── coding_artifacts
│   └── planning_artifacts
└── Transformer_repo # Final output repository
```
---

## 🏗️ Architecture

Paper2Code uses a **three-stage multi-agent pipeline** where each stage is handled by specialized agents:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Paper2Code System                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY SYSTEM (New!)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Node Memory  │  │ Short-Term   │  │   Long-Term Memory   │  │
│  │ (Per-Agent)  │  │   Memory     │  │ (Cross-Session)      │  │
│  │              │  │ (Session)    │  │ • Similar Papers     │  │
│  │ • Planning   │  │              │  │ • Code Patterns      │  │
│  │ • Analyzing  │  │ • Paper      │  │ • Planning Strategies│  │
│  │ • Coding     │  │ • Planning   │  │ • Vector Embeddings  │  │
│  │              │  │ • Analysis   │  │                      │  │
│  │              │  │ • Code       │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────┴────────────────────┐
         │                                          │
         ▼                                          ▼
┌─────────────────┐                      ┌─────────────────┐
│  STAGE 1:       │                      │   Input Paper   │
│  PLANNING       │◄─────────────────────│   (PDF/LaTeX)   │
│                 │                      └─────────────────┘
│ • Overall Plan  │
│ • Architecture  │  Retrieves similar papers
│ • Task List     │  and strategies from memory
│ • Config        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  STAGE 2:       │
│  ANALYZING      │
│                 │  Retrieves code patterns
│ • Logic         │  from long-term memory
│   Analysis      │
│ • File-by-File  │
│   Review        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  STAGE 3:       │
│  CODING         │
│                 │  Uses retrieved patterns
│ • Code          │  and previous context
│   Generation    │
│ • File Writing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Repo    │
│  • model.py     │
│  • trainer.py   │
│  • config.yaml  │
│  • ...          │
└─────────────────┘
```

### Key Components

1. **Planning Agent**
   - Analyzes paper methodology and experiments
   - Creates implementation roadmap
   - Designs architecture and file structure
   - Generates configuration templates
   - **Memory Integration**: Retrieves similar papers and planning strategies

2. **Analyzing Agent**
   - Performs detailed logic analysis for each file
   - Identifies dependencies and interfaces
   - Plans implementation details
   - **Memory Integration**: Uses node memory to track file-level context

3. **Coding Agent**
   - Generates actual code for each file
   - Follows architecture and logic analysis
   - Maintains consistency across files
   - **Memory Integration**: Retrieves relevant code patterns from memory

---

## 🧠 Memory System

The memory system enables Paper2Code to learn from previous papers and improve over time.

### Three-Level Hierarchy

1. **Node Memory** (Per-Agent, Real-Time)
   - Tracks recent agent actions and decisions
   - Maintains current context state
   - 50-entry rolling window per agent
   - Fast in-memory lookups

2. **Short-Term Memory** (Session-Level)
   - Stores current paper being processed
   - Accumulates planning, analysis, and code
   - Persisted at session end
   - Can resume interrupted sessions

3. **Long-Term Memory** (Cross-Session, Persistent)
   - Vector-indexed paper repository
   - Code pattern library
   - Planning strategy database
   - Similarity-based retrieval

### Usage Example with Memory

```bash
# Run planning with memory integration (recommended)
python codes/1_planning_memory.py \
    --paper_name "Transformer" \
    --gpt_version "o3-mini" \
    --pdf_json_path "./examples/Transformer_cleaned.json" \
    --output_dir "./outputs/Transformer" \
    --memory_dir "./memory_store"
```

The memory system will:
1. Generate an embedding for your paper
2. Retrieve 3 most similar papers from memory
3. Find relevant code patterns based on paper content
4. Enhance prompts with retrieved context
5. Store your paper and successful patterns for future use

[📖 Full Memory System Documentation](./MEMORY_SYSTEM.md)

---

## 📚 Detailed Setup Instructions

### 🛠️ Environment Setup

- 💡 To use the `o3-mini` version, make sure you have the latest `openai` package installed.
- 📦 Install only what you need:
  - For OpenAI API: `openai`
  - For open-source models: `vllm`
      - If you encounter any issues installing vLLM, please refer to the [official vLLM repository](https://github.com/vllm-project/vllm).


```bash
pip install openai 
pip install vllm 
```

- Or, if you prefer, you can install all dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Memory System Dependencies

For the memory system features, you'll also need:

```bash
# Vector embeddings support
pip install numpy>=1.24.0

# For local embeddings (optional, for offline use)
pip install sentence-transformers>=2.2.0
```


### 📄 (Option) Convert PDF to JSON
The following process describes how to convert a paper PDF into JSON format.  
If you have access to the LaTeX source and plan to use it with PaperCoder, you may skip this step and proceed to [🚀 Running PaperCoder](#-running-papercoder).  
Note: In our experiments, we converted all paper PDFs to JSON format.

1. Clone the `s2orc-doc2json` repository to convert your PDF file into a structured JSON format.  
   (For detailed configuration, please refer to the [official repository](https://github.com/allenai/s2orc-doc2json).)

```bash
git clone https://github.com/allenai/s2orc-doc2json.git
```

2. Run the PDF processing service.

```bash
cd ./s2orc-doc2json/grobid-0.7.3
./gradlew run
```

3. Convert your PDF into JSON format.

```bash
mkdir -p ./s2orc-doc2json/output_dir/paper_coder
python ./s2orc-doc2json/doc2json/grobid2json/process_pdf.py \
    -i ${PDF_PATH} \
    -t ./s2orc-doc2json/temp_dir/ \
    -o ./s2orc-doc2json/output_dir/paper_coder
```

### 🚀 Running PaperCoder
- Note: The following command runs example paper ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).  
  If you want to run PaperCoder on your own paper, please modify the environment variables accordingly.

#### Using OpenAI API
- 💵 Estimated cost for using o3-mini: $0.50–$0.70


```bash
# Using the PDF-based JSON format of the paper
export OPENAI_API_KEY="<OPENAI_API_KEY>"

cd scripts
bash run.sh
```

```bash
# Using the LaTeX source of the paper
export OPENAI_API_KEY="<OPENAI_API_KEY>"

cd scripts
bash run_latex.sh
```


#### Using Open Source Models with vLLM
- The default model is `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`.

```bash
# Using the PDF-based JSON format of the paper
cd scripts
bash run_llm.sh
```

```bash
# Using the LaTeX source of the paper
cd scripts
bash run_latex_llm.sh
```

---

## 📦 Paper2Code Benchmark Datasets
- Huggingface dataset: [paper2code](https://huggingface.co/datasets/iaminju/paper2code)
  
- You can find the description of the Paper2Code benchmark dataset in [data/paper2code](https://github.com/going-doer/Paper2Code/tree/main/data/paper2code). 
- For more details, refer to Section 4.1 "Paper2Code Benchmark" in the [paper](https://arxiv.org/abs/2504.17192).


---

## 📊 Model-based Evaluation of Repositories Generated by PaperCoder

- We evaluate repository quality using a model-based approach, supporting both reference-based and reference-free settings.  
  The model critiques key implementation components, assigns severity levels, and generates a 1–5 correctness score averaged over 8 samples using **o3-mini-high**.

- For more details, please refer to Section 4.3.1 (*Paper2Code Benchmark*) of the paper.
- **Note:** The following examples evaluate the sample repository (**Transformer_repo**).  
  Please modify the relevant paths and arguments if you wish to evaluate a different repository.

### 🛠️ Environment Setup
```bash
pip install tiktoken
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```


### 📝 Reference-free Evaluation
- `target_repo_dir` is the generated repository.

```bash
cd codes/
python eval.py \
    --paper_name Transformer \
    --pdf_json_path ../examples/Transformer_cleaned.json \
    --data_dir ../data \
    --output_dir ../outputs/Transformer \
    --target_repo_dir ../outputs/Transformer_repo \
    --eval_result_dir ../results \
    --eval_type ref_free \
    --generated_n 8 \
    --papercoder
```

### 📝 Reference-based Evaluation
- `target_repo_dir` is the generated repository.
- `gold_repo_dir` should point to the official repository (e.g., author-released code).

```bash
cd codes/
python eval.py \
    --paper_name Transformer \
    --pdf_json_path ../examples/Transformer_cleaned.json \
    --data_dir ../data \
    --output_dir ../outputs/Transformer \
    --target_repo_dir ../outputs/Transformer_repo \
    --gold_repo_dir ../examples/Transformer_gold_repo \
    --eval_result_dir ../results \
    --eval_type ref_based \
    --generated_n 8 \
    --papercoder
```


### 📄 Example Output
```bash
========================================
🌟 Evaluation Summary 🌟
📄 Paper name: Transformer
🧪 Evaluation type: ref_based
📁 Target repo directory: ../outputs/Transformer_repo
📊 Evaluation result:
        📈 Score: 4.5000
        ✅ Valid: 8/8
========================================
🌟 Usage Summary 🌟
[Evaluation] Transformer - ref_based
🛠️ Model: o3-mini
📥 Input tokens: 44318 (Cost: $0.04874980)
📦 Cached input tokens: 0 (Cost: $0.00000000)
📤 Output tokens: 26310 (Cost: $0.11576400)
💵 Current total cost: $0.16451380
🪙 Accumulated total cost so far: $0.16451380
============================================
```

---

## 🤝 Contributing

We welcome contributions to Paper2Code! Whether you want to:
- Add support for new LLM providers
- Improve the memory system
- Add tests and documentation
- Fix bugs or enhance features

Please see our [Contributing Guide](./CONTRIBUTING.md) for:
- Development setup instructions
- Code style guidelines
- Testing requirements
- How to submit pull requests

### Areas We Need Help With

- **Testing**: Adding unit and integration tests
- **Memory System**: Improving retrieval algorithms and eviction policies
- **Documentation**: More examples and tutorials
- **Support**: Additional LLM backends (Anthropic, Cohere, etc.)

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use Paper2Code in your research, please cite:

```bibtex
@article{papercoder2025,
  title={PaperCoder: Automated Code Generation from Scientific Papers},
  author={Paper2Code Team},
  journal={arXiv preprint arXiv:2504.17192},
  year={2025}
}
```

---

## 🙏 Acknowledgments

This project integrates insights from:
- [CodeGen](https://arxiv.org/abs/2203.13474): Multi-turn conversation and context management
- [AlphaCode](https://arxiv.org/abs/2203.07814): Code clustering and solution retrieval
- [PaperCoder](https://arxiv.org/abs/2504.17192): Cross-paper learning and methodology awareness

Special thanks to the open-source community for making this possible.

---

## 📧 Contact

For questions, suggestions, or collaboration:
- Open an issue on [GitHub](https://github.com/Vikaash-dev/Paper2Code/issues)
- Check our [Discussions](https://github.com/Vikaash-dev/Paper2Code/discussions)

---

**Made with ❤️ by the Paper2Code Team**
