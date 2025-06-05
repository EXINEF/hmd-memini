# 🧠 Memini - Conversational Task Manager

**Memini** is a natural language conversational assistant designed to help you manage tasks, to-do lists, and deadlines through intuitive dialogue powered by large language models (LLMs). Whether you're a student, a professional, or just someone who loves smart productivity tools, Memini understands your goals and gets things done, just by chatting.

> 🏫 **This project was developed as part of the “Human-Machine Dialogue (HMD)” course at the University of Trento (UniTN).**

---

## 🚀 Features

* 📅 **Add Tasks** – e.g., “Add a task to submit my report by Friday.”
* ✅ **Mark Tasks as Done** – e.g., “Mark ‘buy groceries’ as complete.”
* ❌ **Delete Tasks** – with confirmation.
* 🔍 **Retrieve Tasks** – e.g., “What do I have to do next week?”
* ✏️ **Update Tasks** – e.g., “Change ‘call mom’ to ‘call dad’ tomorrow.”
* 🧠 **Multi-intent Handling** – Supports multiple tasks in a single sentence.
* 💬 **Natural Language Dialogue** – Maintains context, confirms actions, asks for clarification.
* ⚙️ **Hybrid DM Options** – Choose between deterministic rules or LLM-based dialogue management.


## 🧪 Models Supported

You can run Memini with the following LLMs:

| Model         | Description                                     |
| ------------- | ----------------------------------------------- |
| `llama3`      | Highest quality, best performance               |
| `llama3_8bit` | Quantized version, less memory usage, medium performance            |
| `llama3_4bit` | Quantized version, even less memory usage, low performance |
| `llama2`      | Older version, low performance     |
| `tinyllama`   | Very small model for quick or edge deployment, bad performance   |
| `llama4`      | Latest model, very large—CPU/GPU intensive        |

> ⚠️ **Note:** The performance of Memini varies significantly based on the model used. For best results, use `llama3` or `llama4` with sufficient GPU resources.
> 💡 Use 4-bit or 8-bit models if you're working on machines with limited GPU (e.g., 4GB VRAM).
> I was not able to test `llama4` due to its size and resource requirements, but it should improve performance significantly over `llama3`.

---

## 🧰 Requirements

* Python ≥ 3.8
* `torch` with CUDA support (if using GPU)
* Your selected model installed locally (e.g., `llama3`)

---

## ⚙️ Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/exinef/hmd-memini.git
   cd hmd-memini
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and configure your preferred model (Llama3, TinyLlama, etc.).

---

## ▶️ Running the Bot

Launch Memini using:

```bash
python main.py --model_name llama3 --cuda_device 0
```

### Command-line Arguments

| Argument             | Description                                                             |
| -------------------- | ----------------------------------------------------------------------- |
| `--model_name`       | Model to use (`llama3`, `llama3_8bit`, etc.)                            |
| `--logging_level`    | Set logging verbosity (`CRITICAL` -> use this if you want just to see the USER-BOT conversation without additional infos, `DEBUG`, `INFO`, `WARNING`, `ERROR`)             |
| `--cuda_device`      | GPU ID to use (e.g., 0) |
| `--deterministic_dm` | Enable deterministic (rule-based) dialogue manager instead of LLM-based, (True or False value), default value is True |

---

### Recommended Setup

**It is recommended to use the deterministic dialogue manager (enabled by default, no need to specify `--deterministic_dm`) together with `llama3`,** as this combination offers the best balance of speed and reliability. It reduces the number of model calls—requiring one less call per interaction—which improves performance and responsiveness.

While the LLM-based (non-deterministic) dialogue manager worked well initially, it started to show limitations as the bot's queries and functionalities became more complex. Despite carefully crafted prompts, sometimes it struggled to consistently select the correct actions. In contrast, the deterministic manager provides more stable and accurate behavior, making it better suited for advanced usage (even for future updates).

---

## 🧪 Evaluation

Memini includes comprehensive evaluation scripts to assess the performance of both the Dialogue Manager (DM) and Natural Language Understanding (NLU) components across different models.

### 📊 Dialogue Manager (DM) Evaluation

The DM evaluation script (`eval_dm.py`) tests how well different dialogue management approaches handle conversation flow and task completion.

#### Basic Usage (deterministic vs llama3) 

```bash
python eval_dm.py --models deterministic llama3 --cuda_device 0
```

#### Multiple Model Comparison

```bash
python eval_dm.py --models deterministic llama3 llama2 tinyllama --cuda_device 0
```

#### Background Execution with Output Logging

```bash
nohup python eval_dm.py --models deterministic llama3 llama2 tinyllama --cuda_device 0 > eval_dm_output.out 2>&1 &
```

#### DM Evaluation Arguments

| Argument      | Description                                           |
| ------------- | ----------------------------------------------------- |
| `--models`    | Space-separated list of models to evaluate           |
| `--cuda_device` | GPU device ID |

> 💡 **Tip:** The `deterministic` option evaluates the rule-based dialogue manager, while model names (e.g., `llama3`) evaluate LLM-based dialogue management.

### 🔍 Natural Language Understanding (NLU) Evaluation

The NLU evaluation script (`eval_nlu.py`) measures how accurately different models can understand and extract intents and entities from user messages.

#### Basic Usage (llama3)

```bash
python eval_nlu.py --models llama3 --cuda_device 0
```

#### Comprehensive Model Comparison

```bash
python eval_nlu.py --models llama3 llama3_8bit llama3_4bit llama2 tinyllama --cuda_device 0
```

#### Background Execution with Output Logging

```bash
nohup python eval_nlu.py --models llama3 llama3_8bit llama3_4bit llama2 tinyllama --cuda_device 0 > eval_nlu.out 2>&1 &
```

#### NLU Evaluation Arguments

| Argument          | Description                                         |
| ----------------- | --------------------------------------------------- |
| `--models`        | Space-separated list of models to evaluate         |
| `--cuda_device`   | GPU device ID |

---

### 📁 Output Files

Evaluation results, the task database (`database.json`), and the full interaction log (`complete_transcript.txt`) are saved in the `tmp/` folder.

> 🗂️ You can check this folder to review evaluation outputs, database and saved conversation.


## 📩 Contact

Developed by **Alessio Pierdominici**
📧 [alessio.pierdominici@studenti.unitn.it](mailto:alessio.pierdominici@studenti.unitn.it)
🎓 University of Trento – Human-Machine Dialogue (HMD), 2025