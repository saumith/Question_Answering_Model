# Question Answering Model

## Overview
This Jupyter Notebook fine-tunes a **RoBERTa-based question-answering model** using the **Hugging Face `transformers` library** and a custom dataset. The notebook walks through **data processing, training, and evaluation** for a **Question Answering (QA) system**.

## Requirements
To run this notebook, install the necessary dependencies:

```sh
pip install datasets transformers
```

Additionally, the notebook uses a **GPU** for model training. Ensure that **CUDA is available** on your system.

## Workflow
1. **Setup**
   - Installs required libraries
   - Disables Weights & Biases logging for simplicity

2. **Data Processing**
   - Loads a dataset from a JSON file (`dev.json`)
   - Extracts context-question-answer pairs
   - Cleans text (removes HTML entities, extra spaces)
   - Saves preprocessed data as `gpt_qa_dataset.jsonl`

3. **Model Selection & Tokenization**
   - Uses **RoBERTa (`deepset/roberta-base-squad2`)** as the base model
   - Prepares the dataset by tokenizing inputs
   - Aligns token positions for accurate answer extraction

4. **Training the Model**
   - Splits the dataset into training & testing sets
   - Defines training arguments:
     - Learning Rate: **3e-6**
     - Batch Size: **8**
     - Epochs: **7**
     - Weight Decay: **0.01**
   - Trains the model using **Hugging Face's `Trainer` class**
   - Saves the fine-tuned model to `./fine_tuned_qa_model`

5. **Inference (Testing the Model)**
   - Loads the fine-tuned model
   - Uses `pipeline()` to answer sample questions:
     - *"How many Ballon d'Or awards has Cristiano Ronaldo won?"*
     - *"How long is the Great Wall of China?"*
     - *"Who was the 44th President of the United States?"*

## Usage
To test the model, run:

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="./fine_tuned_qa_model", tokenizer="./fine_tuned_qa_model", device=0)

example = {
    "context": "The Eiffel Tower is located in Paris, France.",
    "question": "Where is the Eiffel Tower?"
}

result = qa_pipeline(question=example["question"], context=example["context"])
print(f"Predicted Answer: {result['answer']}")
```

## File Structure
```
├── QA_project.ipynb        # Main Jupyter Notebook
├── dev.json                # Raw dataset file
├── gpt_qa_dataset.jsonl    # Processed dataset
├── fine_tuned_qa_model/    # Trained QA model (saved weights)
├── results/                # Training logs & checkpoints
└── logs/                   # Training logs
```

## Future Enhancements
- Expand dataset with more diverse QA pairs
- Experiment with different transformer models (e.g., `bert-large-uncased-whole-word-masking-finetuned-squad`)
- Optimize training parameters for better accuracy
- Deploy as an API for real-world applications
