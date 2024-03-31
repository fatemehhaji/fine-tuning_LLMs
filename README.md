# Fine-Tuning Large Language Models (LLMs)

## Project Description

This project focuses on fine-tuning Large Language Models (LLMs) such as LLaMa-2, Mistral, and Phi-2 for text generation tasks. The fine-tuning process is conducted using PyTorch and the Hugging Face Transformers library.

## Setup

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/fatemehhaji/fine-tuning_LLMs.git


3. Run the notebooks:
- `finetune_llama2.ipynb`: Fine-tuning LLaMa-2 model
- `finetune_phi2.ipynb`: Fine-tuning Phi-2 model
- `finetune_mistral.ipynb`: Fine-tuning Mistral model
- `evaluation.ipynb`: Evaluating the fine-tuned models

## Fine-Tuning Pre-trained LLMs

We fine-tune three different LLMs on a text dataset. The fine-tuning process involves adapting the pre-trained model to generate text based on an instruction. We use the following models:

- LLaMa-2 (7B)
- Phi-2 (2.7B)
- Mistral (7B)

## Metric Measurements

After fine-tuning, we measure various metrics to evaluate the quality of the generated text, including perplexity, BLEU score, ROUGE-L score, BERTScore, and CodeBLEU. We also conduct a small-scale human evaluation to assess the generated text's grammatical correctness, coherence, and correctness of the answer.

## Hyperparameter Tuning

We explore the impact of different hyperparameters (top_k, beam_size, and temperature) on the text generation capabilities of the fine-tuned LLMs. We conduct experiments with varying parameter settings and measure their effects using the defined metrics.

## Discussions

Please refer to the `discussions.md` file for a detailed analysis and comparison of the models based on the metrics and human evaluation.

## Citation

If you use this code in your research, please cite the following sources:

- https://www.datacamp.com/tutorial/fine-tuning-llama-2
- https://www.kaggle.com/code/kingabzpro/fine-tuning-phi-2

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

