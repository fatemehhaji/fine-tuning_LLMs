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

After fine-tuning, we measure various metrics to evaluate the quality of the generated text, including perplexity, BLEU score, ROUGE-L score, and BERTScore. We also conduct a small-scale human evaluation to assess the generated text's grammatical correctness, coherence, and correctness of the answer.

Discussion - The results from the table show that the phi-2-finetuned model slightly outperforms the Llama-2-7b-hf-finetuned and Mistral-7B-finetuned models in terms of BLEU, ROUGE-L, Perplexity, and BERTScore metrics. The BLEU scores are low for all models, suggesting that this metric may not be the best indicator of performance for this specific task. On the other hand, BERTScores are notably higher, indicating a better alignment with semantic similarity in the generated text. Human evaluation scores are also relatively high for all models, suggesting that the generated text is comprehensible to humans. The generation of extra tokens seems to affect all models similarly, as reflected in their scores across the different metrics.

Generated using settings: top_k = 50, num_beams = 5, and temperature = 1.

| Model Name                            | BLEU     | ROUGE-L   | BERTScore | Perplexity | Human Evaluation |
|---------------------------------------|----------|-----------|-----------|------------|------------------|
| phi-2-finetuned_topk50_nb5_t1         | 0.469636 | 0.604907  | 0.931351  | 19.0088    | 0.895062         |
| Llama-2-7b-hf-finetuned_topk50_nb5_t1 | 0.137808 | 0.38557   | 0.897052  | 10.0402    | 0.919753         |
| Mistral-7B-finetuned_topk50_nb5_t1    | 0.110753 | 0.304393  | 0.855278  | 8.10991    | 0.845679         |

## Hyperparameter Tuning

We explore the impact of different hyperparameters (top_k, beam_size, and temperature) on the text generation capabilities of the fine-tuned LLMs. We conduct experiments with varying parameter settings and measure their effects using the defined metrics.

Discussion - The presented data suggests that model performance is sensitive to parameter adjustments, with varying degrees of impact across different metrics. Increasing top_k seems to influence semantic metrics such as BERTScore, possibly introducing noise with a wider selection of tokens. Perplexity trends increase with more beams in one model, hinting at increased uncertainty or exploration in the prediction space. Surprisingly, temperature modifications do not show a significant effect, suggesting a potential robustness in the models or an insufficient range to elicit a notable response. These insights highlight the complex interplay between parameter tuning and model behavior, warranting further investigation to fully understand the underlying mechanisms.

![alt text](https://github.com/fatemehhaji/fine-tuning_LLMs/blob/main/images/parameters_evaluation.jpeg)

## References

- https://www.datacamp.com/tutorial/fine-tuning-llama-2
- https://www.kaggle.com/code/kingabzpro/fine-tuning-phi-2


