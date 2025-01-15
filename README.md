# Question-Answering using GPT-2's PEFT with LoRA


<div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/qa-gpt2-lora/blob/master/Question_Answering_GPT-2_PEFT_LoRA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
</div>


Having constrained resources, the most rational way to fine-tune the language model with many parameters is to perform PEFT (parameter-efficient fine-tuning). One of the preferred PEFT methods is LoRA (low-rank adaptation). LoRA can decompose a complex neural network matrix (W) into two smaller matrices (A × B). These matrices are leveraged to re-parameterize the frozen weight of the language model (y = (W + A × B) × x). In this fashion, we can fine-tune the model inexpensively. In this project, we use GPT-2 as the baseline model. Then, LoRA is applied to the attention and linear layers. We fine-tune the model to carry out the question-answering task on the SQuAD 2.0 dataset. Next, the model is evaluated with BLEU 1-gram. The model, LoRA, dataset, and evaluation are available thanks to the Hugging Face ecosystem.


## Experiment

Go to this [notebook](https://github.com/reshalfahsi/qa-gpt2-lora/blob/master/Question_Answering_GPT-2_PEFT_LoRA.ipynb) for the inquiry.


## Result

## Quantitative Result

We can measure the model performance using BLEU 1-gram.

Test Metric | Score  |
----------- | -----  |
BLEU 1-gram | 13.89% |


## Qualitative Result

This image showcases the model's QA result.

<p align="center"> <img src="https://github.com/reshalfahsi/qa-gpt2-lora/blob/master/assets/qualitative.png" alt="qualitative" > <br /> Testing the model on a Winograd schema question. </p>


## Credit

- [Geek Out Time: Exploring LoRA on Google Colab: the Challenges of Base Model Upgrades](https://www.linkedin.com/pulse/geek-out-time-exploring-lora-google-colab-challenges-base-nedved-yang-79drc)
- [GPT-2 Fine-Tuning Tutorial with PyTorch & Huggingface in Colab](https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh#scrollTo=sh0XKuDvnryn)
- [LoRA](https://colab.research.google.com/github/DanielWarfield1/MLWritingAndResearch/blob/main/LoRA.ipynb)
- [Squad_v2 Question-Answering Roberta](https://www.kaggle.com/code/stpeteishii/squad-v2-question-answering-roberta)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/pdf/1806.03822)
- [A Collection of Winograd Schemas](https://cs.nyu.edu/~davise/papers/WSOld.html)
- [🤗 PEFT](https://github.com/huggingface/peft)
- [🤗 Evaluate](https://github.com/huggingface/evaluate)
- [🤗 Datasets](https://github.com/huggingface/datasets)
- [🤗 Transformers](https://github.com/huggingface/transformers)
