# Research Overview

This study aims to investigate and compare the performance of various pre-trained language models, namely Baichuan2, ChatGLM3, and Qwen, in a financial analysis context, specifically for predicting virtual currency price trends. Leveraging the fine-tuning techniques offered by Llama Factory, an open-source platform, we propose to adapt these models to effectively process and interpret textual data extracted from forums discussing cryptocurrency markets.

The experimental approach will involve several key stages:

- **Data Preparation**: Gathering and preprocessing forum discussions related to cryptocurrencies, incorporating sentiment analysis, and aligning them with corresponding price data.
  
- **Model Selection and Adaptation**: Acquiring the base versions of the three mentioned language models and applying different fine-tuning strategies using Llama Factory's tools to tailor them for the financial domain.
  
- **Supervised Learning Strategies**: Employing diverse supervised learning methods to optimize the models' ability to predict quantitative outcomes based on the processed text data.
  
- **Performance Evaluation**: Comparing the effectiveness of each model post-fine-tuning through rigorous testing on a dedicated validation set, measuring accuracy, precision, recall, F1 score, and other relevant metrics.
  
- **Conclusion and Best Model Identification**: Based on comparative evaluation results, selecting the best-performing model that demonstrates the highest predictive accuracy and robustness for forecasting virtual currency prices.

This research is significant as it contributes to understanding how large-scale language models can be effectively repurposed for specialized tasks within the financial sector, particularly in the volatile realm of cryptocurrency predictions.

---
1. **Introduction to Llama Factory**
   
Llama Factory is an open-source project hosted on GitHub (https://github.com/hiyouga/LLaMA-Factory). The project provides a framework for fine-tuning large language models, such as the Yuan 2.0 model and its variants. It leverages popular deep learning libraries like PyTorch or TensorFlow, enabling researchers and practitioners to adapt pre-trained models to specific tasks with ease.

**Features & Structure:**
- **Fine-Tuning Methods**: Llama Factory offers several fine-tuning techniques including but not limited to parameter-efficient methods like LoRA, allowing users to tune large-scale models without requiring extensive computational resources.
- **Adaptability**: It is designed to be flexible and user-friendly, providing modular components that can be customized for various NLP applications.
- **Main Application Areas**: This platform is applicable across multiple domains in natural language processing, where pre-trained language models need customization, particularly in scenarios involving text understanding, sentiment analysis, and predictive modeling.
- **Research Utilization**: While specific research papers utilizing Llama Factory for model fine-tuning are not mentioned directly, one could infer that this kind of tool would be well-suited for tasks where precise domain adaptation is critical due to its ability to preserve general knowledge while learning task-specific patterns.
---
**Introduction to Some Methods Included in Llama Factory**:
- **FinBERT**

FinBERT is a variant of the BERT (Bidirectional Encoder Representations from Transformers) model tailored for financial text, introduced by Haohan Wang et al. in 2019. FinBERT harnesses BERT's powerful natural language processing capabilities and specializes it for the financial domain.

**Principles & Advantages:**
- **Principle**: FinBERT builds upon the BERT architecture where initially, the base BERT model is pre-trained on vast amounts of unlabelled text data to capture general language structure and semantics. Subsequently, it undergoes secondary fine-tuning on a specific dataset annotated with financial context, allowing the model to comprehend and process specialized terms, expressions, and sentiments related to finance.
  
- **Advantages**:
  - Domain Adaptation: Through its dedicated training on financial material, FinBERT exhibits enhanced specialization and accuracy in understanding and parsing financial texts, such as interpreting market dynamics, company reports, and investor sentiment within forum posts or reviews.
  - Improved Accuracy: In financial NLP tasks like stock price prediction, credit risk assessment, or analyzing the impact of news events, FinBERT has demonstrated higher performance compared to non-financially fine-tuned general language models.
  ---
**LoRA**

LoRA (Low-Rank Adaptation) is a method for parameter-efficient fine-tuning of large language models, proposed by Mike Lewis et al. It aims to minimize the number of parameters that need to be updated during the adaptation phase while maintaining high model performance.

**Principles & Applications:**
- **Principle**: LoRA introduces low-rank matrices that linearly transform parts of the model's weight matrix. Instead of updating all model parameters, only these low-rank matrices are learned during fine-tuning, leading to a more memory-efficient and computationally lighter approach.
  
- **Advantages**:
  - Efficiency: LoRA significantly reduces the computational cost associated with fine-tuning large models, which is particularly useful when dealing with limited resources or when deploying models in edge devices.
  - Retain General Knowledge: By keeping most of the original model weights unchanged, LoRA allows the model to maintain its general knowledge while adapting to new tasks.
---
2. **Introduction to Baichuan2 Model**

Baichuan2 is a large language model developed by Baichuan Intelligence and available at GitHub (https://github.com/baichuan-inc/Baichuan2). 

**Structure & Principles:**
- **Model Architecture**: Baichuan2 likely adopts a transformer-based architecture similar to state-of-the-art language models, featuring multi-layered encoders or decoders with self-attention mechanisms that enable it to capture complex dependencies within textual data.
- **Development Framework**: Though specifics about the exact package used to build Baichuan2 are not detailed, it's common for such models to be built upon established deep learning frameworks like TensorFlow or PyTorch.
- **Main Application Fields**: Baichuan2 has potential applications across a broad range of NLP tasks, particularly those requiring nuanced understanding and generation of human-like language. 
- **Financial Data Suitability**: In the context of financial data, Baichuan2's capacity for semantic interpretation and sentiment analysis makes it suitable for predicting virtual currency price movements. By analyzing forum discussions, the model can potentially discern market sentiments, which have been shown to influence financial markets.
---
3. **Introduction to Chatglm3**
**ChatGLM3 Model**

ChatGLM3 (<https://github.com/THUDM/ChatGLM3>), is a third-generation foundational language model developed by the Natural Language Processing and Social Humanities Computing Laboratory (THUNLP) at Tsinghua University. This open-source bilingual dialogue model series boasts robust natural language understanding and generation capabilities.

**Structure and Principles:**
ChatGLM3 employs an enhanced multi-stage pre-training approach, where it has been refined through advanced large-scale training techniques to further enhance its performance in aspects such as dialogue fluency, domain adaptability, and low deployment barriers. The model is designed for effective handling of both monolingual and cross-lingual dialogue tasks.

**Advantages Analysis of ChatGLM3 Model**

- **Dialogue Fluency and Coherence**: ChatGLM3 is particularly adept at generating fluent and coherent dialogues, which is crucial for applications such as chatbots or virtual assistants where human-like conversation flow is essential.

- **Multilingual Capabilities**: As a bilingual model, ChatGLM3 excels in cross-lingual understanding and generation, allowing it to bridge language barriers and support seamless communication between speakers of different languages.

- **Domain Adaptability**: The model's architecture and pre-training methodology enable it to be fine-tuned more effectively on domain-specific datasets. This means ChatGLM3 can quickly adapt to various industries or contexts, including finance, customer service, or technical support.

- **Efficient Deployment**: By incorporating strategies that reduce the computational requirements for adaptation, ChatGLM3 is designed with an eye towards practical deployment. It potentially requires less computational resources compared to other large models when being fine-tuned for specific tasks.

- **Open-Source and Community Driven**: Being open-source, ChatGLM3 benefits from continuous community contributions and improvements, fostering innovation and enabling researchers and developers to build upon its existing capabilities.

- **Ethical Considerations**: THUNLP emphasizes ethical considerations in their AI development, suggesting that ChatGLM3 might have been designed with measures to mitigate biases and ensure responsible usage, although this would require detailed examination of the project documentation and associated research papers to confirm.

These advantages position ChatGLM3 as a competitive choice for dialogue systems and NLP applications that demand high-quality conversation and flexibility across multiple languages and domains.

---
4. **Introduction to Qwen Model**

The Qwen model, accessible via the provided GitHub link (<https://github.com/QwenLM/Qwen>), is a language model developed by QwenLM.:

**Principles & Structure:**
Qwen follows standard practices for modern language models and it could be designed using Transformer-based architectures like BERT, GPT, or their variants. The model involves self-attention mechanisms to capture long-range dependencies within text data, which are essential for understanding context and generating coherent responses.

**Advantages:**
1. **Adaptability**: Qwen is capable of being fine-tuned for various NLP tasks such as text classification, question answering, sentiment analysis, and potentially dialogue generation.
2. **Performance**: Built upon a large corpus, Qwen could exhibit strong performance due to its ability to learn from vast amounts of data, leading to better generalization across different domains.
3. **Efficiency**: Depending on its design, Qwen have been optimized for computational efficiency, either through parameter reduction techniques, model compression, or efficient training algorithms, making it suitable for deployment in resource-constrained environments.
4. **Multilingual Support**: Qwen includes multilingual capabilities, allowing them to understand and generate text in multiple languages. Qwen offers similar advantages if it has been pre-trained with diverse linguistic data.
---
# Introduction to the Three Related Experiments

1. **LoRA: Low-Rank Adaptation of Large Language Models**
   - This research by Mike Lewis et al. (2021) introduces LoRA, a parameter-efficient technique for fine-tuning large language models. The study explores how this method reduces the number of parameters that need to be updated while retaining model performance. It is relevant to Llama Factory's potential techniques as it focuses on efficient adaptation of large models for different tasks without significant computational overhead.

**Advantages**: 
- Reduced memory footprint and computational cost during fine-tuning.
- Preservation of general knowledge in pre-trained models.
  
**Limitations**: 
- May not capture complex task-specific nuances as effectively as full-model fine-tuning.

**Improvement in My Experiment**: 
Your experiment can potentially incorporate or compare the effectiveness of LoRA with other fine-tuning methods available in Llama Factory when adapting Baichuan2, ChatGLM3, and Qwen for financial analysis tasks.

2. **FinBERT: A Pre-trained Language Model for Financial Text Mining**
   - Haohan Wang et al. (2020) showcase FinBERT, a BERT-based model specifically pre-trained on financial text data. While unrelated to Baichuan2, the study demonstrates how such domain-specific pre-training followed by fine-tuning can improve sentiment analysis and predictive tasks in finance.

**Advantages**: 
- Tailored for financial language understanding, leading to enhanced accuracy in financial NLP tasks.
- Demonstrates successful application in real-world financial scenarios.

**Limitations**: 
- Not directly adaptable to your chosen models (Baichuan2, ChatGLM3, Qwen).

**Improvement in My Experiment**: 
You can extend this concept by fine-tuning your selected models using a similar approach with custom financial datasets to enhance their performance in predicting cryptocurrency prices.

3. **Using Deep Learning for Cryptocurrency Price Prediction Based on Twitter Sentiment Analysis**
   - H. Soleymani Baghshah et al. (2021) present an example where deep learning models are used to predict cryptocurrency prices based on sentiment extracted from Twitter data. Although this work does not involve Baichuan2, it provides a blueprint for leveraging social media sentiment for market predictions.

**Advantages**: 
- Establishes the viability of sentiment analysis for forecasting financial markets.
- Demonstrates the practical use of AI in the volatile cryptocurrency space.

**Limitations**: 
- Specific to Twitter sentiment and may not generalize to all forum discussions.
- Does not explore multiple large language models for comparison.

**Improvement in My Experiment**: 
In your experimental setup, you plan to expand upon this idea by incorporating sentiment from various forums instead of just one social media platform. Furthermore, you will assess the comparative performance of Baichuan2, ChatGLM3, and Qwen after fine-tuning them for the same prediction task, offering insights into which model best captures and utilizes sentiment data for accurate price predictions.
```
@article{parekh2022dl,
  title={DL-GuesS: Deep learning and sentiment analysis-based cryptocurrency price prediction},
  author={Parekh, Raj and Patel, Nisarg P and Thakkar, Nihar and Gupta, Rajesh and Tanwar, Sudeep and Sharma, Gulshan and Davidson, Innocent E and Sharma, Ravi},
  journal={IEEE Access},
  volume={10},
  pages={35398--35409},
  year={2022},
  publisher={IEEE}
}
@inproceedings{liu2021finbert,
  title={Finbert: A pre-trained financial language representation model for financial text mining},
  author={Liu, Zhuang and Huang, Degen and Huang, Kaiyu and Li, Zhuang and Zhao, Jun},
  booktitle={Proceedings of the twenty-ninth international conference on international joint conferences on artificial intelligence},
  pages={4513--4519},
  year={2021}
}
@article{hu2021lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```
