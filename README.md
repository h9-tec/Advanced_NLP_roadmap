
# Self-Study NLP Roadmap

This roadmap combines topics and references from **CMU’s 11-711 Advanced NLP** and **UMass Amherst’s CS 685**. It’s designed for a structured, step-by-step self-study over about **12–14 weeks** (or longer, depending on your schedule). Each “week” is flexible and can span 1–2 weeks of part-time study.

---

## Mind Map / Outline

Below is a simplified *mind map* illustrating how topics connect. It starts from the fundamentals of NLP and branches out to advanced areas like large language models (LLMs), retrieval-augmented generation (RAG), and interpretability.

<details>
<summary>Click to expand the Mind Map</summary>

```
                               ┌─────────────────────────┐
                               │     Week 1: Intro       │
                               │   & NLP Fundamentals    │
                               └─────────────┬───────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
 ┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
 │ Week 2: Word Reps &   │       │  Week 3: Language     │       │ Week 4: Sequence      │
 │  Text Classification  │       │  Modeling             │       │ Modeling (RNNs, LSTMs)│
 └───────────────────────┘       └───────────────────────┘       └───────────────────────┘
             │                               │                           │
             │                               ▼                           │
             ├──────────────┐   ┌───────────────────────────────────┐    │
             │              ▼   │  Week 5: Transformers & Attention │    │
             │  ┌────────────────────────────────────────────────────┘    │
             │  │                        │                                │
             │  ▼                        ▼                                ▼
 ┌───────────────────────┐       ┌─────────────────────────────┐  ┌───────────────────────────────┐
 │ Week 6: Text Gen &    │       │ Week 7: Instruction Tuning & │  │ Week 8: Experimental Design & │
 │ Prompting Basics      │       │  Efficient Fine-Tuning       │  │     Human Annotation          │
 └───────────────────────┘       └─────────────────────────────┘  └───────────────────────────────┘
             │                        │                                │
             │                        └───────────────┬────────────────┘
             │                                        │
             ▼                                        ▼
 ┌────────────────────────────┐            ┌────────────────────────────┐
 │ Week 9: Retrieval & RAG    │            │ Week 10: Distillation,     │
 │ (Retrieval-Augmented Gen)  │            │ Quantization & RL from HF  │
 └────────────────────────────┘            └────────────────────────────┘
             │                                        │
             └───────────────┬────────────────────────┘
                             │
                             ▼
                 ┌────────────────────────────┐
                 │ Week 11: Debugging &       │
                 │ Interpretation (Probing,   │
                 │ Mechanistic Interp.)       │
                 └─────────────┬──────────────┘
                               │
                               ▼
             ┌─────────────────────────────────────────────────────┐
             │ Week 12: Advanced LLMs, Agents & Long Contexts      │
             │ (LLaMa, GPT-4, Toolformer, ReAct, etc.)             │
             └─────────────┬───────────────────────────────────────┘
                           │
                           ▼
          ┌───────────────────────────────────────────────────┐
          │ Week 13: Complex Reasoning & Linguistics         │
          │ (Chain-of-thought, abductive reasoning, etc.)    │
          └─────────────┬─────────────────────────────────────┘
                        │
                        ▼
          ┌───────────────────────────────────────────────────┐
          │ Week 14: Multilingual NLP & Wrap-Up              │
          │ (mBERT, XLM-R, zero-shot cross-lingual)          │
          └───────────────────────────────────────────────────┘
```

</details>

---

## Table of Contents

1. [Week 1: Introduction & NLP Fundamentals](#week-1-introduction--nlp-fundamentals)  
2. [Week 2: Word Representations & Text Classification](#week-2-word-representations--text-classification)  
3. [Week 3: Language Modeling](#week-3-language-modeling)  
4. [Week 4: Sequence Modeling (RNNs, LSTMs, GRUs)](#week-4-sequence-modeling-rnns-lstms-grus)  
5. [Week 5: Transformers & Attention Mechanisms](#week-5-transformers--attention-mechanisms)  
6. [Week 6: Text Generation Algorithms & Prompting Basics](#week-6-text-generation-algorithms--prompting-basics)  
7. [Week 7: Instruction Tuning & Efficient Fine-Tuning Methods](#week-7-instruction-tuning--efficient-fine-tuning-methods)  
8. [Week 8: Experimental Design & Human Annotation](#week-8-experimental-design--human-annotation)  
9. [Week 9: Retrieval & Retrieval-Augmented Generation (RAG)](#week-9-retrieval--retrieval-augmented-generation-rag)  
10. [Week 10: Distillation, Quantization & RL from Human Feedback](#week-10-distillation-quantization--rl-from-human-feedback)  
11. [Week 11: Debugging & Interpretation](#week-11-debugging--interpretation)  
12. [Week 12: Advanced LLMs, Agents & Long Contexts](#week-12-advanced-llms-agents--long-contexts)  
13. [Week 13: Complex Reasoning & Linguistics](#week-13-complex-reasoning--linguistics)  
14. [Week 14: Multilingual NLP & Wrap-Up](#week-14-multilingual-nlp--wrap-up)  
15. [Additional Tips & Final Notes](#additional-tips--final-notes)

---

## Week 1: Introduction & NLP Fundamentals

**Core Topics**  
- Overview of Natural Language Processing (NLP)  
- Rule-based, statistical, and neural approaches  
- Introductory tasks: classification, tagging, QA, generation  

**Suggested References**  
- **CMU 11-711, Lecture 1 (Introduction)**
  - *Intro Slides*  
  - “Examining Power and Agency in Film” – Sap et al. (2017)
- **UMass CS 685, Week 1**  
  - Basic LM intros (Jurafsky & Martin, Sections 3.1–3.5 and 7)

**Practical Exercise**  
- Install a DL framework (e.g., PyTorch). Implement a simple **rule-based** text classifier vs. a **logistic regression** classifier on a small dataset.

---

## Week 2: Word Representations & Text Classification

**Core Topics**  
- Bag-of-words (BoW) and subword models (BPE, SentencePiece)  
- Continuous word embeddings (word2vec, GloVe)  
- Visualizing embeddings (t-SNE, PCA)

**Suggested References**  
- **CMU 11-711, Lecture 2**  
  - Sennrich et al. (2015) – Subword NMT  
  - Kudo (2018) – SentencePiece
- **UMass CS 685, Week 2**  
  - Bengio et al. (2003) – foundational neural LM  
  - Karpathy’s blog post on backprop basics

**Practical Exercise**  
- Train a **CNN or LSTM**-based text classifier using SentencePiece tokenization. Compare with a **bag-of-words** approach.

---

## Week 3: Language Modeling

**Core Topics**  
- N-gram language models  
- Neural LMs (feed-forward vs. RNN-based)  
- Perplexity, smoothing, log-linear models

**Suggested References**  
- **CMU 11-711, Lecture 3**  
  - Goodman (1998) – smoothing  
  - kenlm toolkit  
- **UMass CS 685**  
  - Jurafsky & Martin, sections 3.1–3.5 & 7

**Practical Exercise**  
- Implement a **count-based n-gram** LM (compute perplexity). Then build a **simple feed-forward** LM and compare.

---

## Week 4: Sequence Modeling (RNNs, LSTMs, GRUs)

**Core Topics**  
- Recurrent Neural Networks (RNNs)  
- Vanishing/exploding gradients  
- LSTM and GRU architectures

**Suggested References**  
- **CMU 11-711, Lecture 4**
  - Elman (1990) – “Finding Structure in Time”
  - Hochreiter & Schmidhuber (1997) – LSTM
- **UMass CS 685, Week 3**
  - Pascanu et al. (2013) – vanishing gradients in RNNs

**Practical Exercise**  
- Build an **LSTM language model**. Compare performance to a feed-forward LM on a text corpus.

---

## Week 5: Transformers & Attention Mechanisms

**Core Topics**  
- Attention (Bahdanau, Luong, Vaswani)  
- Self-attention, multi-head attention, positional encodings  
- Encoder–decoder vs. decoder-only Transformers

**Suggested References**  
- **CMU 11-711, Lecture 5**  
  - Bahdanau et al. (2015) – alignment-based attention  
  - Vaswani et al. (2017) – “Attention Is All You Need”
- **UMass CS 685, Weeks 3-4**  
  - Illustrated Transformer blog post by Jay Alammar

**Practical Exercise**  
- Implement a **Transformer encoder** for text classification. Compare speed/accuracy to an LSTM approach.

---

## Week 6: Text Generation Algorithms & Prompting Basics

**Core Topics**  
- Decoding: greedy, beam, top-k, nucleus sampling  
- Intro to prompting for generation  
- (Optional) Minimum Bayes Risk decoding

**Suggested References**  
- **CMU 11-711, Lecture 6**  
  - Holtzmann et al. (2020) – nucleus sampling  
  - Kool et al. (2019) – stochastic beam search
- **UMass CS 685, Week 7**  
  - RankGen (Krishna et al., 2022)

**Practical Exercise**  
- Implement **nucleus sampling** in a Transformer LM. Experiment with different prompts to observe changes in output.

---

## Week 7: Instruction Tuning & Efficient Fine-Tuning Methods

**Core Topics**  
- Few-shot prompting vs. full fine-tuning  
- Parameter-efficient tuning (LoRA, adapters, prompt tuning)  
- Models like T5, BERT, FLAN

**Suggested References**  
- **CMU 11-711, Lectures 7–8**  
  - Brown et al. (2020) – GPT-3 & in-context learning  
  - Wei et al. (2021, 2022) – FLAN / instruction tuning
- **UMass CS 685, Week 5**  
  - Sennrich et al. (2016) – subword units  
  - LoRA (Hu et al., 2021)  
  - Lester et al. (2021) – prompt tuning

**Practical Exercise**  
- Fine-tune a **T5** or **BERT** model with **LoRA** for a QA task. Compare with standard fine-tuning.

---

## Week 8: Experimental Design & Human Annotation

**Core Topics**  
- Designing NLP experiments  
- Human annotation best practices  
- Data collection & inter-annotator agreement

**Suggested References**  
- **CMU 11-711, Lecture 9**  
  - Bender & Friedman (2018) – data statements for NLP  
  - Lones (2021) – “How to avoid ML pitfalls”

**Practical Exercise**  
- Collect a **sentiment dataset**, label it with 2 or more annotators, and compute **Cohen’s Kappa** or **Krippendorff’s Alpha**.

---

## Week 9: Retrieval & Retrieval-Augmented Generation (RAG)

**Core Topics**  
- Information retrieval (BM25, DPR)  
- Retrieval-augmented LMs (REALM, RAG)  
- Dense vs. sparse retrieval, long context

**Suggested References**  
- **CMU 11-711, Lecture 10**  
  - Chen et al. (2017) – DrQA  
  - Karpukhin et al. (2020) – Dense Passage Retrieval  
  - Lewis et al. (2020) – RAG
- **UMass CS 685, Week 8**  
  - Guu et al. (2020) – REALM  
  - Schick et al. (2023) – Toolformer

**Practical Exercise**  
- Implement a **retrieval-augmented QA** system. Compare **BM25** vs. **DPR** for document retrieval.

---

## Week 10: Distillation, Quantization & RL from Human Feedback

**Core Topics**  
- Model compression (pruning, quantization, distillation)  
- RL from human feedback (RLHF, RLAIF, DPO)  
- Impact on performance, inference cost

**Suggested References**  
- **CMU 11-711, Lecture 11**  
  - Sanh et al. (2019) – DistilBERT  
  - Dettmers et al. (2023) – QLoRA  
  - Frankle & Carbin (2019) – Lottery Ticket Hypothesis
- **UMass CS 685, Week 6**  
  - Ouyang et al. (2022) – RLHF  
  - Lee et al. (2023) – RLAIF  
  - Rafailov et al. (2023) – Direct Preference Optimization

**Practical Exercise**  
- Distill a **larger Transformer** to a smaller one. Or set up a **mini RLHF** pipeline with a preference dataset.

---

## Week 11: Debugging & Interpretation

**Core Topics**  
- Model debugging strategies  
- Probing classifiers (edge probing)  
- Mechanistic interpretability (circuits, induction heads)  
- Model editing (ROME)

**Suggested References**  
- **CMU 11-711, Lecture 12**  
  - Tenney et al. (2019) – edge probing  
  - Elhage et al. (2021) – Transformer circuits  
  - Meng et al. (2022) – ROME
- **UMass CS 685, Week 10**  
  - Olsson et al. (2022) – induction heads  
  - Hernandez et al. (2023) – knowledge representations

**Practical Exercise**  
- Perform **edge probing** on a BERT model to analyze linguistic features.  
- Use **ROME** to edit a factual statement in GPT-style LMs.

---

## Week 12: Advanced LLMs, Agents & Long Contexts

**Core Topics**  
- Modern LLMs (LLaMa, GPT-4, Claude, Mistral)  
- Long-context solutions (Transformer-XL, RoPE, FlashAttention)  
- Language agents & tool use (Toolformer, ReAct)

**Suggested References**  
- **CMU 11-711, Lecture 15–16**  
  - Touvron et al. (2023) – LLaMa  
  - Yao et al. (2023) – ReAct  
  - Schick et al. (2023) – Toolformer
- **UMass CS 685**  
  - Su et al. (2021) – RoPE  
  - Dao et al. (2022) – FlashAttention

**Practical Exercise**  
- Experiment with a **decoder-only LLM** (e.g., LLaMa) on a larger context.  
- Integrate “tool use” (e.g., a calculator or database lookup).

---

## Week 13: Complex Reasoning & Linguistics

**Core Topics**  
- Chain-of-thought prompting  
- Abductive reasoning, logic-based inference  
- Linguistic structure in neural models  
- Compositional generalization (COGS, SCAN)

**Suggested References**  
- **CMU 11-711, Lecture 21–22**  
  - Wei et al. (2022) – chain-of-thought  
  - Kojima et al. (2022) – “Let’s Think Step by Step”  
  - Harris (1954) – distributional structure  
  - Kim & Linzen (2020) – COGS
- **UMass CS 685**  
  - Various alignment & reasoning references

**Practical Exercise**  
- Try **chain-of-thought** prompts on multi-step reasoning tasks.  
- Evaluate compositional generalization on a synthetic dataset (COGS).

---

## Week 14: Multilingual NLP & Wrap-Up

**Core Topics**  
- Multilingual embeddings (mBERT, XLM-R)  
- Zero-shot/few-shot cross-lingual transfer  
- Summarizing your entire NLP pipeline

**Suggested References**  
- **CMU 11-711, Lecture 23**  
  - Johnson et al. (2016) – Google’s multilingual NMT  
  - Wu & Dredze (2019) – “Beto, Bentz, Becas”  
  - NLLB Team (2022) – “No Language Left Behind”
- **UMass CS 685**  
  - Apply earlier methods in a multilingual setting

**Practical Exercise**  
- Fine-tune an **mBERT** model on a classification task in one language, test zero-shot in another.

---

## Additional Tips & Final Notes

1. **Time Commitment**  
   - Each “week” can be **1–2 weeks** of part-time study. Expect **3–5 months** for thorough coverage.

2. **Project Building**  
   - Combine modules into a **final project**, e.g., a retrieval-augmented, instruction-tuned mini-LLM tested on compositional tasks.

3. **Tooling**  
   - Frameworks: **PyTorch** or **TensorFlow**  
   - Retrieval libs: **FAISS**, **Chroma**, **Lucene**  
   - Model hub: **Hugging Face Transformers**

4. **Community & Discussion**  
   - Check out **NLP Slack/Discord** groups, **Reddit r/MachineLearning**.  
   - Present mini-projects for feedback.

5. **Math Foundations**  
   - Reinforce **backprop, linear algebra, probability** as needed, especially for RNNs and Transformers.

6. **Flexibility**  
   - Reorder or skip modules to suit your interests (e.g., focus on generation, interpretation, or multilingual).

---

**Good luck with your NLP self-study!** By studying each week’s topics, reading key papers, watching relevant lectures, and coding weekly exercises, you’ll build a robust understanding of modern NLP and large language models.
```
