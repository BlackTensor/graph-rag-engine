"""Author the evaluation set `eval/questions.jsonl` (M9.1).

A *frozen* benchmark of 50 easy + 50 multi-hop questions, each with a
ground-truth answer, used by the M9.2 Ragas harness to compare Vector-only RAG
vs. Graph-RAG.

Why a builder script (rather than hand-writing the JSONL)?
* It keeps the authoring source reviewable and the emitted file valid UTF-8 /
  valid JSON.
* The ground-truth answers were **read off the live data once** (well-cited
  papers' abstracts for the easy split; the M6.1 graph traversals for the
  multi-hop split) and are then *frozen here* — a benchmark's ground truth must
  not drift when the graph is rebuilt, so the data is hardcoded, not re-queried.

Question schema (one JSON object per line):

    id              "easy-001" / "multihop-001"
    split           "easy" | "multi_hop"
    question        the natural-language question
    ground_truth    reference answer (prose) — Ragas `reference`
    answer_entities ranked list of the key answer items (exact strings) so M9.3
                    can score recall deterministically, independent of LLM phrasing
    expected_route  the route that can actually answer it: "vector" | "graph"
    graph_template  M6.1 template that yields the answer (multi-hop) | null
    seed            {"type","name"} the traversal seed (multi-hop) | null
    source          provenance: paper title (easy) | "graph" (multi-hop)

The multi-hop questions are phrased to hit the M7 router's intent cues so they
resolve to the intended template (see src/graph/retrieve.py). Easy questions are
content-descriptive (answerable from a single paper's title+abstract), which is
exactly what the vector baseline should handle.

    python eval/build_questions.py            # writes eval/questions.jsonl
    python eval/build_questions.py --check     # validate without writing
"""

from __future__ import annotations

import json
import os
import sys

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.jsonl")


# --- easy split: definitional / single-fact, answerable from one abstract -----
# (question, ground_truth, [answer_entities], source paper title)
_EASY: list[tuple[str, str, list[str], str]] = [
    (
        "Which neural network architecture is based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions?",
        "The Transformer.",
        ["Transformer"],
        "Attention Is All You Need",
    ),
    (
        "What BLEU score did the Transformer achieve on the WMT 2014 "
        "English-to-German translation task?",
        "28.4 BLEU.",
        ["28.4"],
        "Attention Is All You Need",
    ),
    (
        "What is Kaldi?",
        "A free, open-source toolkit for speech recognition research.",
        ["speech recognition", "toolkit", "open-source"],
        "Kaldi Speech Recognition Toolkit",
    ),
    (
        "In which programming language is the core of the Kaldi speech "
        "recognition toolkit written?",
        "C++.",
        ["C++"],
        "Kaldi Speech Recognition Toolkit",
    ),
    (
        "What is the SimCLR framework used for?",
        "A simple framework for contrastive learning of visual representations.",
        ["contrastive learning", "visual representations"],
        "A simple framework for contrastive learning of visual representations",
    ),
    (
        "According to the SimCLR paper, which component plays a critical role in "
        "defining effective contrastive prediction tasks?",
        "The composition of data augmentations.",
        ["data augmentation"],
        "A simple framework for contrastive learning of visual representations",
    ),
    (
        "What does the paper 'Lost in the Middle' find about how language models "
        "use long input contexts?",
        "Performance degrades significantly when the relevant information is in "
        "the middle of a long context rather than at the beginning or end.",
        ["middle", "long context", "degrade"],
        "Lost in the Middle: How Language Models Use Long Contexts",
    ),
    (
        "Which method detects hallucinations in large language models using "
        "uncertainty estimation over meanings?",
        "Semantic entropy.",
        ["semantic entropy"],
        "Detecting hallucinations in large language models using semantic entropy",
    ),
    (
        "Which medical large language model is presented as the successor to "
        "Med-PaLM for medical question answering?",
        "Med-PaLM 2.",
        ["Med-PaLM 2"],
        "Toward expert-level medical question answering with large language models",
    ),
    (
        "According to 'Benchmarking Large Language Models for News "
        "Summarization', what is the key to a model's zero-shot summarization "
        "ability?",
        "Instruction tuning, not model size.",
        ["instruction tuning"],
        "Benchmarking Large Language Models for News Summarization",
    ),
    (
        "What is Retrieval-Augmented Generation (RAG) used for with large "
        "language models?",
        "Supplying reliable, up-to-date external knowledge to reduce "
        "hallucination and improve generated outputs.",
        ["external knowledge", "retrieval"],
        "A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models",
    ),
    (
        "What does the Chain-of-Verification (CoVe) method reduce in large "
        "language models?",
        "Hallucination.",
        ["hallucination"],
        "Chain-of-Verification Reduces Hallucination in Large Language Models",
    ),
    (
        "How many parameters is the vision foundation model in InternVL scaled "
        "up to?",
        "6 billion parameters.",
        ["6 billion"],
        "Intern VL: Scaling up Vision Foundation Models and Aligning for Generic "
        "Visual-Linguistic Tasks",
    ),
    (
        "What is DailyDialog?",
        "A manually labelled multi-turn dialogue dataset, annotated with "
        "communication intention and emotion.",
        ["dialogue dataset", "multi-turn"],
        "DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset",
    ),
    (
        "What does the acronym BERT stand for?",
        "Bidirectional Encoder Representations from Transformers.",
        ["Bidirectional Encoder Representations from Transformers"],
        "BERT applications in natural language processing: a review",
    ),
    (
        "What new segmentation task does LISA introduce?",
        "Reasoning segmentation: outputting a segmentation mask from a complex, "
        "implicit query using an LLM's reasoning.",
        ["reasoning segmentation"],
        "LISA: Reasoning Segmentation via Large Language Model",
    ),
    (
        "What is TrialGPT designed to do?",
        "Zero-shot patient-to-trial matching for clinical trials using large "
        "language models.",
        ["patient", "clinical trial", "matching"],
        "Matching patients to clinical trials with large language models",
    ),
    (
        "What three modules make up the TrialGPT framework?",
        "TrialGPT-Retrieval, TrialGPT-Matching, and TrialGPT-Ranking.",
        ["Retrieval", "Matching", "Ranking"],
        "Matching patients to clinical trials with large language models",
    ),
    (
        "What is PMC-LLaMA?",
        "An open-source large language model built for the medical domain.",
        ["medicine", "open-source"],
        "PMC-LLaMA: toward building open-source language models for medicine",
    ),
    (
        "What does AnomalyGPT detect, and using what kind of model?",
        "Industrial anomalies, using large vision-language models.",
        ["industrial anomalies", "vision-language"],
        "AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models",
    ),
    (
        "What capability does SpatialVLM add to vision-language models?",
        "Spatial reasoning, including 3D quantitative relationships such as "
        "distances and size differences.",
        ["spatial reasoning"],
        "SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities",
    ),
    (
        "What is the subject of the paper 'A Survey on Model Compression for "
        "Large Language Models'?",
        "Model compression techniques for LLMs, such as quantization and pruning.",
        ["model compression", "quantization"],
        "A Survey on Model Compression for Large Language Models",
    ),
    (
        "What kind of language model architecture is DeepSeekMoE?",
        "A Mixture-of-Experts language model aimed at expert specialization.",
        ["Mixture-of-Experts"],
        "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts "
        "Language Models",
    ),
    (
        "What does the paper 'AI generates covertly racist decisions about people "
        "based on their dialect' find?",
        "Language models exhibit covert racism (dialect prejudice) against "
        "speakers of African American English.",
        ["dialect", "racism"],
        "AI generates covertly racist decisions about people based on their dialect",
    ),
    (
        "What is NaturalSpeech?",
        "An end-to-end text-to-speech synthesis system that achieves human-level "
        "quality.",
        ["text-to-speech", "human-level"],
        "NaturalSpeech: End-to-End Text-to-Speech Synthesis With Human-Level Quality",
    ),
    (
        "What does the paper 'Larger and more instructable language models become "
        "less reliable' argue?",
        "Scaling up and instruction-shaping LLMs can make them less reliable.",
        ["less reliable"],
        "Larger and more instructable language models become less reliable",
    ),
    (
        "What task was the CoNLL-2013 shared task devoted to?",
        "Grammatical error correction.",
        ["grammatical error correction"],
        "The CoNLL-2013 Shared Task on Grammatical Error Correction",
    ),
    (
        "What process does the paper 'VILA: On Pre-training for Visual Language "
        "Models' study?",
        "The pre-training process for visual language models.",
        ["visual language models", "pre-training"],
        "VILA: On Pre-training for Visual Language Models",
    ),
    (
        "What does ConceptGraphs build to support robot perception and planning?",
        "Open-vocabulary 3D scene graphs.",
        ["3D scene graphs", "open-vocabulary"],
        "ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning",
    ),
    (
        "What is the goal of the ChatExtract method?",
        "Fully automated, accurate data extraction from research papers using "
        "conversational LLMs and prompt engineering.",
        ["data extraction", "prompt engineering"],
        "Extracting accurate materials data from research papers with "
        "conversational language models and prompt engineering",
    ),
    (
        "Which two pretrained large language models are fine-tuned for structured "
        "information extraction from scientific text?",
        "GPT-3 and Llama-2.",
        ["GPT-3", "Llama-2"],
        "Structured information extraction from scientific text with large "
        "language models",
    ),
    (
        "Why does the paper 'Unifying Large Language Models and Knowledge Graphs' "
        "argue for combining LLMs with knowledge graphs?",
        "Knowledge graphs store structured, explicit factual knowledge that "
        "black-box LLMs often fail to capture.",
        ["knowledge graphs", "factual knowledge"],
        "Unifying Large Language Models and Knowledge Graphs: A Roadmap",
    ),
    (
        "What does 'A Survey of Large Language Models' review?",
        "Recent advances in large language model techniques (pre-training, "
        "adaptation, utilization, and capability evaluation).",
        ["large language models", "survey"],
        "A Survey of Large Language Models",
    ),
    (
        "What does 'Explainability for Large Language Models: A Survey' provide?",
        "A taxonomy of explainability techniques for LLMs.",
        ["explainability", "taxonomy"],
        "Explainability for Large Language Models: A Survey",
    ),
    (
        "What is the focus of 'Pre-Trained Language Models for Text Generation: A "
        "Survey'?",
        "Using pre-trained language models for text generation.",
        ["text generation", "pre-trained language models"],
        "Pre-Trained Language Models for Text Generation: A Survey",
    ),
    (
        "What does the Text-to-SQL task empowered by large language models aim to "
        "produce?",
        "SQL queries generated from natural-language questions.",
        ["Text-to-SQL", "SQL"],
        "Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation",
    ),
    (
        "What does the paper 'Can large language models reason about medical "
        "questions?' investigate?",
        "Whether models such as GPT-3.5 and Llama 2 can answer and reason about "
        "difficult real-world medical benchmark questions.",
        ["medical questions", "reasoning"],
        "Can large language models reason about medical questions?",
    ),
    (
        "What does the paper 'Benchmarking Retrieval-Augmented Generation for "
        "Medicine' study?",
        "Best practices for retrieval-augmented generation systems in the "
        "medical domain.",
        ["retrieval-augmented generation", "medicine"],
        "Benchmarking Retrieval-Augmented Generation for Medicine",
    ),
    (
        "What does the paper 'Cultural bias and cultural alignment of large "
        "language models' evaluate?",
        "Cultural bias across several widely used large language models.",
        ["cultural bias"],
        "Cultural bias and cultural alignment of large language models",
    ),
    (
        "What is UCCA?",
        "Universal Conceptual Cognitive Annotation, a multi-layered framework "
        "for semantic representation.",
        ["semantic representation", "Universal Conceptual Cognitive Annotation"],
        "Universal Conceptual Cognitive Annotation (UCCA)",
    ),
    (
        "What does the paper on using GPT for multilingual psychological text "
        "analysis conclude?",
        "GPT can serve as an effective tool for automated psychological text "
        "analysis across multiple languages.",
        ["psychological text analysis", "multilingual"],
        "GPT is an effective tool for multilingual psychological text analysis",
    ),
    (
        "What problem does the paper 'Reading digits in natural images with "
        "unsupervised feature learning' address?",
        "Recognizing digits and characters in natural scene images (the Street "
        "View House Numbers problem) using unsupervised feature learning.",
        ["digits", "natural images"],
        "Reading digits in natural images with unsupervised feature learning",
    ),
    (
        "What does the bilingual language model for protein sequence and "
        "structure capture about proteins?",
        "Both the 1D amino-acid sequence and the 3D structure of proteins.",
        ["protein", "sequence", "structure"],
        "Bilingual language model for protein sequence and structure",
    ),
    (
        "What does 'Sentiment Analysis in the Era of Large Language Models: A "
        "Reality Check' assess?",
        "How well current large language models perform across various sentiment "
        "analysis tasks.",
        ["sentiment analysis", "large language models"],
        "Sentiment Analysis in the Era of Large Language Models: A Reality Check",
    ),
    (
        "How is sentiment analysis defined in the NLP literature?",
        "An NLP method that identifies the emotional tone or polarity of text as "
        "positive, negative, or neutral.",
        ["polarity", "positive", "negative", "neutral"],
        "Recent advancements and challenges of NLP-based sentiment analysis: A "
        "state-of-the-art review",
    ),
    (
        "What is the topic of 'Large language models for generative information "
        "extraction: a survey'?",
        "Using generative large language models for information extraction tasks.",
        ["information extraction", "generative"],
        "Large language models for generative information extraction: a survey",
    ),
    (
        "What does the paper 'Fine-tuning ChatGPT for automatic scoring' apply "
        "ChatGPT to?",
        "Automatically scoring student written constructed responses in science "
        "education.",
        ["automatic scoring", "student responses"],
        "Fine-tuning ChatGPT for automatic scoring",
    ),
    (
        "What does the survey 'Foundation Models Defining a New Era in Vision' "
        "cover?",
        "Vision foundation models.",
        ["foundation models", "vision"],
        "Foundation Models Defining a New Era in Vision: A Survey and Outlook",
    ),
    (
        "What detection signal does 'A baseline for detecting misclassified and "
        "out-of-distribution examples in neural networks' use?",
        "The maximum softmax probability from the softmax distribution.",
        ["softmax"],
        "A baseline for detecting misclassified and out-of-distribution examples "
        "in neural networks",
    ),
    (
        "According to 'Benchmarking Large Language Models in Retrieval-Augmented "
        "Generation', what is RAG a promising approach for?",
        "Mitigating the hallucination of large language models.",
        ["hallucination"],
        "Benchmarking Large Language Models in Retrieval-Augmented Generation",
    ),
]


# --- multi-hop split: relational, requires graph traversal --------------------
# (question, template, seed_type, seed_name, ground_truth, [answer_entities])
_MULTI: list[tuple[str, str, str, str, str, list[str]]] = [
    # --- collaborating_institutions ------------------------------------------
    (
        "Which institutions collaborate most with Tsinghua University?",
        "collaborating_institutions", "institution", "Tsinghua University",
        "The University of Hong Kong (14 shared papers), the Chinese University "
        "of Hong Kong (13), Shanghai Artificial Intelligence Laboratory (12), "
        "Hong Kong University of Science and Technology (11), and Beijing Academy "
        "of Artificial Intelligence (10).",
        ["University of Hong Kong", "Chinese University of Hong Kong",
         "Shanghai Artificial Intelligence Laboratory",
         "Hong Kong University of Science and Technology",
         "Beijing Academy of Artificial Intelligence"],
    ),
    (
        "Which institutions co-author the most papers with Stanford University?",
        "collaborating_institutions", "institution", "Stanford University",
        "Stanford Medicine (30 shared papers), Palo Alto University (10), Chan "
        "Zuckerberg Initiative (9), New York University (8), and the University "
        "of California, San Francisco (8).",
        ["Stanford Medicine", "Palo Alto University", "Chan Zuckerberg Initiative",
         "New York University", "University of California, San Francisco"],
    ),
    (
        "Which institutions collaborate most with Shanghai Artificial "
        "Intelligence Laboratory?",
        "collaborating_institutions", "institution",
        "Shanghai Artificial Intelligence Laboratory",
        "Beijing Academy of Artificial Intelligence (31 shared papers), Shanghai "
        "Jiao Tong University (19), the University of Hong Kong (16), the Chinese "
        "University of Hong Kong (15), and Shanghai Open University (13).",
        ["Beijing Academy of Artificial Intelligence",
         "Shanghai Jiao Tong University", "University of Hong Kong",
         "Chinese University of Hong Kong", "Shanghai Open University"],
    ),
    (
        "Which institutions collaborate most with Harvard University?",
        "collaborating_institutions", "institution", "Harvard University",
        "Massachusetts Institute of Technology (16 shared papers), Massachusetts "
        "General Hospital (15), Brigham and Women's Hospital (13), Boston "
        "Children's Hospital (10), and Beth Israel Deaconess Medical Center (9).",
        ["Massachusetts Institute of Technology", "Massachusetts General Hospital",
         "Brigham and Women's Hospital", "Boston Children's Hospital",
         "Beth Israel Deaconess Medical Center"],
    ),
    (
        "Which institutions collaborate most with Peking University?",
        "collaborating_institutions", "institution", "Peking University",
        "Tsinghua University (10 shared papers), Zhejiang University (8), Beijing "
        "Academy of Artificial Intelligence (7), Peng Cheng Laboratory (7), and "
        "Shanghai Jiao Tong University (7).",
        ["Tsinghua University", "Zhejiang University",
         "Beijing Academy of Artificial Intelligence", "Peng Cheng Laboratory",
         "Shanghai Jiao Tong University"],
    ),
    (
        "Which institutions collaborate most with Microsoft?",
        "collaborating_institutions", "institution", "Microsoft",
        "Microsoft Research Asia (13 shared papers), Seattle University (5), "
        "Amazon (4), Harbin Institute of Technology (4), and Microsoft Research "
        "Montréal (4).",
        ["Microsoft Research Asia", "Seattle University", "Amazon",
         "Harbin Institute of Technology", "Microsoft Research Montréal"],
    ),
    (
        "Which institutions collaborate most with Nanyang Technological "
        "University?",
        "collaborating_institutions", "institution",
        "Nanyang Technological University",
        "The University of Sydney (10 shared papers), the National University of "
        "Singapore (9), the University of Hong Kong (9), Zhejiang University (9), "
        "and the Singapore University of Technology and Design (8).",
        ["The University of Sydney", "National University of Singapore",
         "University of Hong Kong", "Zhejiang University",
         "Singapore University of Technology and Design"],
    ),
    (
        "Which institutions collaborate most with Carnegie Mellon University?",
        "collaborating_institutions", "institution", "Carnegie Mellon University",
        "Carnegie Mellon University Africa (11 shared papers), Honda (11), "
        "Shanghai Jiao Tong University (5), Tsinghua University (5), and Johns "
        "Hopkins University (4).",
        ["Carnegie Mellon University Africa", "Honda",
         "Shanghai Jiao Tong University", "Tsinghua University",
         "Johns Hopkins University"],
    ),
    # --- institution_topics ---------------------------------------------------
    (
        "What topics does Tsinghua University research the most?",
        "institution_topics", "institution", "Tsinghua University",
        "Topic Modeling (65 papers), Natural Language Processing Techniques (44), "
        "Multimodal Machine Learning Applications (28), Domain Adaptation and "
        "Few-Shot Learning (12), and Advanced Graph Neural Networks (11).",
        ["Topic Modeling", "Natural Language Processing Techniques",
         "Multimodal Machine Learning Applications",
         "Domain Adaptation and Few-Shot Learning",
         "Advanced Graph Neural Networks"],
    ),
    (
        "What topics does Stanford University research the most?",
        "institution_topics", "institution", "Stanford University",
        "Topic Modeling (45 papers), Artificial Intelligence in Healthcare and "
        "Education (23), Natural Language Processing Techniques (23), Machine "
        "Learning in Healthcare (19), and Multimodal Machine Learning "
        "Applications (13).",
        ["Topic Modeling", "Artificial Intelligence in Healthcare and Education",
         "Natural Language Processing Techniques", "Machine Learning in Healthcare",
         "Multimodal Machine Learning Applications"],
    ),
    (
        "What research topics does Harvard University focus on the most?",
        "institution_topics", "institution", "Harvard University",
        "Artificial Intelligence in Healthcare and Education (28 papers), Topic "
        "Modeling (18), Machine Learning in Healthcare (15), Neurobiology of "
        "Language and Bilingualism (12), and Biomedical Text Mining and "
        "Ontologies (9).",
        ["Artificial Intelligence in Healthcare and Education", "Topic Modeling",
         "Machine Learning in Healthcare",
         "Neurobiology of Language and Bilingualism",
         "Biomedical Text Mining and Ontologies"],
    ),
    (
        "What topics does Carnegie Mellon University research the most?",
        "institution_topics", "institution", "Carnegie Mellon University",
        "Natural Language Processing Techniques (21 papers), Speech Recognition "
        "and Synthesis (18), Topic Modeling (18), Speech and Audio Processing "
        "(8), and Multimodal Machine Learning Applications (7).",
        ["Natural Language Processing Techniques", "Speech Recognition and Synthesis",
         "Topic Modeling", "Speech and Audio Processing",
         "Multimodal Machine Learning Applications"],
    ),
    (
        "What topics does the Chinese Academy of Sciences research the most?",
        "institution_topics", "institution", "Chinese Academy of Sciences",
        "Topic Modeling (35 papers), Multimodal Machine Learning Applications "
        "(27), Natural Language Processing Techniques (22), Advanced Image and "
        "Video Retrieval Techniques (17), and Sentiment Analysis and Opinion "
        "Mining (15).",
        ["Topic Modeling", "Multimodal Machine Learning Applications",
         "Natural Language Processing Techniques",
         "Advanced Image and Video Retrieval Techniques",
         "Sentiment Analysis and Opinion Mining"],
    ),
    (
        "What research topics does Microsoft work on the most?",
        "institution_topics", "institution", "Microsoft",
        "Topic Modeling (28 papers), Natural Language Processing Techniques (24), "
        "Speech Recognition and Synthesis (13), Speech and dialogue systems (6), "
        "and Speech and Audio Processing (4).",
        ["Topic Modeling", "Natural Language Processing Techniques",
         "Speech Recognition and Synthesis", "Speech and dialogue systems",
         "Speech and Audio Processing"],
    ),
    # --- institution_authors ---------------------------------------------------
    (
        "Who are the most published authors at Tsinghua University?",
        "institution_authors", "institution", "Tsinghua University",
        "Yuxiao Dong (6 papers), Zhiyuan Liu (6), Jie Tang (5), Maosong Sun (5), "
        "and Yinghui Li (5).",
        ["Yuxiao Dong", "Zhiyuan Liu", "Jie Tang", "Maosong Sun", "Yinghui Li"],
    ),
    (
        "Who are the most published authors at Stanford University?",
        "institution_authors", "institution", "Stanford University",
        "James Zou (11 papers), Akshay Chaudhari (6), Curtis P. Langlotz (5), "
        "Jean-Benoit Delbrouck (5), and Laura Gwilliams (5).",
        ["James Zou", "Akshay Chaudhari", "Curtis P. Langlotz",
         "Jean-Benoit Delbrouck", "Laura Gwilliams"],
    ),
    (
        "Who are the most published authors at Nanyang Technological University?",
        "institution_authors", "institution", "Nanyang Technological University",
        "Erik Cambria (21 papers), Dacheng Tao (8), Hongyang Du (8), Rui Mao "
        "(8), and Yang Liu (8).",
        ["Erik Cambria", "Dacheng Tao", "Hongyang Du", "Rui Mao", "Yang Liu"],
    ),
    (
        "Who are the most published authors at Carnegie Mellon University?",
        "institution_authors", "institution", "Carnegie Mellon University",
        "Shinji Watanabe (12 papers), Xuankai Chang (7), Carlos Busso (5), "
        "Jee-weon Jung (5), and Deva Ramanan (4).",
        ["Shinji Watanabe", "Xuankai Chang", "Carlos Busso", "Jee-weon Jung",
         "Deva Ramanan"],
    ),
    (
        "Who are the most published authors at Shanghai Artificial Intelligence "
        "Laboratory?",
        "institution_authors", "institution",
        "Shanghai Artificial Intelligence Laboratory",
        "Yu Qiao (15 papers), Kai Chen (8), Dahua Lin (7), Ping Luo (7), and "
        "Weidi Xie (7).",
        ["Yu Qiao", "Kai Chen", "Dahua Lin", "Ping Luo", "Weidi Xie"],
    ),
    # --- topic_institutions ----------------------------------------------------
    (
        "Which institutions work most on sentiment analysis and opinion mining?",
        "topic_institutions", "topic", "Sentiment Analysis and Opinion Mining",
        "Wuhan University (16 papers), the Chinese Academy of Sciences (15), "
        "Harbin Institute of Technology (14), King Saud University (10), and "
        "Nanyang Technological University (10).",
        ["Wuhan University", "Chinese Academy of Sciences",
         "Harbin Institute of Technology", "King Saud University",
         "Nanyang Technological University"],
    ),
    (
        "Which institutions are most active in multimodal machine applications?",
        "topic_institutions", "topic", "Multimodal Machine Learning Applications",
        "Beijing Academy of Artificial Intelligence (29 papers), Tsinghua "
        "University (28), the Chinese Academy of Sciences (27), the University of "
        "Hong Kong (27), and Shanghai Artificial Intelligence Laboratory (26).",
        ["Beijing Academy of Artificial Intelligence", "Tsinghua University",
         "Chinese Academy of Sciences", "University of Hong Kong",
         "Shanghai Artificial Intelligence Laboratory"],
    ),
    (
        "Which institutions are most active in speech recognition and synthesis?",
        "topic_institutions", "topic", "Speech Recognition and Synthesis",
        "Carnegie Mellon University (18 papers), Microsoft (13), Shanghai Jiao "
        "Tong University (13), Carnegie Mellon University Africa (11), and Honda "
        "(11).",
        ["Carnegie Mellon University", "Microsoft", "Shanghai Jiao Tong University",
         "Carnegie Mellon University Africa", "Honda"],
    ),
    (
        "Which institutions work most on advanced graph neural networks?",
        "topic_institutions", "topic", "Advanced Graph Neural Networks",
        "Wuhan University (12 papers), Tsinghua University (11), Huawei "
        "Technologies (10), the University of Hong Kong (9), and the University "
        "of Science and Technology of China (9).",
        ["Wuhan University", "Tsinghua University", "Huawei Technologies",
         "University of Hong Kong", "University of Science and Technology of China"],
    ),
    (
        "Which institutions are most active in machine-based healthcare research?",
        "topic_institutions", "topic", "Machine Learning in Healthcare",
        "Stanford University (19 papers), Yale University (17), Harvard "
        "University (15), the University of Texas Health Science Center at "
        "Houston (15), and Vanderbilt University (12).",
        ["Stanford University", "Yale University", "Harvard University",
         "The University of Texas Health Science Center at Houston",
         "Vanderbilt University"],
    ),
    (
        "Which institutions work most on biomedical text mining?",
        "topic_institutions", "topic", "Biomedical Text Mining and Ontologies",
        "Yale University (12 papers), Stanford University (10), Harvard "
        "University (9), the University of Texas Health Science Center at Houston "
        "(9), and the National Center for Biotechnology Information (8).",
        ["Yale University", "Stanford University", "Harvard University",
         "The University of Texas Health Science Center at Houston",
         "National Center for Biotechnology Information"],
    ),
    # --- topic_authors ---------------------------------------------------------
    (
        "Who are the leading authors on sentiment analysis and opinion mining?",
        "topic_authors", "topic", "Sentiment Analysis and Opinion Mining",
        "Erik Cambria (7 papers), Changqin Huang (5), Rui Wang (5), Chenxi Shi "
        "(4), and Geng Tu (4).",
        ["Erik Cambria", "Changqin Huang", "Rui Wang", "Chenxi Shi", "Geng Tu"],
    ),
    (
        "Who are the leading authors on speech recognition and synthesis?",
        "topic_authors", "topic", "Speech Recognition and Synthesis",
        "Shinji Watanabe (11 papers), Xuankai Chang (7), Lei Xie (6), Shujie Liu "
        "(6), and Jinyu Li (5).",
        ["Shinji Watanabe", "Xuankai Chang", "Lei Xie", "Shujie Liu", "Jinyu Li"],
    ),
    (
        "Who are the leading authors on biomedical text mining?",
        "topic_authors", "topic", "Biomedical Text Mining and Ontologies",
        "Hua Xu (8 papers), Qingyu Chen (6), Yan Hu (6), Zhiyong Lu (5), and "
        "James Zou (4).",
        ["Hua Xu", "Qingyu Chen", "Yan Hu", "Zhiyong Lu", "James Zou"],
    ),
    (
        "Who are the leading authors on topic modeling?",
        "topic_authors", "topic", "Topic Modeling",
        "Yang Liu (13 papers), Hua Xu (11), Ji-Rong Wen (10), Lei Wang (10), and "
        "Min Zhang (10).",
        ["Yang Liu", "Hua Xu", "Ji-Rong Wen", "Lei Wang", "Min Zhang"],
    ),
    (
        "Who are the leading authors on multimodal machine applications?",
        "topic_authors", "topic", "Multimodal Machine Learning Applications",
        "Yu Qiao (8 papers), Ping Luo (7), Yang Liu (6), Hongxu Yin (5), and "
        "Jian Yang (5).",
        ["Yu Qiao", "Ping Luo", "Yang Liu", "Hongxu Yin", "Jian Yang"],
    ),
    # --- topic_top_cited_papers ------------------------------------------------
    (
        "What are the most cited papers on sentiment analysis and opinion mining?",
        "topic_top_cited_papers", "topic", "Sentiment Analysis and Opinion Mining",
        "Within the corpus: 'Sentiment Analysis in the Age of Generative AI' (6 "
        "in-corpus citations), 'Analyzing Sentiments in eLearning: A Comparative "
        "Study of Bangla and Romanized Bangla Text Using Transformers' (5), "
        "'Frontiers: Determining the Validity of Large Language Models for "
        "Automated Perceptual Analysis' (5), 'Predictive Analytics in Mental "
        "Health Leveraging LLM Embeddings and Machine Learning Models for Social "
        "Media Analysis' (5), and 'A multimodal approach to cross-lingual "
        "sentiment analysis with ensemble of transformer and LLM' (4).",
        ["Sentiment Analysis in the Age of Generative AI",
         "Analyzing Sentiments in eLearning: A Comparative Study of Bangla and "
         "Romanized Bangla Text Using Transformers",
         "Frontiers: Determining the Validity of Large Language Models for "
         "Automated Perceptual Analysis",
         "Predictive Analytics in Mental Health Leveraging LLM Embeddings and "
         "Machine Learning Models for Social Media Analysis",
         "A multimodal approach to cross-lingual sentiment analysis with "
         "ensemble of transformer and LLM"],
    ),
    (
        "What are the most cited papers on advanced graph neural networks?",
        "topic_top_cited_papers", "topic", "Advanced Graph Neural Networks",
        "Within the corpus: 'Unifying Large Language Models and Knowledge Graphs: "
        "A Roadmap' (30 in-corpus citations), 'Representation Learning with Large "
        "Language Models for Recommendation' (11), 'Exploring the Potential of "
        "Large Language Models (LLMs) in Learning on Graphs' (10), 'ReLLa: "
        "Retrieval-enhanced Large Language Models for Lifelong Sequential "
        "Behavior Comprehension in Recommendation' (7), and 'Collaborative Large "
        "Language Model for Recommender Systems' (6).",
        ["Unifying Large Language Models and Knowledge Graphs: A Roadmap",
         "Representation Learning with Large Language Models for Recommendation",
         "Exploring the Potential of Large Language Models (LLMs)in Learning on Graphs",
         "ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential "
         "Behavior Comprehension in Recommendation",
         "Collaborative Large Language Model for Recommender Systems"],
    ),
    (
        "What are the most cited papers on machine-based healthcare?",
        "topic_top_cited_papers", "topic", "Machine Learning in Healthcare",
        "Within the corpus: 'Adapted large language models can outperform medical "
        "experts in clinical text summarization' (22 in-corpus citations), "
        "'PMC-LLaMA: toward building open-source language models for medicine' "
        "(19), 'Explainability for Large Language Models: A Survey' (18), 'Can "
        "large language models reason about medical questions?' (10), and "
        "'Advancing entity recognition in biomedicine via instruction tuning of "
        "large language models' (9).",
        ["Adapted large language models can outperform medical experts in "
         "clinical text summarization",
         "PMC-LLaMA: toward building open-source language models for medicine",
         "Explainability for Large Language Models: A Survey",
         "Can large language models reason about medical questions?",
         "Advancing entity recognition in biomedicine via instruction tuning of "
         "large language models"],
    ),
    (
        "What are the most cited papers on speech recognition and synthesis?",
        "topic_top_cited_papers", "topic", "Speech Recognition and Synthesis",
        "Within the corpus: 'WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio "
        "Captioning Dataset for Audio-Language Multimodal Research' (7 in-corpus "
        "citations), 'Prompting Large Language Models with Speech Recognition "
        "Abilities' (6), 'SpeechX: Neural Codec Language Model as a Versatile "
        "Speech Transformer' (5), 'AnyMAL: An Efficient and Scalable "
        "Any-Modality Augmented Language Model' (4), and 'NaturalSpeech: "
        "End-to-End Text-to-Speech Synthesis With Human-Level Quality' (3).",
        ["WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset "
         "for Audio-Language Multimodal Research",
         "Prompting Large Language Models with Speech Recognition Abilities",
         "SpeechX: Neural Codec Language Model as a Versatile Speech Transformer",
         "AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model",
         "NaturalSpeech: End-to-End Text-to-Speech Synthesis With Human-Level Quality"],
    ),
    (
        "What are the most cited papers on natural language processing "
        "techniques?",
        "topic_top_cited_papers", "topic", "Natural Language Processing Techniques",
        "Within the corpus: 'Affordance-Compiled Intelligence: Observable-Only "
        "Cognitive Impedance Matching for No-Meta LLM-Integrated Systems' (92 "
        "in-corpus citations), 'A Survey of Large Language Models' (76), 'Helping "
        "Cancer Patients to Choose the Best Treatment' (48), 'Unifying Large "
        "Language Models and Knowledge Graphs: A Roadmap' (30), and 'Lost in the "
        "Middle: How Language Models Use Long Contexts' (24).",
        ["Affordance-Compiled Intelligence: Observable-Only Cognitive Impedance "
         "Matching for No-Meta LLM-Integrated Systems",
         "A Survey of Large Language Models",
         "Helping Cancer Patients to Choose the Best Treatment",
         "Unifying Large Language Models and Knowledge Graphs: A Roadmap",
         "Lost in the Middle: How Language Models Use Long Contexts"],
    ),
    # --- peer_institutions_via_topics -----------------------------------------
    (
        "Which institutions work on the same topics as Ashish Vaswani?",
        "peer_institutions_via_topics", "author", "Ashish Vaswani",
        "Institutions publishing on the same topics Ashish Vaswani studies, by "
        "paper count: Tsinghua University (86 papers across 3 shared topics), "
        "Nanyang Technological University (69), Beijing Academy of Artificial "
        "Intelligence (67), the University of Science and Technology of China "
        "(65), and the University of Hong Kong (63).",
        ["Tsinghua University", "Nanyang Technological University",
         "Beijing Academy of Artificial Intelligence",
         "University of Science and Technology of China", "University of Hong Kong"],
    ),
    (
        "Which institutions research the same topics as Erik Cambria?",
        "peer_institutions_via_topics", "author", "Erik Cambria",
        "By paper count: Tsinghua University (105 papers), Nanyang Technological "
        "University (94), the Chinese Academy of Sciences (88), Beijing Academy "
        "of Artificial Intelligence (82), and Zhejiang University (80).",
        ["Tsinghua University", "Nanyang Technological University",
         "Chinese Academy of Sciences",
         "Beijing Academy of Artificial Intelligence", "Zhejiang University"],
    ),
    (
        "Which institutions work on the same topics as Christopher D. Manning?",
        "peer_institutions_via_topics", "author", "Christopher D. Manning",
        "By paper count: Tsinghua University (76 papers), Nanyang Technological "
        "University (57), Stanford University (55), Beijing Academy of Artificial "
        "Intelligence (52), and the University of Hong Kong (51).",
        ["Tsinghua University", "Nanyang Technological University",
         "Stanford University", "Beijing Academy of Artificial Intelligence",
         "University of Hong Kong"],
    ),
    (
        "Which institutions research the same topics as Shinji Watanabe?",
        "peer_institutions_via_topics", "author", "Shinji Watanabe",
        "By paper count: Tsinghua University (79 papers), Nanyang Technological "
        "University (61), Beijing Academy of Artificial Intelligence (57), "
        "Shanghai Jiao Tong University (55), and Stanford University (54).",
        ["Tsinghua University", "Nanyang Technological University",
         "Beijing Academy of Artificial Intelligence",
         "Shanghai Jiao Tong University", "Stanford University"],
    ),
    # --- coauthors -------------------------------------------------------------
    (
        "Who collaborates with Ashish Vaswani?",
        "coauthors", "author", "Ashish Vaswani",
        "His co-authors in the corpus include Aidan N. Gomez, Illia Polosukhin, "
        "Jakob Uszkoreit, Llion Jones, and Niki Parmar (the 'Attention Is All "
        "You Need' team), each with 1 shared paper.",
        ["Aidan N. Gomez", "Illia Polosukhin", "Jakob Uszkoreit", "Llion Jones",
         "Niki Parmar"],
    ),
    (
        "Who collaborates with Erik Cambria?",
        "coauthors", "author", "Erik Cambria",
        "Rui Mao (7 shared papers), Soujanya Poria (3), Xianxun Zhu (3), Chaopeng "
        "Guo (2), and Frank Xing (2).",
        ["Rui Mao", "Soujanya Poria", "Xianxun Zhu", "Chaopeng Guo", "Frank Xing"],
    ),
    (
        "Who collaborates with Shinji Watanabe?",
        "coauthors", "author", "Shinji Watanabe",
        "Xuankai Chang (7 shared papers), Yifan Peng (5), Jee-weon Jung (4), "
        "Jiatong Shi (4), and Jinchuan Tian (4).",
        ["Xuankai Chang", "Yifan Peng", "Jee-weon Jung", "Jiatong Shi",
         "Jinchuan Tian"],
    ),
    (
        "Who collaborates with Rui Mao?",
        "coauthors", "author", "Rui Mao",
        "Erik Cambria (7 shared papers), Frank Xing (2), Kelvin Du (2), Amber "
        "Hogarth (1), and Bernard J. Jansen (1).",
        ["Erik Cambria", "Frank Xing", "Kelvin Du", "Amber Hogarth",
         "Bernard J. Jansen"],
    ),
    # --- author_institutions ---------------------------------------------------
    (
        "Where does Ashish Vaswani work?",
        "author_institutions", "author", "Ashish Vaswani",
        "Google.",
        ["Google"],
    ),
    (
        "Which institution is Christopher D. Manning affiliated with?",
        "author_institutions", "author", "Christopher D. Manning",
        "Stanford University.",
        ["Stanford University"],
    ),
    (
        "Which institutions is James Zou affiliated with?",
        "author_institutions", "author", "James Zou",
        "Chan Zuckerberg Initiative, Stanford Medicine, and Stanford University.",
        ["Chan Zuckerberg Initiative", "Stanford Medicine", "Stanford University"],
    ),
    # --- author_topics ---------------------------------------------------------
    (
        "Which research topics does Erik Cambria study the most?",
        "author_topics", "author", "Erik Cambria",
        "Sentiment Analysis and Opinion Mining (7 papers), Topic Modeling (7), "
        "Natural Language Processing Techniques (5), Advanced Text Analysis "
        "Techniques (4), and Emotion and Mood Recognition (4).",
        ["Sentiment Analysis and Opinion Mining", "Topic Modeling",
         "Natural Language Processing Techniques",
         "Advanced Text Analysis Techniques", "Emotion and Mood Recognition"],
    ),
    (
        "Which topics does Yu Qiao study the most?",
        "author_topics", "author", "Yu Qiao",
        "Multimodal Machine Learning Applications (8 papers), Natural Language "
        "Processing Techniques (8), Topic Modeling (6), Advanced Image and Video "
        "Retrieval Techniques (3), and Human Pose and Action Recognition (2).",
        ["Multimodal Machine Learning Applications",
         "Natural Language Processing Techniques", "Topic Modeling",
         "Advanced Image and Video Retrieval Techniques",
         "Human Pose and Action Recognition"],
    ),
    (
        "Which research topics does Furu Wei study the most?",
        "author_topics", "author", "Furu Wei",
        "Natural Language Processing Techniques (11 papers), Topic Modeling (8), "
        "Speech Recognition and Synthesis (4), Speech and dialogue systems (2), "
        "and Algorithms and Data Compression (1).",
        ["Natural Language Processing Techniques", "Topic Modeling",
         "Speech Recognition and Synthesis", "Speech and dialogue systems",
         "Algorithms and Data Compression"],
    ),
    (
        "Which topics does Christopher D. Manning study?",
        "author_topics", "author", "Christopher D. Manning",
        "Biomedical Text Mining and Ontologies (2 papers), Topic Modeling (2), "
        "Artificial Intelligence in Law (1), Ethics and Social Impacts of AI (1), "
        "and Law, AI, and Intellectual Property (1).",
        ["Biomedical Text Mining and Ontologies", "Topic Modeling",
         "Artificial Intelligence in Law", "Ethics and Social Impacts of AI",
         "Law, AI, and Intellectual Property"],
    ),
]


def build() -> list[dict]:
    records: list[dict] = []
    for i, (q, gt, ents, src) in enumerate(_EASY, 1):
        records.append(
            {
                "id": f"easy-{i:03d}",
                "split": "easy",
                "question": q,
                "ground_truth": gt,
                "answer_entities": ents,
                "expected_route": "vector",
                "graph_template": None,
                "seed": None,
                "source": src,
            }
        )
    for i, (q, tmpl, stype, sname, gt, ents) in enumerate(_MULTI, 1):
        records.append(
            {
                "id": f"multihop-{i:03d}",
                "split": "multi_hop",
                "question": q,
                "ground_truth": gt,
                "answer_entities": ents,
                "expected_route": "graph",
                "graph_template": tmpl,
                "seed": {"type": stype, "name": sname},
                "source": "graph",
            }
        )
    return records


def _check(records: list[dict]) -> None:
    """Self-validate the set before writing (fail loud on authoring mistakes)."""
    easy = [r for r in records if r["split"] == "easy"]
    multi = [r for r in records if r["split"] == "multi_hop"]
    assert len(easy) == 50, f"expected 50 easy, got {len(easy)}"
    assert len(multi) == 50, f"expected 50 multi_hop, got {len(multi)}"
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids), "duplicate ids"
    questions = [r["question"] for r in records]
    assert len(set(questions)) == len(questions), "duplicate questions"
    from graph import templates as T  # noqa: PLC0415

    for r in multi:
        assert r["graph_template"] in T.TEMPLATES, (
            f"{r['id']}: unknown template {r['graph_template']!r}"
        )
        assert r["answer_entities"], f"{r['id']}: empty answer_entities"
    for r in easy:
        assert r["answer_entities"], f"{r['id']}: empty answer_entities"


def main() -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    records = build()
    _check(records)
    if "--check" in sys.argv[1:]:
        print(f"OK: {len(records)} questions (50 easy + 50 multi_hop) validate.")
        return 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} questions to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
