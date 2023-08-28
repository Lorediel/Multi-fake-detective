# Multimodal deep learning model for fake news recognition

This is the code for the multimodal model that I developed for my master's thesis, tackling the problem of fake news detection. It obtained the first place in the [MULTI-Fake-DetectiVE](https://sites.google.com/unipi.it/multi-fake-detective/home) competition proposed in the workshop [EVALITA](https://www.evalita.it/) 2023. <br>

The research paper regarding this work will be available soon!

## The Problem

The task proposed asks to classify a piece of information composed of **a text** and **one or more images** in four classes:
<ul>
  <li>Certainly Fake</li>
  <li>Probably Fake</li>
  <li>Probably Real</li>
  <li>Certainly Real</li>
</ul>

## The proposed solution

First of all, the texts in the dataset were of arbitrary length. Since the NLP models like BERT present a limit to how long a text can be, a way to process longer texts was found in literature and implemented. The text was transformed in tokens, then divided in pieces that could be processed by BERT and the [CLS] tokens of the result of the single pieces were averaged to find a final representation for the text. <br>

Several tests were conducted in order to find the best model. In the end what worked out best was to re-implement as base an existing model called [FND-CLIP](https://arxiv.org/abs/2205.14304), enhanced with some original extensions. The base model includes 3 flows of information:

<ol>
  <li>
    The visual information one, achieved by the concatenation of the embeddings obtained by ResNet and the visual encoder of CLIP
  </li>
  <li>
    The textual information one, achieved by the cooncation of obtained by BERT and the textual encoder of CLIP
  </li>
  <li>
    The mixed one, achieved by the cooncation of obtained by the visual encoder of CLIP and the textual encoder of CLIP
  </li>
</ol>

The embeddings are then processed by some FC layers, weighted by [Squeeze-and-excitation](https://arxiv.org/abs/1709.01507) layers and then summed to be finally classified by a classifier.

The extensions applied are:

<ul>
  <li>
    A. Sentiment Analysis: Usage of the embeddings extracted with BERT pretrained for sentiment analysis. It includes two sub-cases: <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A1. The sentiment embeddings are concatenated with the textual flow <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A2. The sentiment embeddings constitute a separate flow
  </li>
  <li>
    B. Images frequency domain analysis: Images generated from other AI often produce artifacts that can be easily spotted in the frequency domain. So, this extension uses the Discrete Fourier Transform on the images to go from the spatial domain to the frequency domain. The real and imaginary parts obtained are both processed by a VGG19 net and then concatenated to form another flow.
  </li>
  <li>
    C. Concatenation of the embeddings: Instead of summing the flow embeddings at the end, I decided to concatenate them in order to give the network more possibility to gather the relationship between the different modalities.
  </li>
  <li>
    D. Back translation: Since the dataset presents classes that are not balanced, the underrepresented classes were augmented with the process of back-translation, that consists in the translation in a language and back in the original language, in order to maintain the semantics while changing the words. 
  </li>
  <li>
    E. Squeeze and excitation layers: Insted of using the Squeeze and excitation layers only at the end of the network, I tested their usage at the start, on the embeddings that will form the flow, but before the concatenation. This will help the network to learn how to weight the embeddings of the same modality to reduce overfitting.
  <li>
    F. Focus on Region of Interest: Using [Detectron2](https://github.com/facebookresearch/detectron2) I extracted the embeddings of the region of interest from the images, averaged them and concatened the result with the visual flow.
  </li>
</ul>

# Results

The metrics used are Accuracy, Precision, Recall, F1-score, F1-weighted. The main metric used is the F1-weighted since the dataset was not balanced. The extensions were used by themselves and combined. The best performance between all the tested combinations was obtained with the combination A1, C, D. <br>
The overall best performance was however obtained by a weighed ensemble between the combinations:
<ul>
  <li>A2, B</li>
  <li>A1, E, D</li>
  <li>A1, C, D</li>
</ul>

The table of these results is presented below:

| Model             | F1-weighted |
|-------------------|:-----------:|
| A2, B             |    0.581    |
| A1, E, D          |    0.596    |
| A1, C, D          |    0.606    |
| weighted ensemble |    0.653    |

The weighted ensemble model won the first place in the competition. The table is taken from the [official website](https://sites.google.com/unipi.it/multi-fake-detective/competition-results?authuser=0) and here reported:

| Rank |         TEAM-RUN        | Weighted Avg. F1-Score |
|:----:|:-----------------------:|:----------------------:|
|   1  |        Polito-P1        |          0.512         |
|   2  | extremITA-camoscio_lora |          0.507         |
|   3  |    AIMH-MYPRIMARYRUN    |          0.488         |
|   4  |    Baseline-SVM_TEXT    |          0.479         |
|   5  |    Baseline-SVM_MULTI   |          0.463         |
|   6  |    Baseline-MLP_TEXT    |          0.448         |
|   7  |    Baseline-MLP_IMAGE   |          0.402         |
|   8  |   HIJLI-JU-CLEF-Multi   |          0.393         |
|   9  |    Baseline-SVM_IMAGE   |          0.386         |
|  10  |    Baseline-MLP_MULTI   |          0.374         |

