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

Several tests were conducted in order to find the best model. In the end what worked out best was to use as base an existing model called [FND-CLIP](https://arxiv.org/abs/2205.14304), enhanced with some original extensions. The base model includes 3 flows of information:

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
    Sentiment Analysis: 
  </li>
  <li>
    Back translation:
  </li>
  <li>
    Images frequency domain analysis:
  </li>
  <li>
    Squeeze and excitation layers:
  </li>
  <li>
    Concatenation of the embeddings:
  </li>
  <li>
    Focus on Region of Interest:
  </li>
</ul>


