# 24B4530_AIC
🚀 Local Setup Instructions

Follow these steps to run the notebooks locally on your machine.

📦 Requirements

1.Python 3.8 or higher

2.pip or conda

3.Recommended: Virtual environment (venv or conda)


🔧 Setup Steps
1. Clone the Repository
2. Create Virtual Environment (Optional but Recommended)
3. Install Dependencies
  pip install -r requirements.txt

Then open the following notebooks in your browser:
TQ1.ipynb
Q2.ipynb
q3.ipynb

✅ Notes

  Run each cell in sequence (Shift + Enter).

  If any errors related to missing modules occur, install them using pip install <module-name>.

  Make sure all required datasets or files (if any) are in the correct relative paths.

# Final Report: Comprehensive Analysis of NLP and CV Projects  

## Table of Contents  
1. [Project Overview](#1-project-overview)  
2. [Text Classification with BERT](#2-text-classification-with-bert) 
3. [Transfer Learning for Fashion-MNIST](#3-transfer-learning-for-fashion-mnist)  
4. [Retrieval-Augmented Generation (RAG) Chatbot](#4-retrieval-augmented-generation-rag-chatbot)
5. [General Troubleshooting](#5-general-troubleshooting)  
6. [Conclusion and Future Work](#6-conclusion-and-future-work)  
7. [References](#7-references)  

---

## 1. Project Overview  
This report consolidates the findings and outcomes from three distinct projects:  

1. **Retrieval-Augmented Generation (RAG) Chatbot**: A system for answering queries using context extracted from PDFs, enhanced with semantic retrieval and knowledge graphs.  
2. **Transfer Learning for Fashion-MNIST**: Adaptation of pretrained CNN models (ResNet50, VGG16, MobileNetV2) for classifying grayscale Fashion-MNIST images.  
3. **Text Classification with BERT**: Fine-tuning a BERT model for multi-class text classification, addressing challenges like label mapping and class imbalance.  

Each project involved rigorous experimentation, error resolution, and optimization to achieve state-of-the-art performance.  

---

## 2. Text Classification with BERT  

### Objectives  
- Fine-tune a BERT model (`bert-base-uncased`) for multi-class text classification (43 classes).  
- Address challenges like label mapping, class imbalance, and preprocessing.  

### Implementation  
- **Preprocessing**:  
  - Normalized text (lowercase, removed special characters), eliminated stop words, and applied lemmatization.  
  - Mapped string labels to integers for model compatibility.  
- **Model Training**:  
  - Used `AutoTokenizer` and `DataCollatorWithPadding` for tokenization.  
  - Incorporated class weights to handle imbalance via a custom `Trainer` class.  
- **Evaluation**:  
  - Tracked accuracy, precision, recall, and F1 score (weighted averages).  
  - Visualized training/validation loss curves for convergence analysis.  

### Results  
- Achieved robust performance with preprocessing and class weight adjustments.  
- Confirmed necessity of label mapping for string-to-integer conversion.  
- Resolved critical errors (e.g., `TypeError`, `RuntimeError`) through method updates and tensor type alignment.   

---

## 3. Transfer Learning for Fashion-MNIST  

### Objectives  
- Leverage pretrained CNN models (ResNet50, VGG16, MobileNetV2) to classify Fashion-MNIST images.  
- Optimize performance through fine-tuning and hyperparameter experimentation.  

### Implementation  
- **Data Pipeline**:  
  - Resized grayscale images to 224x224 and converted to 3-channel format.  
  - Used `tf.data.Dataset` for efficient data streaming and augmentation.  
- **Model Design**:  
  - Froze initial layers of pretrained models and added custom heads (Global Average Pooling, Dense, Dropout).  
  - Fine-tuned last few convolutional blocks with a reduced learning rate (`1e-5`).  
- **Training**:  
  - Applied data augmentation (flips, brightness jittering) and regularization (dropout, L2).  
  - Used `ReduceLROnPlateau` and `CosineDecay` for learning rate scheduling.  

### Results  
- Pretrained models outperformed scratch-trained models significantly.  
- MobileNetV2 achieved ~91% accuracy after fine-tuning.  
- MobileNetV2 demonstrated efficient performance with mixed precision training on Apple M3 GPU.  

---

## 4. Retrieval-Augmented Generation (RAG) Chatbot  

### Objectives  
- Build a chatbot capable of answering queries using contextually relevant information from PDFs.  
- Incorporate advanced NLP techniques like entity extraction and knowledge graph construction.  

### Implementation  
- **Ingestion & Indexing**:  
  - PDFs were parsed into semantically meaningful chunks using LangChain.  
  - Chunks were encoded into dense vectors using `all-MiniLM-L6-v2` and stored in a FAISS vector store.  
- **Retrieval & Generation**:  
  - User queries retrieved top-k chunks from FAISS.  
  - Context was appended to prompts and processed using the Groq API with models like `llama3-8b-8192`.  
- **Advanced Techniques**:  
  - Integrated `stanza` for Named Entity Recognition (NER) and dependency parsing.  
  - Constructed a knowledge graph using `networkx` for enhanced understanding.  

### Results  
- Successfully built a mini RAG chatbot with semantic grounding.  
- Knowledge graph visualization provided insights into relationships between entities.  
- Performance improved by filtering incomplete triples and controlling graph size.  

---

## 5. General Troubleshooting  

### Common Issues and Fixes  
| **Problem**                          | **Resolution**                                                                 |  
|--------------------------------------|--------------------------------------------------------------------------------|  
| Kernel crashes in Jupyter            | Reduced batch size; closed background apps.                                    |  
| Keras 3.x breaking changes          | Pinned to `tensorflow==2.15.0` and installed `tf-keras`.                      |  
| Label shape mismatches               | Flattened labels or switched to one-hot encoding.                              |  
| OOM errors during training           | Reduced batch size; used `tf.data.AUTOTUNE` and caching.                       |  
| "None cannot be a node" in KG        | Skipped triples with `None` values during graph construction.                  |  

### Best Practices  
- **Environment Isolation**: Used Conda environments for reproducibility.  
- **Incremental Testing**: Validated components (preprocessing, tokenization) independently.  
- **Documentation**: Maintained detailed logs and references for future iterations.  

---

## 6. Conclusion and Future Work  

### Summary of Achievements  
- Developed a modular RAG chatbot with semantic retrieval and knowledge graph capabilities.  
- Achieved high accuracy (~91%) on Fashion-MNIST using transfer learning.  
- Built a robust BERT-based text classifier with comprehensive error handling.  

### Future Recommendations  
- **RAG Chatbot**: Expand knowledge graph features for reasoning capabilities.  
- **Fashion-MNIST**: Experiment with lighter models (e.g., EfficientNet) for deployment.  
- **Text Classification**: Optimize hyperparameters and explore data augmentation.  

---

## 7. References  
- [Hugging Face Transformers: Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)  
- [TensorFlow Transfer Learning Guide](https://keras.io/api/applications/)  
- [FAISS Documentation](https://github.com/facebookresearch/faiss)  
- [NLTK Data](https://www.nltk.org/data.html)  

**Prepared by**: Anant  
**Date**: May 25, 2025  
