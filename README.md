# FinBERT-Financial-Sentiment-Analyzer

Overview
FinBERT is a specialized BERT model pretrained from scratch, aimed at gaining insights from companies' earnings calls. The model has been trained and fine-tuned to excel at financial sentiment analysis, offering a robust tool for processing financial news, reports, and other text sources.

Key Features
Pretraining: FinBERT was pretrained from scratch on a large subset of Wikipedia and over 400,000 financial articles.
Finetuning: The model was finetuned on the SST-2 dataset for sentiment analysis, followed by finetuning on FiQA and additional financial sentiment datasets.
Implementation: FinBERT is built using Python, leveraging Keras and TensorFlow for deep learning workflows.
Performance: The model achieved an accuracy of 93% on the test dataset.

Model Structure
Pretraining Batch Size: 128
Finetuning Batch Size: 32
Sequence Length: 128
Model Layers: 8 Transformer layers
Model Dimension: 512
Dropout: 0.1

Install dependencies:
```
pip install tensorflow 
```

# Usage
To use the pretrained FinBERT model for sentiment analysis, download the model and use the provided inference script:


# Load the model
```
from tensorflow import keras

model = tf.keras.models.load_model("final_model.keras", compile=True)
```

# Run inference
```
SENTIMENT_MAPPING = {2: 'neutral', 1: 'positive', 0: 'negative'}

prediction = restored_model.predict(tf.constant(["Pre-tax gain totaled 0.3 million, compared to a loss of euro 8 million in the first quarter of 2005"]))
predicted_sentiment = SENTIMENT_MAPPING[tf.argmax(prediction, axis=1).numpy()[0]]
print(predicted_sentiment)
```


