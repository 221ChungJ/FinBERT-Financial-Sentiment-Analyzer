# FinBERT-Financial-Sentiment-Analyzer

FinBERT is a specialized BERT model pretrained from scratch, aimed at gaining insights from companies' earnings calls. The model has been trained and fine-tuned to excel at financial sentiment analysis, offering a robust tool for processing financial news, reports, and other text sources.

Key Features: <br>
Pretraining: FinBERT was pretrained from scratch on a large subset of Wikipedia and over 400,000 financial articles.<br>
Finetuning: The model was finetuned on the SST-2 dataset for sentiment analysis, followed by finetuning on FiQA and additional financial sentiment datasets.<br>
Implementation: FinBERT is built using Python, leveraging Keras and TensorFlow for deep learning workflows.<br>
Performance: The model achieved an accuracy of 93% on the test dataset.<br>

Model Structure:<br>
Pretraining Batch Size: 128<br>
Finetuning Batch Size: 32<br>
Sequence Length: 128<br>
Model Layers: 8 Transformer layers<br>
Model Dimension: 512<br>
Dropout: 0.1<br>

Install dependencies:
```
pip install tensorflow 
```

# Usage
To use the pretrained FinBERT model for sentiment analysis, download the model and use the provided inference script:


# Load the model
```
import tensorflow
from tensorflow import keras

model = keras.models.load_model("final_model.keras", compile=True)
```

# Run inference
```
SENTIMENT_MAPPING = {2: 'neutral', 1: 'positive', 0: 'negative'}

prediction = restored_model.predict(tf.constant(["Pre-tax gain totaled 0.3 million, compared to a loss of euro 8 million in the first quarter of 2005"]))
predicted_sentiment = SENTIMENT_MAPPING[tf.argmax(prediction, axis=1).numpy()[0]]
print(predicted_sentiment)
```


