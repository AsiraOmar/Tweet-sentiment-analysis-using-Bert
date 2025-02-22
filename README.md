# Tweet-sentiment-analysis-using-Bert

This project performs sentiment analysis on tweets, identifying whether they are complaints or non-complaints. The sentiment analysis is powered by BERT, a transformer-based model known for its strong language understanding capabilities. This README will guide you through the steps to set up and use the model.

## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers library
- NumPy
- Matplotlib

You can install the required libraries using pip:

```
pip install torch transformers numpy matplotlib
```

## Project Overview

The main goal of this project is to classify tweets as either complaints or non-complaints. The classification is based on sentiment analysis, where the model identifies whether the sentiment expressed in the tweet is negative or positive.

## Dataset

The dataset consists of tweets that have been labeled as complaints or non-complaints. Each tweet is associated with a unique ID and the tweet text. We preprocess the tweets by tokenizing them and converting them into a format that BERT can understand.

## Model

We use the BERT model from Hugging Face Transformers. BERT is fine-tuned on the preprocessed tweet dataset to classify tweets into two categories: complaints and non-complaints.

The process includes:
- Preprocessing tweets by tokenizing the text.
- Fine-tuning the BERT model on the training data.
- Evaluating the model on a validation set.
- Using the trained model to predict sentiment on a test set.

## Steps to Run the Code

1. **Clone the Repository**

   Clone this repository to your local machine:

   ```
   git clone <repository_url>
   ```

2. **Prepare the Dataset**

   The dataset should be placed in a CSV file where each row contains a tweet and its corresponding label. The CSV file should have columns for tweet text and tweet ID.

3. **Preprocess the Data**

   The preprocessing step involves tokenizing the tweet text using BERT’s tokenizer. This prepares the data for input into the model.

4. **Train the Model**

   Run the following script to start training the model on the tweet dataset:

   ```
   python train.py
   ```

   This will fine-tune the BERT model on the training data. The model will be trained for a specified number of epochs, and the training process will be displayed on the console.

5. **Evaluate the Model**

   Once the model is trained, you can evaluate its performance on a validation dataset. The evaluation metrics such as accuracy and loss will be printed to the console.

6. **Make Predictions**

   After training, use the model to predict the sentiment of new tweets. You can provide a set of tweets and the model will classify them as complaints or non-complaints.

## Evaluation Metrics

- **Accuracy**: The percentage of correctly predicted labels.
- **AUC (Area Under the Curve)**: Measures the ability of the model to distinguish between the two classes.

## Visualizations

The script also includes functionality to visualize the predicted probabilities for non-complaint tweets. A histogram of predicted probabilities is plotted to help understand the distribution of the model's confidence.

## Example Output

After running the model, you will get outputs like:

```
Number of tweets predicted non-negative: 1464
```

You will also see visualizations of the distribution of predicted probabilities, which helps in understanding how confident the model is in its predictions.

## Conclusion

This sentiment analysis model can effectively classify tweets into complaints or non-complaints. Fine-tuning BERT on tweet data allows it to understand the context and sentiments expressed in short, informal text, making it a powerful tool for social media sentiment analysis.

## License

This project is licensed under the MIT License.

