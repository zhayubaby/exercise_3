import nltk
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg, stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

# Download NLTK data (if not already downloaded)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("gutenberg")
nltk.download("vader_lexicon")

# Load Moby Dick from the Gutenberg dataset
moby_dick_text = gutenberg.raw("melville-moby_dick.txt")

# Tokenization
tokens = word_tokenize(moby_dick_text)

# Stopwords filtering
stop_words = set(stopwords.words("english"))
filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = pos_tag(filtered_tokens)

# Sentiment analysis via VADER
analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(sentence)["compound"] for sentence in nltk.sent_tokenize(moby_dick_text)]
average_sentiment = sum(sentiments) / len(sentiments)

if average_sentiment > 0.05:
    overall_sentiments = 'positive'
if average_sentiment < 0.05:
    overall_sentiments = 'negative'

print("Average Sentiment Score:", average_sentiment)
print("Overall Text Sentiment:", overall_sentiments)

# Count POS frequency
pos_freq = FreqDist(tag for word, tag in pos_tags)
common_pos = pos_freq.most_common(5)

print("5 Most Common Parts of Speech and Their Frequencies:")
for pos, count in common_pos:
    print(f"{pos}: {count}")

# Lemmatization
lemmatizer = WordNetLemmatizer()
top_20_tokens = [word for word, _ in FreqDist(filtered_tokens).most_common(20)]
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in top_20_tokens]

# Plotting frequency distribution
pos_freq.plot(20, title=" POS Frequency Distribution ")
plt.show()

