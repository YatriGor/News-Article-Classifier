import csv
import random
import string
import math

import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# --- 1. DATA LOADING AND SPLITTING ---


def load_data(filename):
    """Loads the BBC CSV file        and returns articles and labels."""
    articles = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row (ArticleId, Article, Category)
        for row in reader:
            # Assuming row[1] is the Article text and row[2] is the Category
            articles.append(row[1])
            labels.append(row[2])
    return articles, labels


def split_data(articles, labels, test_split_ratio=0.3):
    """Shuffles and splits the data into training and validation sets."""

    # Combine articles and labels to shuffle them together
    combined = list(zip(articles, labels))
    random.shuffle(combined)

    # Unzip them after shuffling
    shuffled_articles, shuffled_labels = zip(*combined)

    # Decide the split index
    split_index = int(len(shuffled_articles) * (1 - test_split_ratio))

    # Split the data
    train_articles = shuffled_articles[:split_index]
    train_labels = shuffled_labels[:split_index]

    val_articles = shuffled_articles[split_index:]
    val_labels = shuffled_labels[split_index:]

    return train_articles, train_labels, val_articles, val_labels


# --- Main execution ---
print("Loading data...")
# Make sure 'BBC News Train.csv' is in the same folder as your script
all_articles, all_labels = load_data('BBC News Train.csv')

# Split the data
train_articles, train_labels, val_articles, val_labels = split_data(
    all_articles, all_labels)

print(f"Total articles: {len(all_articles)}")
print(f"Training articles: {len(train_articles)}")
print(f"Validation articles: {len(val_articles)}")

# You now have 4 lists ready for your model:
# train_articles, train_labels (for your .fit() method)
# val_articles, val_labels (for your .predict() method and accuracy check)


# --- 2. TEXT PREPROCESSING ---

# A simple list of common English stop words.
# You can add more to this list if you want.
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'down',
    'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i',
    'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most',
    'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only',
    'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same',
    'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were',
    'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
    'you', 'your', 'yours', 'yourself', 'yourselves'
}


def preprocess_text(text, remove_stopwords=False):
    """
    Cleans and tokenizes a single text string.
    Args:
        text: Input text string
        remove_stopwords: If True, removes stop words; if False, keeps all words
    """

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    # Creates a translator that replaces all punctuation with a space
    translator = str.maketrans(
        string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    # 3. Remove numbers (optional, but good for this task)
    text = ''.join(char for char in text if not char.isdigit())

    # 4. Tokenize (split into words)
    tokens = text.split()

    # 5. Remove stop words (conditionally)
    if remove_stopwords:
        cleaned_tokens = []
        for word in tokens:
            if word not in STOP_WORDS:
                cleaned_tokens.append(word)
        return cleaned_tokens
    else:
        # Keep all words (for normal model to maintain realistic accuracy)
        return tokens


# --- Preprocess your split data ---
print("\nPreprocessing data...")

# Preprocess for normal model (keep stop words for realistic accuracy)
X_train_processed = [preprocess_text(article, remove_stopwords=False) for article in train_articles]
X_val_processed = [preprocess_text(article, remove_stopwords=False) for article in val_articles]

# Preprocess for TF-IDF model (remove stop words to focus on contextual words)
X_train_processed_tfidf = [preprocess_text(article, remove_stopwords=True) for article in train_articles]
X_val_processed_tfidf = [preprocess_text(article, remove_stopwords=True) for article in val_articles]

print("Preprocessing complete.")
print(f"Example original article: {train_articles[0][:100]}...")
print(f"Example processed article (normal): {X_train_processed[0][:15]}...")
print(f"Example processed article (TF-IDF, stop words removed): {X_train_processed_tfidf[0][:15]}...")


# --- 2.5. TF-IDF COMPUTATION (FROM SCRATCH) ---

def compute_idf(documents):
    """
    Computes IDF (Inverse Document Frequency) for each word in the vocabulary.
    IDF(word) = log(N / DF(word))
    where N = total documents, DF = number of documents containing that word.
    """
    N = len(documents)
    document_frequency = defaultdict(int)  # DF(word)
    
    # Count in how many documents each word appears
    for doc in documents:
        unique_words_in_doc = set(doc)
        for word in unique_words_in_doc:
            document_frequency[word] += 1
    
    # Compute IDF for each word
    idf_scores = {}
    for word, df in document_frequency.items():
        if df > 0:
            idf_scores[word] = math.log(N / df)
        else:
            idf_scores[word] = 0.0
    
    return idf_scores


def compute_tfidf(documents, idf_scores):
    """
    Computes TF-IDF for each document.
    TF-IDF(word, doc) = (count(word in doc) / total words in doc) * IDF(word)
    
    Normalizes each document's TF-IDF vector so values sum to 1.
    
    Returns a list of dictionaries, where each dict maps word -> TF-IDF score.
    """
    tfidf_documents = []
    
    for doc in documents:
        if len(doc) == 0:
            tfidf_documents.append({})
            continue
        
        # Count word frequencies in this document
        word_counts = defaultdict(int)
        for word in doc:
            word_counts[word] += 1
        
        total_words = len(doc)
        
        # Compute TF-IDF for each word in the document
        doc_tfidf = {}
        for word, count in word_counts.items():
            tf = count / total_words  # Term Frequency
            idf = idf_scores.get(word, 0.0)
            doc_tfidf[word] = tf * idf
        
        # Normalize TF-IDF vector so values sum to 1
        sum_tfidf = sum(doc_tfidf.values())
        if sum_tfidf > 0:
            for word in doc_tfidf:
                doc_tfidf[word] /= sum_tfidf
        
        tfidf_documents.append(doc_tfidf)
    
    return tfidf_documents


# --- 3. NAIVE BAYES CLASSIFIER ---

class MyNaiveBayes:
    def __init__(self, smoothing_alpha=1, use_tfidf=False):
        self.alpha = smoothing_alpha  # This is for Laplace Smoothing
        self.use_tfidf = use_tfidf    # Flag to use TF-IDF weights instead of raw counts
        self.class_priors = {}        # Stores P(Class)
        self.word_likelihoods = {}    # Stores P(Word | Class)
        self.vocabulary = set()
        self.classes = set()
        self.last_predictions = []
        self.last_confidences = []
        self.last_posteriors = []
        self.tfidf_train = None       # Store TF-IDF weights for training data

    def fit(self, X_train, y_train, tfidf_train=None):
        """
        Trains the model on the training data.
        If use_tfidf=True, tfidf_train should be provided (list of dicts: word -> TF-IDF score).
        """
        mode_str = "TF-IDF weighted" if self.use_tfidf else "raw counts"
        print(f"\nFitting model (using {mode_str})...")
        num_docs = len(X_train)
        self.classes = set(y_train)

        # --- Build Vocabulary ---
        for doc in X_train:
            self.vocabulary.update(doc)

        vocab_size = len(self.vocabulary)

        # --- Calculate Class Priors P(Class) ---
        for c in self.classes:
            docs_in_class = sum(1 for label in y_train if label == c)
            self.class_priors[c] = docs_in_class / num_docs

        # --- Calculate Word Likelihoods P(Word | Class) ---
        for c in self.classes:
            self.word_likelihoods[c] = {}

            if self.use_tfidf and tfidf_train is not None:
                # Use TF-IDF weights instead of raw counts
                # Sum TF-IDF weights for each word across all documents in this class
                # Note: doc_tfidf already contains TF-IDF scores (TF accounts for frequency)
                word_tfidf_sums = defaultdict(float)
                total_tfidf_sum = 0.0
                
                for doc, label, doc_tfidf in zip(X_train, y_train, tfidf_train):
                    if label == c:
                        # Iterate over unique words in doc_tfidf (each word's TF-IDF already accounts for frequency)
                        for word, tfidf_score in doc_tfidf.items():
                            if word in self.vocabulary:
                                word_tfidf_sums[word] += tfidf_score
                                total_tfidf_sum += tfidf_score
                
                # Calculate likelihood for each word in the *entire* vocabulary
                for word in self.vocabulary:
                    tfidf_sum = word_tfidf_sums.get(word, 0.0)
                    
                    # Apply smoothing: add alpha to numerator, add vocab_size * alpha to denominator
                    # This ensures all words have non-zero probability
                    likelihood = (tfidf_sum + self.alpha) / \
                        (total_tfidf_sum + (vocab_size * self.alpha))
                    self.word_likelihoods[c][word] = likelihood
            else:
                # Original method: use raw frequency counts
                # Get all words from all docs in this class
                all_words_in_class = []
                for doc, label in zip(X_train, y_train):
                    if label == c:
                        all_words_in_class.extend(doc)

                total_word_count_in_class = len(all_words_in_class)

                # Count frequency of each word in this class
                word_counts = {}
                for word in all_words_in_class:
                    word_counts[word] = word_counts.get(word, 0) + 1

                # Calculate likelihood for each word in the *entire* vocabulary
                for word in self.vocabulary:
                    count = word_counts.get(word, 0)

                    # --- This is the Laplace Smoothing formula ---
                    likelihood = (count + self.alpha) / \
                        (total_word_count_in_class + (vocab_size * self.alpha))
                    self.word_likelihoods[c][word] = likelihood

        print("Fit complete.")

    def predict(self, X_test):
        """Predicts the class for a list of processed documents."""
        predictions = []
        for doc in X_test:
            # --- Use Log Probabilities to prevent underflow ---
            scores = {}
            for c in self.classes:
                # Start with the log of the class prior
                scores[c] = math.log(self.class_priors[c])

                # Add the log likelihood for each word *in the document*
                for word in doc:
                    if word in self.vocabulary:
                        scores[c] += math.log(self.word_likelihoods[c][word])

            # The class with the highest log-score is the winner
            predicted_class = max(scores, key=scores.get)
            predictions.append(predicted_class)

        return predictions

    def predict_with_confidence(self, X_test):
        """
        Predicts the class for each document and returns confidence scores
        derived from posterior probabilities.
        """
        predictions = []
        confidences = []
        posterior_distributions = []

        for doc in X_test:
            log_scores = {}
            for c in self.classes:
                log_prob = math.log(self.class_priors[c])
                for word in doc:
                    if word in self.vocabulary:
                        log_prob += math.log(self.word_likelihoods[c][word])
                log_scores[c] = log_prob

            predicted_class = max(log_scores, key=log_scores.get)

            max_log_score = max(log_scores.values())
            exp_scores = {c: math.exp(score - max_log_score)
                          for c, score in log_scores.items()}
            normalization_constant = sum(exp_scores.values())
            posteriors = {c: exp_score / normalization_constant
                          for c, exp_score in exp_scores.items()}

            predictions.append(predicted_class)
            confidences.append(posteriors[predicted_class])
            posterior_distributions.append(posteriors)

        self.last_predictions = predictions
        self.last_confidences = confidences
        self.last_posteriors = posterior_distributions

        return predictions, confidences, posterior_distributions


# --- 4. DETAILED EVALUATION METRICS ---

def build_confusion_matrix(y_true, y_pred, classes):
    """Builds a confusion matrix as a dictionary of dictionaries."""

    # Create a blank matrix (a 2D dictionary)
    matrix = {}
    for true_class in classes:
        matrix[true_class] = {}
        for pred_class in classes:
            matrix[true_class][pred_class] = 0

    # Populate the matrix
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1

    return matrix


def print_confusion_matrix(matrix, classes):
    """Prints a formatted confusion matrix."""

    print("\n--- Confusion Matrix ---")

    # Print header
    header = " " * 15 + " |".join([f" {c[:5]:<5} " for c in classes])
    print(header)
    print("-" * len(header))

    # Print rows
    for true_class in classes:
        row = f"Actual {true_class[:5]:<8} |"
        for pred_class in classes:
            count = matrix[true_class][pred_class]
            row += f" {count:<5} |"
        print(row)


def calculate_precision_recall(matrix, classes):
    """Calculates Precision and Recall for each class."""

    print("\n--- Precision & Recall per Class ---")

    metrics = {}

    for c in classes:
        # --- Precision ---
        # (True Positives) / (Total Predicted as this Class)
        true_positives = matrix[c][c]
        total_predicted_as_c = 0
        for true_label in classes:
            total_predicted_as_c += matrix[true_label][c]

        precision = 0
        if total_predicted_as_c > 0:
            precision = true_positives / total_predicted_as_c

        # --- Recall ---
        # (True Positives) / (Total Actually in this Class)
        total_actual_as_c = 0
        for pred_label in classes:
            total_actual_as_c += matrix[c][pred_label]

        recall = 0
        if total_actual_as_c > 0:
            recall = true_positives / total_actual_as_c

        metrics[c] = {"precision": precision, "recall": recall}

        print(f"Class: {c}")
        print(f"  -> Precision: {precision:.4f}")
        print(f"  -> Recall:    {recall:.4f}")
    return metrics


def plot_top_words(model, classes, top_n=10):
    """
    Plots the top N most probable words for each class.
    This will open in a new window.
    """
    print("\n--- Generating Visualizations ---")

    # Set up the plot grid
    # We have 5 classes, so a 3x2 grid is good
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()  # Makes it easy to loop through

    for i, c in enumerate(classes):
        # Get the dictionary of word likelihoods for this class
        class_likelihoods = model.word_likelihoods[c]

        # Sort the words by likelihood (highest first)
        sorted_words = sorted(class_likelihoods.items(),
                              key=lambda item: item[1], reverse=True)

        # Get the top N words and their scores
        top_words = [word for word, score in sorted_words[:top_n]]
        top_scores = [score for word, score in sorted_words[:top_n]]

        # Plot
        ax = axes[i]
        ax.barh(top_words, top_scores, color='skyblue')
        ax.set_title(f"Top 10 Words for Category: '{c}'")
        ax.set_xlabel("Probability (Likelihood)")
        ax.invert_yaxis()  # Puts the most probable word at the top

    # Hide any unused subplots
    for j in range(len(classes), len(axes)):
        axes[j].axis('off')

    fig.tight_layout(pad=3.0)
    print("Close the plot window to finish the script.")
    plt.show()  # <-- This command opens the new window


# --- 5. COMPUTE TF-IDF FOR TRAINING DATA ---
print("\n--- Computing TF-IDF (using preprocessed data without stop words) ---")
idf_scores = compute_idf(X_train_processed_tfidf)
tfidf_train = compute_tfidf(X_train_processed_tfidf, idf_scores)
print("TF-IDF computation complete (vectors normalized to sum to 1).")


# --- 6. TRAIN, PREDICT, AND EVALUATE ---

def calculate_accuracy(y_true, y_pred):
    correct_count = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_count += 1
    return correct_count / len(y_true)


# ===== TRAIN NORMAL NAIVE BAYES (RAW COUNTS) =====
print("\n" + "="*60)
print("TRAINING NORMAL NAIVE BAYES (Raw Counts)")
print("="*60)

model_normal = MyNaiveBayes(smoothing_alpha=1, use_tfidf=False)
model_normal.fit(X_train_processed, train_labels)
y_predictions_normal = model_normal.predict(X_val_processed)

# Get a sorted list of your class names
all_classes = sorted(list(model_normal.classes))

# Evaluate normal model
accuracy_normal = calculate_accuracy(val_labels, y_predictions_normal)
print(f"\n--- Normal Naive Bayes Evaluation ---")
print(f"Validation Accuracy: {accuracy_normal * 100:.2f}%")

conf_matrix_normal = build_confusion_matrix(val_labels, y_predictions_normal, all_classes)
print_confusion_matrix(conf_matrix_normal, all_classes)
metrics_normal = calculate_precision_recall(conf_matrix_normal, all_classes)


# ===== TRAIN TF-IDF WEIGHTED NAIVE BAYES =====
print("\n" + "="*60)
print("TRAINING TF-IDF WEIGHTED NAIVE BAYES")
print("="*60)

model_tfidf = MyNaiveBayes(smoothing_alpha=1.5, use_tfidf=True)  # Increased smoothing for TF-IDF
model_tfidf.fit(X_train_processed_tfidf, train_labels, tfidf_train=tfidf_train)
y_predictions_tfidf = model_tfidf.predict(X_val_processed_tfidf)

# Evaluate TF-IDF model
accuracy_tfidf = calculate_accuracy(val_labels, y_predictions_tfidf)
print(f"\n--- TF-IDF Weighted Naive Bayes Evaluation ---")
print(f"Validation Accuracy: {accuracy_tfidf * 100:.2f}%")

conf_matrix_tfidf = build_confusion_matrix(val_labels, y_predictions_tfidf, all_classes)
print_confusion_matrix(conf_matrix_tfidf, all_classes)
metrics_tfidf = calculate_precision_recall(conf_matrix_tfidf, all_classes)


# ===== COMPARISON SUMMARY =====
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"Normal Naive Bayes (Raw Counts):     {accuracy_normal * 100:.2f}%")
print(f"TF-IDF Weighted Naive Bayes:        {accuracy_tfidf * 100:.2f}%")
print(f"Difference:                          {abs(accuracy_tfidf - accuracy_normal) * 100:.2f}%")
print("="*60)

# Use the TF-IDF model for subsequent visualizations (or you can change to model_normal)
model = model_tfidf
y_predictions = y_predictions_tfidf
conf_matrix = conf_matrix_tfidf
metrics = metrics_tfidf


# --- CONFIDENCE-BASED OUTLIER ANALYSIS & VISUALIZATION ---
print("\n--- Confidence Analysis ---")
# Use appropriate validation data based on which model is selected
val_data_for_confidence = X_val_processed_tfidf if model.use_tfidf else X_val_processed
conf_predictions, confidences, posterior_distributions = model.predict_with_confidence(
    val_data_for_confidence)

confidence_threshold = 0.6
low_confidence_indices = [
    idx for idx, score in enumerate(confidences) if score < confidence_threshold]

print(f"Confidence threshold: {confidence_threshold:.2f}")
print(
    f"Low-confidence predictions (< threshold): {len(low_confidence_indices)} / {len(confidences)}")

if low_confidence_indices:
    print("\nSample low-confidence cases:")
    for sample_idx in low_confidence_indices[:5]:
        print(f"- Article snippet: {val_articles[sample_idx][:80]}...")
        print(f"  True label: {val_labels[sample_idx]}")
        print(f"  Predicted: {conf_predictions[sample_idx]}")
        print(
            f"  Confidence: {confidences[sample_idx]:.3f}")

# Confidence distribution histogram
plt.figure(figsize=(10, 6))
plt.hist(confidences, bins=20, color='steelblue', edgecolor='black')
plt.axvline(confidence_threshold, color='red',
            linestyle='--', linewidth=2, label=f'Threshold = {confidence_threshold:.2f}')
plt.title("Validation Confidence Distribution")
plt.xlabel("Posterior Probability of Predicted Class")
plt.ylabel("Number of Articles")
plt.legend()
plt.tight_layout()
plt.show()

# Per-class accuracy bar chart
per_class_accuracy = []
for c in all_classes:
    total_actual = sum(conf_matrix[c].values())
    accuracy_c = conf_matrix[c][c] / \
        total_actual if total_actual > 0 else 0.0
    per_class_accuracy.append(accuracy_c)

plt.figure(figsize=(10, 6))
bars = plt.bar(all_classes, per_class_accuracy, color='mediumseagreen')
plt.ylim(0, 1)
plt.title("Per-Class Accuracy (Validation Set)")
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha='right')

for bar, acc in zip(bars, per_class_accuracy):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(
    ) + 0.01, f"{acc:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Optional word cloud visualization per class
if WORDCLOUD_AVAILABLE:
    tokens_by_class = defaultdict(list)
    for tokens, label in zip(X_train_processed, train_labels):
        tokens_by_class[label].extend(tokens)

    rows = math.ceil(len(all_classes) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    if hasattr(axes, "ravel"):
        axes = axes.ravel()
    else:
        axes = [axes]

    for idx, cls in enumerate(all_classes):
        ax = axes[idx]
        token_string = " ".join(tokens_by_class[cls])
        if token_string.strip():
            wordcloud = WordCloud(width=800, height=400,
                                  background_color='white').generate(token_string)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f"Word Cloud: {cls}")
        else:
            ax.text(0.5, 0.5, "No data available",
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('off')

    for idx in range(len(all_classes), len(axes)):
        axes[idx].axis('off')

    fig.tight_layout(pad=3.0)
    plt.show()
else:
    print(
        "WordCloud package is not installed. Skipping per-class word cloud visualization.")


# Print a few examples
print("\nExample Predictions:")
for i in range(5):
    print(f"Article (start): {val_articles[i][:50]}...")
    print(f"  -> True Category: {val_labels[i]}")
    print(f"  -> Predicted Category: {y_predictions[i]}\n")

# --- THIS IS THE FINAL CORRECTED PART ---
plot_top_words(model, all_classes)