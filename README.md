# News Article Topic Classifier using Naive Bayes

A machine learning project that classifies BBC news articles into different categories using the Naive Bayes algorithm implemented from scratch in Python.

## Table of Contents

- [Overview](#overview)
- [Naive Bayes Algorithm Explained](#naive-bayes-algorithm-explained)
- [Example Walkthrough](#example-walkthrough)
- [Code Implementation](#code-implementation)
- [Dataset](#dataset)
- [Features](#features)
- [Results and Evaluation](#results-and-evaluation)
- [Requirements](#requirements)
- [Usage](#usage)
- [📚 Detailed Code Walkthrough](#detailed-code-walkthrough)

## Overview

This project implements a **Naive Bayes classifier** from scratch to categorize BBC news articles into topics like Sports, Politics, Business, Entertainment, and Technology. The implementation includes text preprocessing, model training, prediction, and comprehensive evaluation metrics with visualizations.

## Naive Bayes Algorithm Explained

### What is Naive Bayes?

Naive Bayes is a probabilistic classification algorithm based on **Bayes' Theorem** with a "naive" assumption that all features (words in our case) are independent of each other.

### Bayes' Theorem

```
P(Class|Document) = P(Document|Class) × P(Class) / P(Document)
```

Where:

- **P(Class|Document)**: Probability of a document belonging to a class (what we want to find)
- **P(Document|Class)**: Probability of seeing this document given the class
- **P(Class)**: Prior probability of the class
- **P(Document)**: Probability of the document (constant for all classes)

### The "Naive" Assumption

We assume that each word in a document is independent of every other word. So:

```
P(Document|Class) = P(word1|Class) × P(word2|Class) × ... × P(wordN|Class)
```

### Why Use Log Probabilities?

Since we multiply many small probabilities, we use logarithms to prevent numerical underflow:

```
log(P(Class|Document)) = log(P(Class)) + Σ log(P(word|Class))
```

## Example Walkthrough

Let's say we want to classify this article: **"The football team won the championship"**

### Step 1: Training Data

Imagine we have training data:

- **Sports**: "football team", "basketball game", "championship match"
- **Politics**: "government policy", "election results", "political party"

### Step 2: Calculate Prior Probabilities

```
P(Sports) = Number of Sports articles / Total articles = 1/2 = 0.5
P(Politics) = Number of Politics articles / Total articles = 1/2 = 0.5
```

### Step 3: Calculate Word Likelihoods

For the word "football":

```
P("football"|Sports) = (Count of "football" in Sports + α) / (Total words in Sports + α × Vocabulary size)
P("football"|Politics) = (Count of "football" in Politics + α) / (Total words in Politics + α × Vocabulary size)
```

Where α (alpha) is the **Laplace smoothing** parameter to handle unseen words.

### Step 4: Make Prediction

For our test sentence "The football team won the championship":

```
Score(Sports) = log(P(Sports)) + log(P("football"|Sports)) + log(P("team"|Sports)) + log(P("won"|Sports)) + log(P("championship"|Sports))
Score(Politics) = log(P(Politics)) + log(P("football"|Politics)) + log(P("team"|Politics)) + log(P("won"|Politics)) + log(P("championship"|Politics))
```

The class with the higher score wins!

## Code Implementation

### 1. Data Loading and Splitting (`load_data`, `split_data`)

```python
# Loads BBC News CSV file
all_articles, all_labels = load_data('BBC News Train.csv')

# Splits data into 80% training and 20% validation
train_articles, train_labels, val_articles, val_labels = split_data(all_articles, all_labels)
```

**What it does**: Reads the CSV file containing news articles and their categories, then randomly splits the data for training and testing.

### 2. Text Preprocessing (`preprocess_text`)

```python
def preprocess_text(text):
    # 1. Convert to lowercase
    # 2. Remove punctuation
    # 3. Remove numbers
    # 4. Split into words (tokenization)
    # 5. Remove stop words (common words like "the", "and", "is")
    return cleaned_tokens
```

**Example**:

- Input: "The U.S. government announced new policies!"
- Output: ['government', 'announced', 'new', 'policies']

**Why preprocessing?**: Raw text contains noise. Preprocessing standardizes text and focuses on meaningful words.

### 3. Naive Bayes Implementation (`MyNaiveBayes`)

#### Training Phase (`fit` method):

1. **Build Vocabulary**: Collect all unique words from training data
2. **Calculate Prior Probabilities**: `P(Class) = Documents in Class / Total Documents`
3. **Calculate Word Likelihoods**: For each word and class combination using Laplace smoothing

#### Prediction Phase (`predict` method):

1. For each test document, calculate log probabilities for each class
2. Return the class with the highest probability

### 4. Laplace Smoothing

```python
likelihood = (word_count + α) / (total_words_in_class + α × vocabulary_size)
```

**Why needed?**: If a word appears in the test data but not in the training data for a specific class, the probability would be 0, making the entire prediction 0. Laplace smoothing adds a small value (α=1) to avoid this.

### 5. Evaluation Metrics

#### Confusion Matrix

Shows how many articles were correctly/incorrectly classified for each category:

```
                Predicted
Actual    Sport  Politics  Business
Sport       85      2         3
Politics     1     78         1
Business     2      1        87
```

#### Precision and Recall

- **Precision**: Of all articles predicted as Sports, how many were actually Sports?
- **Recall**: Of all actual Sports articles, how many were correctly identified?

#### Visualization

The code generates bar charts showing the top 10 most probable words for each news category, helping understand what the model learned.

## Dataset

The project uses the **BBC News Dataset** with columns:

- **ArticleId**: Unique identifier
- **Article**: News article text
- **Category**: Topic category (Sport, Politics, Business, Entertainment, Tech)

## Features

### Core Features:

- ✅ **From-scratch implementation** of Naive Bayes
- ✅ **Text preprocessing pipeline** (lowercasing, punctuation removal, stop words filtering)
- ✅ **Laplace smoothing** for handling unseen words
- ✅ **Log probabilities** to prevent numerical underflow
- ✅ **Comprehensive evaluation** (accuracy, precision, recall, confusion matrix)
- ✅ **Data visualization** of top words per category

### Advanced Features:

- 📊 **Interactive plots** showing most characteristic words per category
- 🔀 **Random data shuffling** for unbiased train/test split
- 📈 **Detailed performance metrics** for each category
- 🎯 **Example predictions** with actual vs predicted labels

## Results and Evaluation

The model provides:

1. **Overall accuracy percentage** on validation data
2. **Confusion matrix** showing classification performance per category
3. **Precision and Recall** for each news category
4. **Visual analysis** of top discriminative words
5. **Sample predictions** with explanations

## Requirements

```
matplotlib
```

No external ML libraries required - pure Python implementation!

## Usage

1. **Ensure your BBC News CSV file is in the project directory**
2. **Run the script**:
   ```bash
   python project.py
   ```
3. **View results**:
   - Console output shows training progress and metrics
   - Plot window displays top words per category
   - Close the plot window to finish execution

## How the Algorithm Works in This Scenario

### Real Example from the Code:

1. **Training**: The model learns that words like "goal", "match", "player" frequently appear in Sports articles
2. **Vocabulary Building**: Creates a set of all unique words across all articles
3. **Probability Calculation**: For each category, calculates how likely each word is to appear
4. **Classification**: When given a new article about "football championship", it calculates:
   - How likely is this article to be Sports? (high probability for "football", "championship")
   - How likely is this article to be Politics? (low probability for these words)
   - Chooses the category with highest probability

### Key Strengths:

- ✅ **Fast training and prediction**
- ✅ **Works well with limited data**
- ✅ **Interpretable results** (can see which words influence decisions)
- ✅ **Handles new words** gracefully with smoothing

### Limitations:

- ❌ **Independence assumption** (words aren't truly independent)
- ❌ **Doesn't capture word order** ("not good" vs "good not")
- ❌ **Sensitive to skewed data** (some categories have more examples)

This implementation demonstrates fundamental NLP and machine learning concepts while providing a complete, working text classifier!

## 📚 Detailed Code Walkthrough

For a **complete step-by-step breakdown** of every function with mathematical formulas and practical examples, see:

**[📖 CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)**

This detailed guide includes:

- 🧮 **Exact mathematical formulas** used in each function
- 📝 **Line-by-line code explanations** with examples
- 📊 **Numerical examples** showing calculations
- 🔍 **Algorithm insights** and computational complexity
- 💡 **Why each step is necessary** for the classification process

Perfect for understanding the implementation details and mathematical foundations!
