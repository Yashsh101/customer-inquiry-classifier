# Customer Inquiry Classification System
# Author: Yash Sharma
# End-to-end NLP pipeline for automated customer query classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class CustomerInquiryClassifier:
    def __init__(self, random_state=42):
        """
        Initialize Customer Inquiry Classification Pipeline
        
        Parameters:
        random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.pipeline = None
        self.vectorizer = None
        self.model = None
        self.categories = None
        
        # Define inquiry categories
        self.category_mapping = {
            'billing': 0,
            'technical_support': 1,
            'product_inquiry': 2,
            'shipping': 3,
            'refund_return': 4,
            'account_management': 5,
            'general_inquiry': 6
        }
        
        self.reverse_mapping = {v: k for k, v in self.category_mapping.items()}
    
    def generate_synthetic_data(self, n_samples=3000):
        """
        Generate synthetic customer inquiry data
        
        Parameters:
        n_samples (int): Number of samples to generate
        
        Returns:
        pd.DataFrame: Generated dataset
        """
        np.random.seed(self.random_state)
        
        # Sample inquiries for each category
        inquiry_templates = {
            'billing': [
                "I have a question about my bill for this month",
                "Why was I charged twice for the same service",
                "Can you explain the charges on my account",
                "I need help understanding my invoice",
                "There's an error in my billing statement",
                "How can I update my payment information",
                "I want to dispute a charge on my bill",
                "Can I get a refund for overcharges",
                "My payment didn't go through, what should I do",
                "I need a copy of my billing history"
            ],
            'technical_support': [
                "My device is not working properly",
                "I'm having trouble logging into my account",
                "The application keeps crashing on my phone",
                "I need help setting up my new device",
                "The website is not loading correctly",
                "I forgot my password and can't reset it",
                "There's a bug in the software",
                "My internet connection is very slow",
                "I'm getting error messages when I try to use the service",
                "The app won't sync with my other devices"
            ],
            'product_inquiry': [
                "What are the features of this product",
                "Do you have this item in different colors",
                "Can you tell me more about the warranty",
                "Is this product compatible with my device",
                "What's the difference between these two models",
                "When will new products be available",
                "I need specifications for this item",
                "Are there any discounts available",
                "Can I get a product demonstration",
                "What accessories come with this product"
            ],
            'shipping': [
                "When will my order be delivered",
                "I haven't received my package yet",
                "Can I track my shipment",
                "I need to change my delivery address",
                "My package was damaged during shipping",
                "Can I expedite my shipping",
                "What shipping options do you offer",
                "I received the wrong item in my order",
                "Can I schedule a specific delivery time",
                "My tracking number isn't working"
            ],
            'refund_return': [
                "I want to return this item",
                "How do I get a refund for my purchase",
                "What is your return policy",
                "I'm not satisfied with my order",
                "Can I exchange this for a different size",
                "I ordered the wrong item by mistake",
                "The product doesn't match the description",
                "I need to cancel my recent order",
                "How long does the refund process take",
                "Can I return this item without the receipt"
            ],
            'account_management': [
                "I want to update my profile information",
                "How do I close my account",
                "I need to change my email address",
                "Can I upgrade my service plan",
                "I want to add another user to my account",
                "How do I download my account data",
                "I need to verify my account",
                "Can I merge two accounts",
                "I want to change my subscription settings",
                "How do I enable two-factor authentication"
            ],
            'general_inquiry': [
                "What are your business hours",
                "Do you have a physical store location",
                "How can I contact customer service",
                "What payment methods do you accept",
                "Are you hiring for any positions",
                "I have a suggestion for improvement",
                "Can I speak to a supervisor",
                "What is your privacy policy",
                "Do you offer student discounts",
                "How long has your company been in business"
            ]
        }
        
        # Generate synthetic data
        data = []
        samples_per_category = n_samples // len(inquiry_templates)
        
        for category, templates in inquiry_templates.items():
            for _ in range(samples_per_category):
                # Add variations to templates
                base_template = np.random.choice(templates)
                
                # Add random variations
                variations = [
                    "Hello, " + base_template.lower(),
                    "Hi there, " + base_template.lower(),
                    base_template + ". Please help me.",
                    "Can you help me? " + base_template.lower(),
                    base_template + " Thanks!",
                    "URGENT: " + base_template.lower(),
                    base_template + " I need assistance.",
                    "Please assist: " + base_template.lower(),
                    base_template,
                    base_template.upper()
                ]
                
                inquiry = np.random.choice(variations)
                
                # Add some noise occasionally
                if np.random.random() < 0.1:
                    noise_words = ["please", "help", "urgent", "asap", "thanks", "hello", "hi"]
                    inquiry += " " + " ".join(np.random.choice(noise_words, size=np.random.randint(1, 3)))
                
                data.append({
                    'inquiry_text': inquiry,
                    'category': category,
                    'category_id': self.category_mapping[category]
                })
        
        return pd.DataFrame(data)
    
    def preprocess_text(self, text):
        """
        Preprocess text data
        
        Parameters:
        text (str): Input text
        
        Returns:
        str: Processed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def create_pipeline(self, model_type='svc'):
        """
        Create machine learning pipeline
        
        Parameters:
        model_type (str): Type of model to use ('svc', 'logistic', 'nb', 'rf')
        
        Returns:
        Pipeline: Scikit-learn pipeline
        """
        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 2)
        )
        
        # Choose model
        if model_type == 'svc':
            model = LinearSVC(random_state=self.random_state, max_iter=10000)
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif model_type == 'nb':
            model = MultinomialNB()
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError("Invalid model type. Choose from: 'svc', 'logistic', 'nb', 'rf'")
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', model)
        ])
        
        return pipeline
    
    def train_model(self, df, model_type='svc', test_size=0.2):
        """
        Train the classification model
        
        Parameters:
        df (pd.DataFrame): Training data
        model_type (str): Type of model to use
        test_size (float): Test set size
        
        Returns:
        dict: Training results
        """
        print("Preprocessing text data...")
        
        # Preprocess text
        df['processed_text'] = df['inquiry_text'].apply(self.preprocess_text)
        
        # Prepare features and target
        X = df['processed_text']
        y = df['category_id']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Create and train pipeline
        print(f"Training {model_type.upper()} model...")
        self.pipeline = self.create_pipeline(model_type)
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            y_pred_proba = self.pipeline.named_steps['classifier'].predict_proba(
                self.pipeline.named_steps['tfidf'].transform(X_test)
            )
        elif hasattr(self.pipeline.named_steps['classifier'], 'decision_function'):
            decision_scores = self.pipeline.named_steps['classifier'].decision_function(
                self.pipeline.named_steps['tfidf'].transform(X_test)
            )
            # Convert to probabilities using softmax for multi-class
            if len(decision_scores.shape) > 1:
                y_pred_proba = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=list(self.category_mapping.keys()),
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'X_test': X_test
        }
        
        print(f"Model training completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return results
    
    def predict_inquiry(self, text):
        """
        Predict category for a single inquiry
        
        Parameters:
        text (str): Inquiry text
        
        Returns:
        dict: Prediction results
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Make prediction
        prediction = self.pipeline.predict([processed_text])[0]
        category = self.reverse_mapping[prediction]
        
        # Get confidence if available
        confidence = None
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            proba = self.pipeline.named_steps['classifier'].predict_proba(
                self.pipeline.named_steps['tfidf'].transform([processed_text])
            )[0]
            confidence = np.max(proba)
        elif hasattr(self.pipeline.named_steps['classifier'], 'decision_function'):
            scores = self.pipeline.named_steps['classifier'].decision_function(
                self.pipeline.named_steps['tfidf'].transform([processed_text])
            )[0]
            confidence = np.max(scores)
        
        return {
            'category': category,
            'category_id': prediction,
            'confidence': confidence,
            'original_text': text,
            'processed_text': processed_text
        }
    
    def cross_validate(self, df, cv_folds=5, model_type='svc'):
        """
        Perform cross-validation
        
        Parameters:
        df (pd.DataFrame): Training data
        cv_folds (int): Number of CV folds
        model_type (str): Model type
        
        Returns:
        dict: Cross-validation results
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        # Preprocess text
        df['processed_text'] = df['inquiry_text'].apply(self.preprocess_text)
        
        X = df['processed_text']
        y = df['category_id']
        
        # Create pipeline
        pipeline = self.create_pipeline(model_type)
        
        # Cross-validation
        cv_scores = {
            'accuracy': cross_val_score(pipeline, X, y, cv=cv_folds, scoring='accuracy'),
            'f1_weighted': cross_val_score(pipeline, X, y, cv=cv_folds, scoring='f1_weighted'),
            'precision_weighted': cross_val_score(pipeline, X, y, cv=cv_folds, scoring='precision_weighted'),
            'recall_weighted': cross_val_score(pipeline, X, y, cv=cv_folds, scoring='recall_weighted')
        }
        
        print("\nCROSS-VALIDATION RESULTS:")
        for metric, scores in cv_scores.items():
            print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_scores
    
    def visualize_results(self, results, save_plots=True):
        """
        Create visualizations for model results
        
        Parameters:
        results (dict): Training results
        save_plots (bool): Whether to save plots
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Customer Inquiry Classification Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = results['confusion_matrix']
        category_names = list(self.category_mapping.keys())
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=category_names, yticklabels=category_names)
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted Category')
        axes[0, 0].set_ylabel('Actual Category')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].tick_params(axis='y', rotation=0)
        
        # 2. Category Distribution
        category_counts = pd.Series(results['y_test']).value_counts().sort_index()
        category_labels = [self.reverse_mapping[i] for i in category_counts.index]
        
        axes[0, 1].bar(category_labels, category_counts.values, color='skyblue', alpha=0.7)
        axes[0, 1].set_title('Test Set Category Distribution')
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Per-class Performance
        class_report = results['classification_report']
        categories = [cat for cat in category_names if cat in class_report]
        
        precision_scores = [class_report[cat]['precision'] for cat in categories]
        recall_scores = [class_report[cat]['recall'] for cat in categories]
        f1_scores = [class_report[cat]['f1-score'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.25
        
        axes[1, 0].bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        axes[1, 0].bar(x, recall_scores, width, label='Recall', alpha=0.8)
        axes[1, 0].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Per-Category Performance Metrics')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1.1)
        
        # 4. Model Performance Summary
        overall_metrics = {
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score']
        }
        
        metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in overall_metrics.items()])
        axes[1, 1].text(0.1, 0.5, f"Overall Performance:\n\n{metrics_text}", 
                       fontsize=14, verticalalignment='center',
                       transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        axes[1, 1].set_title('Model Performance Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('customer_inquiry_classification_results.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath='customer_inquiry_classifier.joblib'):
        """
        Save trained model to file
        
        Parameters:
        filepath (str): Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("No trained model to save. Train the model first.")
        
        model_data = {
            'pipeline': self.pipeline,
            'category_mapping': self.category_mapping,
            'reverse_mapping': self.reverse_mapping
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='customer_inquiry_classifier.joblib'):
        """
        Load trained model from file
        
        Parameters:
        filepath (str): Path to load the model from
        """
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.category_mapping = model_data['category_mapping']
        self.reverse_mapping = model_data['reverse_mapping']
        print(f"Model loaded from {filepath}")
    
    def print_detailed_results(self, results):
        """
        Print detailed classification results
        
        Parameters:
        results (dict): Training results
        """
        print("\n" + "="*60)
        print("CUSTOMER INQUIRY CLASSIFICATION RESULTS")
        print("="*60)
        
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Weighted Precision: {results['precision']:.4f}")
        print(f"Weighted Recall: {results['recall']:.4f}")
        print(f"Weighted F1-Score: {results['f1_score']:.4f}")
        
        print("\nPER-CATEGORY PERFORMANCE:")
        class_report = results['classification_report']
        
        for category in self.category_mapping.keys():
            if category in class_report:
                metrics = class_report[category]
                print(f"{category.replace('_', ' ').title()}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1-score']:.4f}")
                print(f"  Support:   {metrics['support']}")
                print()
        
        print("CONFUSION MATRIX:")
        cm = results['confusion_matrix']
        category_names = list(self.category_mapping.keys())
        
        print("\nPredicted ->")
        print("Actual ↓    ", end="")
        for name in category_names:
            print(f"{name[:8]:>8}", end="")
        print()
        
        for i, actual in enumerate(category_names):
            print(f"{actual[:12]:12}", end="")
            for j in range(len(category_names)):
                print(f"{cm[i][j]:8d}", end="")
            print()

def demo_customer_inquiry_classification():
    """
    Demonstrate the customer inquiry classification system
    """
    print("="*60)
    print("CUSTOMER INQUIRY CLASSIFICATION DEMO")
    print("="*60)
    
    # Initialize classifier
    classifier = CustomerInquiryClassifier()
    
    # Generate synthetic data
    print("\nGenerating synthetic customer inquiry data...")
    df = classifier.generate_synthetic_data(n_samples=2800)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Categories: {list(classifier.category_mapping.keys())}")
    print(f"Category distribution:")
    print(df['category'].value_counts())
    
    # Train multiple models for comparison
    models = ['svc', 'logistic', 'nb']
    best_model = None
    best_score = 0
    
    for model_type in models:
        print(f"\n{'='*40}")
        print(f"Training {model_type.upper()} Model")
        print('='*40)
        
        results = classifier.train_model(df, model_type=model_type)
        
        if results['accuracy'] > best_score:
            best_score = results['accuracy']
            best_model = model_type
            best_results = results
    
    print(f"\nBest model: {best_model.upper()} with accuracy: {best_score:.4f}")
    
    # Cross-validation on best model
    print(f"\nPerforming cross-validation on {best_model.upper()} model...")
    cv_results = classifier.cross_validate(df, model_type=best_model)
    
    # Print detailed results
    classifier.print_detailed_results(best_results)
    
    # Generate visualizations
    print("\nGenerating result visualizations...")
    classifier.visualize_results(best_results)
    
    # Test predictions on sample inquiries
    print("\n" + "="*60)
    print("TESTING PREDICTIONS ON SAMPLE INQUIRIES")
    print("="*60)
    
    test_inquiries = [
        "I need help with my monthly bill charges",
        "My app keeps crashing when I try to login",
        "Can you tell me more about your product warranty",
        "When will my order arrive? I ordered it last week",
        "I want to return this item and get my money back",
        "How do I update my profile information",
        "What are your customer service hours"
    ]
    
    for inquiry in test_inquiries:
        prediction = classifier.predict_inquiry(inquiry)
        print(f"\nInquiry: '{inquiry}'")
        print(f"Predicted Category: {prediction['category'].replace('_', ' ').title()}")
        if prediction['confidence'] is not None:
            print(f"Confidence: {prediction['confidence']:.3f}")
    
    # Save the model
    print(f"\nSaving trained model...")
    classifier.save_model('customer_inquiry_classifier.joblib')
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print(f"✓ Best Model Accuracy: {best_score:.1%}")
    print(f"✓ Model Type: {best_model.upper()}")
    print("✓ Model saved for future use")
    print("✓ Ready for deployment!")
    print("="*60)
    
    return classifier, best_results

if __name__ == "__main__":
    # Run the demo
    classifier, results = demo_customer_inquiry_classification()