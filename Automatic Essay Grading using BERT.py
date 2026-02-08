import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import torch and transformers with error handling
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    from torch.utils.data import Dataset, DataLoader
    BERT_AVAILABLE = True
    print("BERT and PyTorch libraries loaded successfully")
except ImportError as e:
    print(f"Warning: {e}")
    print("Using alternative TF-IDF approach instead of BERT")
    BERT_AVAILABLE = False

# Load dataset
df = pd.read_csv('d:/python_ka_chilla/AI Projects/Automatic Essay Grading using BERT/ASAP.csv')

# Display first few rows of the dataset
print("Essay Grading DataSet:")
print(df.head())

# Check available columns
print(f"\nAvailable columns: {df.columns.tolist()}")

# Select relevant columns - USING YOUR ACTUAL COLUMN NAMES
df = df[['full_text', 'score']]  # Changed from ['Essay', 'OverallScore']
df = df.rename(columns={'full_text': 'Essay', 'score': 'OverallScore'})  # Rename for compatibility

print(f"\nAfter selecting columns: {df.shape}")
print(df.head())

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Remove rows with missing values
df = df.dropna()
print(f"\nAfter removing missing values: {df.shape}")

# Display score statistics
print("\nScore Statistics:")
print(f"Minimum score: {df['OverallScore'].min()}")
print(f"Maximum score: {df['OverallScore'].max()}")
print(f"Average score: {df['OverallScore'].mean():.2f}")
print(f"Standard deviation: {df['OverallScore'].std():.2f}")

# Visualize score distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['OverallScore'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.boxplot(df['OverallScore'])
plt.title('Score Box Plot')
plt.ylabel('Score')
plt.tight_layout()
plt.show()

if BERT_AVAILABLE:
    # =============================================
    # BERT APPROACH (if libraries are available)
    # =============================================
    print("\n" + "="*60)
    print("USING BERT FOR ESSAY GRADING")
    print("="*60)
    
    try:
        # Basic data preprocessing
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Custom dataset class
        class EssayDataset(Dataset):
            def __init__(self, essays, scores, tokenizer, max_length=256):  # Reduced from 512
                self.essays = essays
                self.scores = scores
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.essays)

            def __getitem__(self, idx):
                essay = str(self.essays[idx])
                score = self.scores[idx]
                encoding = self.tokenizer.encode_plus(
                    essay,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                return {
                    'essay_text': essay,
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'score': torch.tensor(score, dtype=torch.float)
                }
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['Essay'], df['OverallScore'], test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        train_dataset = EssayDataset(X_train.tolist(), y_train.tolist(), tokenizer)
        test_dataset = EssayDataset(X_test.tolist(), y_test.tolist(), tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reduced batch size
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        # Build the model
        class EssayGradingModel(torch.nn.Module):
            def __init__(self):
                super(EssayGradingModel, self).__init__()
                self.bert = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased', 
                    num_labels=1,
                    output_hidden_states=True  # We need hidden states for our custom head
                )
                self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)
                
            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                # Get the hidden state of the [CLS] token (first token)
                cls_output = outputs.hidden_states[-1][:, 0, :]
                score = self.regressor(cls_output)
                return score.squeeze()
        
        # Initialize the model, optimizer, and loss function
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = EssayGradingModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        def train_model(model, data_loader, optimizer, loss_fn, device):
            model.train()
            total_loss = 0
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                scores = batch['score'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, scores)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(data_loader)

        # Evaluation loop
        def eval_model(model, data_loader, loss_fn, device):
            model.eval()
            total_loss = 0
            predictions = []
            actuals = []
            with torch.no_grad():
                for batch in data_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    scores = batch['score'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = loss_fn(outputs, scores)
                    total_loss += loss.item()
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(scores.cpu().numpy())
            return total_loss / len(data_loader), predictions, actuals

        # Train the model
        epochs = 3
        print(f"\nTraining for {epochs} epochs...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_predictions, val_actuals = eval_model(model, test_loader, loss_fn, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        # Test the model
        test_text = "This is an example essay text to be graded. It discusses the importance of education in modern society."
        encoding = tokenizer.encode_plus(
            test_text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        model.eval()
        with torch.no_grad():
            predicted_score = model(input_ids, attention_mask)
        print(f'\nPredicted Score for test essay: {predicted_score.item():.2f}')

        # Evaluate the model
        test_loss, test_predictions, test_actuals = eval_model(model, test_loader, loss_fn, device)
        print("\nModel Evaluation Results:")
        print(f"Test Loss: {test_loss:.4f}")
        
        mse = mean_squared_error(test_actuals, test_predictions)
        mae = mean_absolute_error(test_actuals, test_predictions)
        r2 = r2_score(test_actuals, test_predictions)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")

        # Visualization of training loss
        plt.figure(figsize=(10,6))
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Scatter plot of predictions vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(test_actuals, test_predictions, alpha=0.7)
        plt.plot([min(test_actuals), max(test_actuals)], 
                 [min(test_actuals), max(test_actuals)], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Scores')
        plt.ylabel('Predicted Scores')
        plt.title('Actual vs Predicted Scores (BERT Model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"Error in BERT model: {e}")
        print("Falling back to TF-IDF approach...")
        BERT_AVAILABLE = False

if not BERT_AVAILABLE:
    # =============================================
    # TF-IDF APPROACH (fallback method)
    # =============================================
    print("\n" + "="*60)
    print("USING TF-IDF FOR ESSAY GRADING")
    print("="*60)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    import re
    
    # Text preprocessing function
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Preprocess essays
    print("Preprocessing essays...")
    df['cleaned_essay'] = df['Essay'].apply(preprocess_text)
    
    # Feature extraction
    def extract_features(essays):
        features = []
        for essay in essays:
            words = essay.split()
            sentences = [s for s in essay.split('.') if s.strip()]
            
            word_count = len(words)
            char_count = len(essay)
            sentence_count = len(sentences)
            
            avg_word_length = char_count / max(word_count, 1)
            avg_sentence_length = word_count / max(sentence_count, 1)
            unique_words = len(set(words))
            lexical_diversity = unique_words / max(word_count, 1)
            
            features.append({
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'lexical_diversity': lexical_diversity
            })
        return pd.DataFrame(features)
    
    # Extract features
    print("Extracting text features...")
    features_df = extract_features(df['cleaned_essay'])
    
    # TF-IDF Vectorization
    print("Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_essay'])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    )
    
    # Combine features
    X = pd.concat([features_df, tfidf_df], axis=1)
    y = df['OverallScore']
    
    print(f"\nTotal features: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        })
        
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Select best model
    best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name}")
    
    # Visualize predictions
    y_pred_best = best_model.predict(X_test)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_best, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title(f'Actual vs Predicted ({best_model_name})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    errors = y_test - y_pred_best
    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Function to grade new essays
    def grade_essay_tfidf(essay_text, model=best_model, vectorizer=vectorizer):
        cleaned_essay = preprocess_text(essay_text)
        
        # Extract features
        features = extract_features([cleaned_essay])
        
        # Apply TF-IDF
        tfidf_features = vectorizer.transform([cleaned_essay]).toarray()
        tfidf_df = pd.DataFrame(
            tfidf_features,
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Combine features
        all_features = pd.concat([features, tfidf_df], axis=1)
        
        # Ensure all columns are present
        missing_cols = set(X.columns) - set(all_features.columns)
        for col in missing_cols:
            all_features[col] = 0
        
        # Reorder columns
        all_features = all_features[X.columns]
        
        # Predict score
        predicted_score = model.predict(all_features)[0]
        
        # Clip to reasonable range
        min_score, max_score = y.min(), y.max()
        predicted_score = max(min_score, min(max_score, predicted_score))
        
        return {
            'score': round(predicted_score, 2),
            'word_count': features['word_count'].iloc[0],
            'sentence_count': features['sentence_count'].iloc[0]
        }
    
    # Test grading
    test_essay = "This is an example essay text to be graded. It discusses the importance of education in modern society."
    result = grade_essay_tfidf(test_essay)
    print(f"\nTF-IDF Model Prediction:")
    print(f"Essay: {test_essay[:100]}...")
    print(f"Predicted Score: {result['score']:.2f}")
    print(f"Word Count: {result['word_count']}")
    print(f"Sentence Count: {result['sentence_count']}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)