#!/usr/bin/env python3
"""
Rule-Augmented Machine Learning for Hate Speech Detection
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class HateSpeechRules:
    """
    Collection of human-written rules for hate speech detection
    """
    
    def __init__(self):
        """
        Initialize hate speech detection rules
        """
        # Hate speech keywords and patterns
        self.hate_keywords = {
            'racial_slurs': [
                r'\b(n[i1]gg[ae]r|f[a@]gg[o0]t|k[i1]k[e3]|sp[i1]c|ch[i1]nk|g[o0]0k|t[o0]w[e3]lhead|s[a@]ndn[i1]gg[e3]r)\b',
                r'\b(w[e3]tback|b[e3]an[e3]r|t[a@]c0|g[rr]e[a@]s[e3]r|d[i1]ng[e3]r|r[a@]ghead)\b'
            ],
            'gender_hate': [
                r'\b(b[i1]tch|wh[o0]r[e3]|sl[u1]t|c[u1]nt|p[u1]ssy|d[i1]ck|p[e3]n[i1]s)\b',
                r'\b(f[a@]gg[o0]t|d[yy]k[e3]|l[e3]sb[i1][a@]n|tr[a@]nn[yy]|sh[e3]m[a@]l[e3])\b'
            ],
            'religious_hate': [
                r'\b(m[u1]sl[i1]m|j[e3]w|h[i1]nd[u1]|b[u1]ddh[i1]st|chr[i1]st[i1][a@]n)\b',
                r'\b(t[e3]rr[o0]r[i1]st|j[i1]h[a@]d[i1]st|f[u1]nd[a@]m[e3]nt[a@]l[i1]st)\b'
            ],
            'disability_hate': [
                r'\b(r[e3]t[a@]rd|m[o0]r[o0]n|i[d]i[o0]t|d[u1]mb|cr[a@]zy|n[u1]ts|w[e3]ird[o0])\b',
                r'\b(sp[e3]c[i1][a@]l|d[e3]f[e3]ct[i1]v[e3]|br[o0]k[e3]n|d[a@]m[a@]g[e3]d)\b'
            ],
            'sexual_orientation': [
                r'\b(f[a@]gg[o0]t|d[yy]k[e3]|l[e3]sb[i1][a@]n|g[a@]y|h[o0]m[o0]|qu[e3][e3]r)\b',
                r'\b(tr[a@]nn[yy]|sh[e3]m[a@]l[e3]|h[e3]t[e3]r[o0]|str[a1]ght)\b'
            ]
        }
        
        # Threatening patterns
        self.threat_patterns = [
            r'\b(k[i1]ll|m[u1]rd[e3]r|d[i1][e3]|b[e3][a@]t|h[i1]t|p[u1]nch|str[i1]k[e3])\b',
            r'\b(b[u1]rn|b[u1]r[yy]|d[e3]str[o0]y|cr[a@]sh|sm[a@]sh|br[e3][a@]k)\b',
            r'\b(g[e3]t|f[i1]nd|c[a@]tch|gr[a@]b|h[o0]ld|k[e3][e3]p)\b.*\b(y[o0]u|u|y[a@])\b',
            r'\b(w[i1]sh|h[o0]p[e3]|pr[a@]y|w[a@]nt)\b.*\b(d[i1][e3]|d[e3][a@]th|g[o0]n[e3])\b'
        ]
        
        # Dehumanizing patterns
        self.dehumanizing_patterns = [
            r'\b(tr[a@]sh|g[a@]rb[a@]g[e3]|w[a@]st[e3]|sc[u1]m|v[e3]rm[i1]n|r[a@]t)\b',
            r'\b(an[i1]m[a@]l|b[e3][a@]st|cr[e3][a@]t[u1]r[e3]|m[o0]nst[e3]r|d[e3]v[i1]l)\b',
            r'\b(n[o0]t.*h[u1]m[a@]n|sub.*h[u1]m[a@]n|b[e3]l[o0]w.*h[u1]m[a@]n)\b'
        ]
        
        # Intensity modifiers
        self.intensity_modifiers = [
            r'\b(v[e3]ry|r[e3][a@]lly|extr[e3]m[e3]ly|t[o0]t[a@]lly|c[o0]mpl[e3]t[e3]ly)\b',
            r'\b(absl[u1]t[e3]ly|p[e3]rf[e3]ctly|wh[o0]lly|ent[i1]r[e3]ly|f[u1]lly)\b'
        ]
        
        # Obfuscation patterns (leetspeak, misspellings)
        self.obfuscation_patterns = {
            'number_substitutions': {
                'a': ['4', '@'], 'e': ['3'], 'i': ['1', '!'], 'o': ['0'], 's': ['5', '$'],
                't': ['7'], 'g': ['9'], 'l': ['1'], 'b': ['8'], 'z': ['2']
            },
            'common_misspellings': [
                r'\b(h[a@]t[e3]|h[a@]t[e3]d|h[a@]t[i1]ng)\b',  # hate variations
                r'\b(k[i1]ll[e3]d|k[i1]ll[i1]ng|k[i1]ll[e3]r)\b',  # kill variations
                r'\b(d[i1][e3]d|d[i1][e3]s|d[i1][e3]d)\b'  # die variations
            ]
        }
        
    def detect_hate_keywords(self, text: str) -> Dict[str, int]:
        """
        Detect hate speech keywords in text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with category counts
        """
        text_lower = text.lower()
        results = {}
        
        for category, patterns in self.hate_keywords.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                count += len(matches)
            results[category] = count
            
        return results
    
    def detect_threats(self, text: str) -> int:
        """
        Detect threatening language
        
        Args:
            text: Input text
            
        Returns:
            Number of threat patterns found
        """
        text_lower = text.lower()
        threat_count = 0
        
        for pattern in self.threat_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            threat_count += len(matches)
            
        return threat_count
    
    def detect_dehumanization(self, text: str) -> int:
        """
        Detect dehumanizing language
        
        Args:
            text: Input text
            
        Returns:
            Number of dehumanizing patterns found
        """
        text_lower = text.lower()
        dehumanizing_count = 0
        
        for pattern in self.dehumanizing_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            dehumanizing_count += len(matches)
            
        return dehumanizing_count
    
    def detect_intensity(self, text: str) -> int:
        """
        Detect intensity modifiers
        
        Args:
            text: Input text
            
        Returns:
            Number of intensity modifiers found
        """
        text_lower = text.lower()
        intensity_count = 0
        
        for pattern in self.intensity_modifiers:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            intensity_count += len(matches)
            
        return intensity_count
    
    def detect_obfuscation(self, text: str) -> Dict[str, int]:
        """
        Detect obfuscation techniques
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with obfuscation counts
        """
        text_lower = text.lower()
        results = {}
        
        # Number substitutions
        substitution_count = 0
        for char, substitutes in self.obfuscation_patterns['number_substitutions'].items():
            for substitute in substitutes:
                if substitute in text_lower:
                    substitution_count += text_lower.count(substitute)
        results['number_substitutions'] = substitution_count
        
        # Misspellings
        misspelling_count = 0
        for pattern in self.obfuscation_patterns['common_misspellings']:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            misspelling_count += len(matches)
        results['misspellings'] = misspelling_count
        
        return results
    
    def apply_all_rules(self, text: str) -> Dict[str, any]:
        """
        Apply all rules to text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all rule results
        """
        results = {
            'hate_keywords': self.detect_hate_keywords(text),
            'threats': self.detect_threats(text),
            'dehumanization': self.detect_dehumanization(text),
            'intensity': self.detect_intensity(text),
            'obfuscation': self.detect_obfuscation(text)
        }
        
        # Calculate total hate score
        total_hate_keywords = sum(results['hate_keywords'].values())
        total_obfuscation = sum(results['obfuscation'].values())
        
        # Weighted score
        hate_score = (
            total_hate_keywords * 2.0 +
            results['threats'] * 1.5 +
            results['dehumanization'] * 1.5 +
            results['intensity'] * 0.5 +
            total_obfuscation * 0.3
        )
        
        results['hate_score'] = hate_score
        results['is_hate_speech'] = hate_score > 1.0  # Threshold
        
        return results


class RuleAugmentedML:
    """
    Rule-augmented machine learning system
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize rule-augmented ML system
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.is_trained = False
        
        # Rule engine
        self.rules = HateSpeechRules()
        
        # ML models
        self.rule_model = None  # Model trained on rule features
        self.hybrid_model = None  # Model combining rules and ML features
        self.label_model = None  # Model for combining weak labels
        
        # Results storage
        self.results = {}
        
    def _extract_rule_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from rule application
        
        Args:
            texts: List of text strings
            
        Returns:
            Rule features matrix
        """
        print("Extracting rule-based features...")
        
        features = []
        
        for text in texts:
            rule_results = self.rules.apply_all_rules(text)
            
            # Extract numerical features
            text_features = [
                rule_results['hate_score'],
                rule_results['threats'],
                rule_results['dehumanization'],
                rule_results['intensity'],
                rule_results['obfuscation']['number_substitutions'],
                rule_results['obfuscation']['misspellings'],
                sum(rule_results['hate_keywords'].values()),
                rule_results['hate_keywords']['racial_slurs'],
                rule_results['hate_keywords']['gender_hate'],
                rule_results['hate_keywords']['religious_hate'],
                rule_results['hate_keywords']['disability_hate'],
                rule_results['hate_keywords']['sexual_orientation'],
                int(rule_results['is_hate_speech'])
            ]
            
            features.append(text_features)
        
        features = np.array(features)
        print(f"✓ Extracted {features.shape[1]} rule features")
        
        return features
    
    def _create_weak_labels(self, texts: List[str]) -> np.ndarray:
        """
        Create weak labels using rules
        
        Args:
            texts: List of text strings
            
        Returns:
            Weak labels array
        """
        print("Creating weak labels from rules...")
        
        weak_labels = []
        
        for text in texts:
            rule_results = self.rules.apply_all_rules(text)
            weak_labels.append(int(rule_results['is_hate_speech']))
        
        weak_labels = np.array(weak_labels)
        print(f"✓ Created {len(weak_labels)} weak labels")
        
        return weak_labels
    
    def train(self, texts: List[str], y: np.ndarray = None,
             ml_features: np.ndarray = None) -> Dict[str, float]:
        """
        Train rule-augmented ML system
        
        Args:
            texts: List of text strings
            y: True labels (optional for semi-supervised)
            ml_features: ML features (optional)
            
        Returns:
            Training results
        """
        print("=" * 60)
        print("TRAINING RULE-AUGMENTED ML SYSTEM")
        print("=" * 60)
        
        # Extract rule features
        rule_features = self._extract_rule_features(texts)
        
        # Create weak labels
        weak_labels = self._create_weak_labels(texts)
        
        # Train rule-based model
        print("\nTraining rule-based model...")
        self.rule_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        
        # Use weak labels for training if no true labels
        training_labels = y if y is not None else weak_labels
        self.rule_model.fit(rule_features, training_labels)
        
        # Train hybrid model if ML features provided
        if ml_features is not None:
            print("\nTraining hybrid model...")
            
            # Combine rule and ML features
            hybrid_features = np.hstack([rule_features, ml_features])
            
            self.hybrid_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=self.random_state
            )
            
            self.hybrid_model.fit(hybrid_features, training_labels)
        
        # Train label model for combining weak labels
        print("\nTraining label model...")
        self.label_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        
        # Create multiple weak label sources
        weak_label_sources = []
        
        # Source 1: Rule-based hate score threshold
        hate_scores = rule_features[:, 0]
        weak_label_sources.append((hate_scores > 1.0).astype(int))
        
        # Source 2: Threat detection
        threat_counts = rule_features[:, 1]
        weak_label_sources.append((threat_counts > 0).astype(int))
        
        # Source 3: Dehumanization detection
        dehumanization_counts = rule_features[:, 2]
        weak_label_sources.append((dehumanization_counts > 0).astype(int))
        
        # Source 4: High intensity
        intensity_counts = rule_features[:, 3]
        weak_label_sources.append((intensity_counts > 1).astype(int))
        
        # Combine weak label sources
        weak_label_matrix = np.column_stack(weak_label_sources)
        
        # Train label model
        if y is not None:
            self.label_model.fit(weak_label_matrix, y)
        else:
            # Use majority vote as proxy
            majority_vote = (np.mean(weak_label_matrix, axis=1) > 0.5).astype(int)
            self.label_model.fit(weak_label_matrix, majority_vote)
        
        # Evaluate if true labels available
        if y is not None:
            # Rule model evaluation
            rule_predictions = self.rule_model.predict(rule_features)
            rule_accuracy = accuracy_score(y, rule_predictions)
            rule_f1 = f1_score(y, rule_predictions)
            
            # Hybrid model evaluation
            if ml_features is not None:
                hybrid_predictions = self.hybrid_model.predict(hybrid_features)
                hybrid_accuracy = accuracy_score(y, hybrid_predictions)
                hybrid_f1 = f1_score(y, hybrid_predictions)
            else:
                hybrid_accuracy = hybrid_f1 = 0.0
            
            # Label model evaluation
            label_predictions = self.label_model.predict(weak_label_matrix)
            label_accuracy = accuracy_score(y, label_predictions)
            label_f1 = f1_score(y, label_predictions)
            
            # Store results
            self.results = {
                'rule_accuracy': rule_accuracy,
                'rule_f1': rule_f1,
                'hybrid_accuracy': hybrid_accuracy,
                'hybrid_f1': hybrid_f1,
                'label_accuracy': label_accuracy,
                'label_f1': label_f1
            }
            
            print(f"\nTraining Results:")
            print(f"  Rule Model - Accuracy: {rule_accuracy:.4f}, F1: {rule_f1:.4f}")
            print(f"  Hybrid Model - Accuracy: {hybrid_accuracy:.4f}, F1: {hybrid_f1:.4f}")
            print(f"  Label Model - Accuracy: {label_accuracy:.4f}, F1: {label_f1:.4f}")
        
        self.is_trained = True
        
        return self.results
    
    def predict(self, texts: List[str], ml_features: np.ndarray = None) -> np.ndarray:
        """
        Make predictions using rule-augmented ML
        
        Args:
            texts: List of text strings
            ml_features: ML features (optional)
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract rule features
        rule_features = self._extract_rule_features(texts)
        
        # Use hybrid model if available and ML features provided
        if self.hybrid_model is not None and ml_features is not None:
            hybrid_features = np.hstack([rule_features, ml_features])
            predictions = self.hybrid_model.predict(hybrid_features)
        else:
            # Use rule model
            predictions = self.rule_model.predict(rule_features)
        
        return predictions
    
    def predict_proba(self, texts: List[str], ml_features: np.ndarray = None) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            texts: List of text strings
            ml_features: ML features (optional)
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract rule features
        rule_features = self._extract_rule_features(texts)
        
        # Use hybrid model if available and ML features provided
        if self.hybrid_model is not None and ml_features is not None:
            hybrid_features = np.hstack([rule_features, ml_features])
            probabilities = self.hybrid_model.predict_proba(hybrid_features)
        else:
            # Use rule model
            probabilities = self.rule_model.predict_proba(rule_features)
        
        return probabilities
    
    def get_rule_confidence(self, texts: List[str]) -> np.ndarray:
        """
        Get confidence scores from rules
        
        Args:
            texts: List of text strings
            
        Returns:
            Rule confidence scores
        """
        rule_features = self._extract_rule_features(texts)
        
        # Use hate score as confidence
        confidence_scores = rule_features[:, 0]  # hate_score column
        
        # Normalize to [0, 1]
        confidence_scores = np.clip(confidence_scores / 5.0, 0, 1)
        
        return confidence_scores
    
    def analyze_rules(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze rule application results
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with rule analysis
        """
        print("Analyzing rule application...")
        
        rule_results = []
        
        for i, text in enumerate(texts):
            results = self.rules.apply_all_rules(text)
            
            row = {
                'text_id': i,
                'text_length': len(text),
                'hate_score': results['hate_score'],
                'threats': results['threats'],
                'dehumanization': results['dehumanization'],
                'intensity': results['intensity'],
                'obfuscation_substitutions': results['obfuscation']['number_substitutions'],
                'obfuscation_misspellings': results['obfuscation']['misspellings'],
                'total_hate_keywords': sum(results['hate_keywords'].values()),
                'racial_slurs': results['hate_keywords']['racial_slurs'],
                'gender_hate': results['hate_keywords']['gender_hate'],
                'religious_hate': results['hate_keywords']['religious_hate'],
                'disability_hate': results['hate_keywords']['disability_hate'],
                'sexual_orientation': results['hate_keywords']['sexual_orientation'],
                'rule_prediction': int(results['is_hate_speech'])
            }
            
            rule_results.append(row)
        
        df = pd.DataFrame(rule_results)
        
        print(f"✓ Analyzed {len(texts)} texts with rules")
        return df
    
    def evaluate(self, texts: List[str], y: np.ndarray,
                ml_features: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate the rule-augmented ML system
        
        Args:
            texts: List of text strings
            y: True labels
            ml_features: ML features (optional)
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        predictions = self.predict(texts, ml_features)
        probabilities = self.predict_proba(texts, ml_features)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        auc = roc_auc_score(y, probabilities)
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return results


def create_rule_augmented_ml(texts: List[str], y: np.ndarray = None,
                            ml_features: np.ndarray = None,
                            random_state: int = 42) -> RuleAugmentedML:
    """
    Convenience function to create and train a rule-augmented ML system
    
    Args:
        texts: List of text strings
        y: True labels (optional for semi-supervised)
        ml_features: ML features (optional)
        random_state: Random seed
        
    Returns:
        Trained RuleAugmentedML instance
    """
    print("=" * 60)
    print("CREATING RULE-AUGMENTED ML SYSTEM")
    print("=" * 60)
    
    # Initialize system
    system = RuleAugmentedML(random_state=random_state)
    
    # Train system
    training_results = system.train(texts, y, ml_features)
    
    print("\n" + "=" * 60)
    print("RULE-AUGMENTED ML SYSTEM CREATED!")
    print("=" * 60)
    
    return system


if __name__ == "__main__":
    # Example usage
    print("Rule-Augmented ML Example")
    print("=" * 50)
    
    # Create dummy text data
    np.random.seed(42)
    n_samples = 100
    
    # Dummy texts with various hate speech patterns
    dummy_texts = [
        "I hate all of you and wish you would die",
        "This is a wonderful community",
        "You're all idiots and should be killed",
        "Thanks for the helpful information",
        "I despise everything about this place",
        "What a fantastic experience",
        "You're all trash and don't deserve to live",
        "Great discussion, very informative",
        "I want to burn this place down",
        "This is amazing, thank you everyone"
    ] * (n_samples // 10)
    
    # Add some obfuscated texts
    obfuscated_texts = [
        "I h4t3 4ll 0f y0u",
        "Y0u'r3 4ll 1d10ts",
        "K1ll 3v3ry0n3",
        "Th1s 1s gr34t",
        "L0v3 th1s pl4c3"
    ] * (n_samples // 20)
    
    dummy_texts.extend(obfuscated_texts)
    
    # Dummy labels
    dummy_labels = np.array([1 if any(word in text.lower() for word in ["hate", "kill", "die", "trash", "idiot"]) else 0 
                            for text in dummy_texts])
    
    # Dummy ML features
    dummy_ml_features = np.random.rand(len(dummy_texts), 50)
    
    # Create and train system
    system = create_rule_augmented_ml(dummy_texts, dummy_labels, dummy_ml_features)
    
    # Make predictions
    predictions = system.predict(dummy_texts, dummy_ml_features)
    probabilities = system.predict_proba(dummy_texts, dummy_ml_features)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Analyze rules
    rule_analysis = system.analyze_rules(dummy_texts)
    print(f"\nRule analysis summary:")
    print(rule_analysis.describe())
    
    # Get rule confidence
    confidence_scores = system.get_rule_confidence(dummy_texts)
    print(f"\nRule confidence scores - Mean: {np.mean(confidence_scores):.4f}, Std: {np.std(confidence_scores):.4f}")
    
    # Evaluate
    evaluation_results = system.evaluate(dummy_texts, dummy_labels, dummy_ml_features) 