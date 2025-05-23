import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re

def calculate_bleu(predictions: List[str], references: List[str], max_n: int = 4) -> float:
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        if len(tokens) < n:
            return Counter()
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    
    def calculate_bleu_score(pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
        scores = {}
        
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            if len(pred_ngrams) == 0:
                scores[f'bleu_{n}'] = 0.0
            else:
                overlap = sum(min(pred_ngrams[ngram], ref_ngrams[ngram]) for ngram in pred_ngrams)
                scores[f'bleu_{n}'] = overlap / len(pred_ngrams)
        
        return scores
    
    all_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        scores = calculate_bleu_score(pred_tokens, ref_tokens)
        all_scores.append(scores)
    
    bleu_4_scores = [score.get('bleu_4', 0.0) for score in all_scores]
    return np.mean(bleu_4_scores) * 100

def calculate_rouge_l(predictions: List[str], references: List[str]) -> float:
    def lcs_length(seq1: List[str], seq2: List[str]) -> int:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    rouge_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            rouge_scores.append(0.0)
        else:
            precision = lcs_len / len(pred_tokens)
            recall = lcs_len / len(ref_tokens)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            rouge_scores.append(f1)
    
    return np.mean(rouge_scores) * 100

def calculate_content_preservation(predictions: List[str], references: List[str]) -> float:
    stopwords = {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'à', 'au', 'aux',
        'et', 'ou', 'mais', 'car', 'ni', 'or', 'donc', 'que', 'qui', 'quoi',
        'ce', 'cette', 'ces', 'cet', 'il', 'elle', 'ils', 'elles', 'je', 'tu',
        'nous', 'vous', 'me', 'te', 'se', 'nous', 'vous', 'leur', 'leurs',
        'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre',
        'votre', 'dans', 'sur', 'sous', 'avec', 'sans', 'pour', 'par', 'vers',
        'chez', 'entre', 'depuis', 'pendant', 'avant', 'après', 'être', 'avoir',
        'est', 'sont', 'était', 'étaient', 'a', 'ont', 'avait', 'avaient'} # very crude setup, maybe swap for nltk+textstat?
    
    preservation_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(word.lower() for word in pred.split() if word.lower() not in stopwords)
        ref_tokens = set(word.lower() for word in ref.split() if word.lower() not in stopwords)
        
        if len(ref_tokens) == 0:
            preservation_scores.append(1.0)
        else:
            overlap = len(pred_tokens.intersection(ref_tokens))
            preservation_scores.append(overlap / len(ref_tokens))
    
    return np.mean(preservation_scores) * 100

def calculate_functional_word_accuracy(predictions: List[str], references: List[str]) -> float:
  
    functional_words = {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'à', 'au', 'aux',
        'et', 'ou', 'mais', 'car', 'ni', 'or', 'donc', 'dans', 'sur', 'sous',
        'avec', 'sans', 'pour', 'par', 'vers', 'chez', 'entre', 'depuis',
        'pendant', 'avant', 'après', 'est', 'sont', 'était', 'étaient',
        'a', 'ont', 'avait', 'avaient'}
    
    accuracy_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_functional = [word.lower() for word in pred.split() if word.lower() in functional_words]
        ref_functional = [word.lower() for word in ref.split() if word.lower() in functional_words]
        
        if len(ref_functional) == 0:
            accuracy_scores.append(1.0)
        else:
            pred_counter = Counter(pred_functional)
            ref_counter = Counter(ref_functional)
            
            correct = sum(min(pred_counter[word], ref_counter[word]) for word in ref_counter)
            accuracy_scores.append(correct / len(ref_functional))
    
    return np.mean(accuracy_scores) * 100


#standard NMT evaluation, more or less
def evaluate_translations(predictions: List[str], references: List[str]) -> Dict[str, float]:
    metrics = {
        'bleu': calculate_bleu(predictions, references),
        'rouge_l': calculate_rouge_l(predictions, references),
        'content_preservation': calculate_content_preservation(predictions, references),
        'functional_word_accuracy': calculate_functional_word_accuracy(predictions, references)
    }
    
    return metrics

def calculate_schema_alignment_score(predictions: List[str], schemas: List[Dict]) -> float:
    alignment_scores = []
    
    for pred, schema in zip(predictions, schemas):
        pred_tokens = pred.lower().split()
        expected_structure = schema.get('structure_type', 0)
        has_determiner = any(word in ['le', 'la', 'les', 'un', 'une', 'des'] for word in pred_tokens)
        has_verb = any(word in ['est', 'sont', 'a', 'ont', 'va', 'vont'] for word in pred_tokens)
        has_preposition = any(word in ['à', 'de', 'dans', 'sur', 'avec'] for word in pred_tokens)
        
        # linguistics again, need to approximate an justify the methods here
        structure_score = 0.0
        if expected_structure == 0:  # Simple SV
            if has_determiner and has_verb:
                structure_score = 1.0
        elif expected_structure == 1:  # SVO
            if has_determiner and has_verb:
                structure_score = 0.8
                if len(pred_tokens) >= 4:  # Likely has object
                    structure_score = 1.0
        elif expected_structure in [2, 3]:  # Complex structures
            if has_determiner and has_verb and has_preposition:
                structure_score = 1.0
            elif has_determiner and has_verb:
                structure_score = 0.7
        
        alignment_scores.append(structure_score)
    
    return np.mean(alignment_scores) * 100