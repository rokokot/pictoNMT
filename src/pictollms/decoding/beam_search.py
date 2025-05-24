# src/pictollms/decoding/beam_search.py - UPDATED VERSION
import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

class SearchStrategy(ABC):
    """Base class for search strategies"""
    
    def __init__(self, max_length: int = 100):
        self.max_length = max_length
    
    @abstractmethod
    def search(self, decoder, encoder_outputs: torch.Tensor, 
              attention_mask: Optional[torch.Tensor] = None, 
              tokenizer=None, **kwargs) -> List[List[int]]:
        pass

class GreedySearch(SearchStrategy):
    """Simple greedy decoding"""
    
    def search(self, decoder, encoder_outputs: torch.Tensor, 
              attention_mask: Optional[torch.Tensor] = None, 
              tokenizer=None, **kwargs) -> List[List[int]]:
        
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Start with BOS token
        sequences = torch.full((batch_size, 1), tokenizer.bos_token_id, 
                              dtype=torch.long, device=device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(self.max_length):
            if finished.all():
                break
            
            # Get next token logits
            logits = decoder.generate_step(
                input_ids=sequences,
                encoder_outputs=encoder_outputs,
                encoder_mask=attention_mask
            )
            
            # Greedy selection
            next_tokens = torch.argmax(logits, dim=-1)
            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for EOS
            finished = finished | (next_tokens == tokenizer.eos_token_id)
        
        return [seq.tolist() for seq in sequences]

class BeamSearch(SearchStrategy):
    """Standard beam search"""
    
    def __init__(self, beam_size: int = 4, max_length: int = 100, length_penalty: float = 0.6):
        super().__init__(max_length)
        self.beam_size = beam_size
        self.length_penalty = length_penalty
    
    def search(self, decoder, encoder_outputs: torch.Tensor, 
              attention_mask: Optional[torch.Tensor] = None, 
              tokenizer=None, **kwargs) -> List[List[int]]:
        
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        results = []
        
        # Process each batch item separately
        for batch_idx in range(batch_size):
            # Get encoder outputs for this batch item
            batch_encoder = encoder_outputs[batch_idx:batch_idx+1]
            batch_mask = attention_mask[batch_idx:batch_idx+1] if attention_mask is not None else None
            
            # Initialize beam
            sequences = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
            scores = torch.zeros(1, device=device)
            finished_sequences = []
            
            for step in range(self.max_length):
                if len(finished_sequences) >= self.beam_size:
                    break
                
                # Expand sequences for beam size
                current_beam_size = sequences.size(0)
                expanded_encoder = batch_encoder.expand(current_beam_size, -1, -1)
                expanded_mask = batch_mask.expand(current_beam_size, -1) if batch_mask is not None else None
                
                # Get logits
                logits = decoder.generate_step(
                    input_ids=sequences,
                    encoder_outputs=expanded_encoder,
                    encoder_mask=expanded_mask
                )
                
                # Convert to log probabilities  
                log_probs = F.log_softmax(logits, dim=-1)
                vocab_size = log_probs.size(-1)
                
                # Calculate scores
                next_scores = scores.unsqueeze(-1) + log_probs
                next_scores = next_scores.view(-1)
                
                # Get top candidates
                top_scores, top_indices = next_scores.topk(min(self.beam_size * 2, len(next_scores)))
                
                # Convert back to beam and token indices
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                # Create new sequences
                new_sequences = []
                new_scores = []
                
                for score, beam_idx, token_idx in zip(top_scores, beam_indices, token_indices):
                    new_seq = torch.cat([sequences[beam_idx], token_idx.unsqueeze(0)])
                    
                    if token_idx.item() == tokenizer.eos_token_id:
                        # Apply length penalty
                        final_score = score.item() / (len(new_seq) ** self.length_penalty)
                        finished_sequences.append((new_seq.tolist(), final_score))
                    else:
                        new_sequences.append(new_seq)
                        new_scores.append(score)
                
                if new_sequences:
                    # Keep top beams
                    sequences = torch.stack(new_sequences[:self.beam_size])
                    scores = torch.stack(new_scores[:self.beam_size])
                else:
                    break
            
            # Select best sequence
            if finished_sequences:
                best_seq = max(finished_sequences, key=lambda x: x[1])[0]
            else:
                best_seq = sequences[0].tolist()
            
            results.append(best_seq)
        
        return results

class CAsiBeamSearch(SearchStrategy):
    """Context-Aware Schema Induction beam search - FIXED VERSION"""
    
    def __init__(self, beam_size: int = 4, max_length: int = 100, 
                 length_penalty: float = 0.6, schema_weight: float = 0.3):
        super().__init__(max_length)
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.schema_weight = schema_weight
        
        # FIXED: Initialize functional word mappings properly
        self.functional_categories = {
            'determiners': ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de'],
            'prepositions': ['à', 'au', 'aux', 'dans', 'sur', 'avec', 'pour', 'par', 'de'],
            'auxiliaries': ['est', 'sont', 'était', 'étaient', 'a', 'ont', 'avait', 'avaient'],
            'conjunctions': ['et', 'ou', 'mais', 'car', 'donc'],
            'pronouns': ['il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous']
        }
    
    def search(self, decoder, encoder_outputs: torch.Tensor, 
              schema: Dict[str, torch.Tensor],
              attention_mask: Optional[torch.Tensor] = None, 
              tokenizer=None, **kwargs) -> List[List[int]]:
        
        if tokenizer is None:
            raise ValueError("Tokenizer required for CASI beam search")
        
        # FIXED: Create functional token mappings from actual tokenizer
        self.functional_token_ids = self._create_functional_mappings(tokenizer)
        
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        results = []
        
        # Process each batch item separately
        for batch_idx in range(batch_size):
            # Get batch-specific data
            batch_encoder = encoder_outputs[batch_idx:batch_idx+1]
            batch_mask = attention_mask[batch_idx:batch_idx+1] if attention_mask is not None else None
            batch_schema = self._extract_batch_schema(schema, batch_idx)
            
            # Initialize beam
            sequences = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
            scores = torch.zeros(1, device=device)
            states = [self._initialize_beam_state(batch_schema)]
            finished_sequences = []
            
            for step in range(self.max_length):
                if len(finished_sequences) >= self.beam_size:
                    break
                
                candidates = []
                
                # Expand each beam
                for beam_idx in range(len(sequences)):
                    seq = sequences[beam_idx:beam_idx+1]
                    score = scores[beam_idx]
                    state = states[beam_idx]
                    
                    # Get logits
                    logits = decoder.generate_step(
                        input_ids=seq,
                        encoder_outputs=batch_encoder,
                        encoder_mask=batch_mask
                    )
                    
                    # FIXED: Apply schema guidance with actual token IDs
                    guided_logits = self._apply_schema_guidance(
                        logits.squeeze(0), state, batch_schema, step, tokenizer
                    )
                    
                    # Get probabilities
                    log_probs = F.log_softmax(guided_logits, dim=-1)
                    top_log_probs, top_tokens = log_probs.topk(self.beam_size)
                    
                    # Create candidates
                    for log_prob, token in zip(top_log_probs, top_tokens):
                        new_seq = torch.cat([seq.squeeze(0), token.unsqueeze(0)])
                        new_score = score + log_prob.item()
                        new_state = self._update_state(state.copy(), token.item(), tokenizer)
                        
                        # Schema alignment bonus
                        alignment_bonus = self._calculate_alignment(new_seq, new_state, batch_schema)
                        final_score = new_score + self.schema_weight * alignment_bonus
                        
                        if token.item() == tokenizer.eos_token_id:
                            # Apply length penalty
                            norm_score = final_score / (len(new_seq) ** self.length_penalty)
                            finished_sequences.append((new_seq.tolist(), norm_score))
                        else:
                            candidates.append((new_seq, final_score, new_state))
                
                # Keep top candidates
                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    candidates = candidates[:self.beam_size]
                    
                    sequences = torch.stack([c[0] for c in candidates]).unsqueeze(1)
                    scores = torch.tensor([c[1] for c in candidates], device=device)
                    states = [c[2] for c in candidates]
                else:
                    break
            
            # Select best sequence
            if finished_sequences:
                best_seq = max(finished_sequences, key=lambda x: x[1])[0]
            else:
                best_seq = sequences[0].squeeze().tolist() if len(sequences) > 0 else [tokenizer.bos_token_id, tokenizer.eos_token_id]
            
            results.append(best_seq)
        
        return results
    
    def _create_functional_mappings(self, tokenizer):
        """FIXED: Create mappings from functional categories to actual token IDs"""
        mappings = {}
        
        for category, words in self.functional_categories.items():
            token_ids = []
            for word in words:
                # Try different tokenization approaches
                candidates = [
                    word,
                    word.lower(),
                    word.upper(),
                    f" {word}",  # With leading space
                    f"{word} ",  # With trailing space
                ]
                
                for candidate in candidates:
                    if candidate in tokenizer.vocab:
                        token_id = tokenizer.vocab[candidate]
                        if token_id not in token_ids:
                            token_ids.append(token_id)
                        break
                else:
                    # Use tokenizer's convert_tokens_to_ids as fallback
                    try:
                        token_id = tokenizer.convert_tokens_to_ids(word)
                        if token_id != tokenizer.unk_token_id:
                            token_ids.append(token_id)
                    except:
                        pass
            
            mappings[category] = set(token_ids)
        
        return mappings
    
    def _extract_batch_schema(self, schema: Dict, batch_idx: int):
        """Extract schema for specific batch item"""
        batch_schema = {}
        for key, value in schema.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                batch_schema[key] = value[batch_idx] if value.size(0) > batch_idx else value[0]
            else:
                batch_schema[key] = value
        return batch_schema
    
    def _initialize_beam_state(self, schema):
        """Initialize beam state with schema information"""
        return {
            'functional_used': set(),
            'content_generated': set(),
            'position': 0,
            'structure_type': schema.get('structure_type', torch.tensor(0)).item() if isinstance(schema.get('structure_type'), torch.Tensor) else 0,
            'expected_functional': self._get_expected_functional_words(schema),
            'complexity_score': schema.get('complexity_score', torch.tensor(0.5)).item() if isinstance(schema.get('complexity_score'), torch.Tensor) else 0.5
        }
    
    def _get_expected_functional_words(self, schema):
        """Get expected functional words based on schema structure type"""
        structure_type = schema.get('structure_type', torch.tensor(0))
        if isinstance(structure_type, torch.Tensor):
            structure_type = structure_type.item()
        
        # Define expected functional words for different structures
        structure_expectations = {
            0: ['determiners'],  # Simple structures need determiners
            1: ['determiners', 'auxiliaries'],  # SVO needs determiners and verbs
            2: ['determiners', 'prepositions'],  # Structures with locations
            3: ['determiners', 'auxiliaries', 'prepositions'],  # Complex structures
        }
        
        return structure_expectations.get(structure_type, ['determiners'])
    
    def _apply_schema_guidance(self, logits, state, schema, step, tokenizer):
        """FIXED: Apply schema-based guidance with actual token IDs"""
        guided_logits = logits.clone()
        
        # Get expected functional categories for current position
        expected_categories = state['expected_functional']
        position = state['position']
        
        # Apply guidance based on position and schema
        if position < 8:  # Guide early tokens more strongly
            guidance_strength = 2.0 - (position * 0.2)  # Decreasing strength
            
            for category in expected_categories:
                if category in self.functional_token_ids:
                    token_ids = self.functional_token_ids[category]
                    for token_id in token_ids:
                        if token_id < len(guided_logits):
                            # Check if we haven't used too many of this category
                            category_used = len([t for t in state['functional_used'] if t in token_ids])
                            if category_used < 3:  # Limit repetition
                                guided_logits[token_id] += guidance_strength
        
        # Boost content words if we haven't generated enough content
        content_ratio = len(state['content_generated']) / max(position, 1)
        if content_ratio < 0.4 and position > 2:  # Need more content words
            # Boost tokens that are likely content words (higher token IDs, not functional)
            all_functional = set()
            for token_set in self.functional_token_ids.values():
                all_functional.update(token_set)
            
            for i in range(len(guided_logits)):
                if i not in all_functional and i > 1000:  # Heuristic for content words
                    guided_logits[i] += 0.5
        
        # Penalty for repeating functional words too much
        for token_id in state['functional_used']:
            if token_id < len(guided_logits):
                count = list(state['functional_used']).count(token_id)
                if count > 1:
                    guided_logits[token_id] -= count * 0.5
        
        return guided_logits
    
    def _update_state(self, state, token_id, tokenizer):
        """Update beam state based on generated token"""
        state['position'] += 1
        
        # Check if token is functional
        is_functional = False
        for category, token_ids in self.functional_token_ids.items():
            if token_id in token_ids:
                state['functional_used'].add(token_id)
                is_functional = True
                break
        
        # If not functional, consider it content
        if not is_functional and token_id not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
            state['content_generated'].add(token_id)
        
        return state
    
    def _calculate_alignment(self, sequence, state, schema):
        """Calculate schema alignment bonus"""
        alignment_score = 0.0
        
        # Reward appropriate functional word usage
        expected_categories = state['expected_functional']
        for category in expected_categories:
            if category in self.functional_token_ids:
                category_tokens = self.functional_token_ids[category]
                used_from_category = len([t for t in state['functional_used'] if t in category_tokens])
                if used_from_category > 0:
                    alignment_score += 0.3
        
        # Reward content generation
        if len(state['content_generated']) > 0:
            alignment_score += 0.2
        
        # Penalty for sequences that are too short or too long
        seq_len = len(sequence)
        optimal_length = 8 + state['complexity_score'] * 12  # 8-20 tokens based on complexity
        length_penalty = abs(seq_len - optimal_length) * 0.05
        alignment_score -= length_penalty
        
        return alignment_score