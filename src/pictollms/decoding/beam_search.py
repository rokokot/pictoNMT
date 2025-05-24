# src/pictollms/decoding/beam_search.py
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
    """Context-Aware Schema Induction beam search"""
    
    def __init__(self, beam_size: int = 4, max_length: int = 100, 
                 length_penalty: float = 0.6, schema_weight: float = 0.3):
        super().__init__(max_length)
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.schema_weight = schema_weight
    
    def search(self, decoder, encoder_outputs: torch.Tensor, 
              schema: Dict[str, torch.Tensor],
              attention_mask: Optional[torch.Tensor] = None, 
              tokenizer=None, **kwargs) -> List[List[int]]:
        
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        results = []
        
        # Simple functional word token IDs (would be set from tokenizer in practice)
        functional_tokens = self._get_functional_tokens(tokenizer)
        
        # Process each batch item separately
        for batch_idx in range(batch_size):
            # Get batch-specific data
            batch_encoder = encoder_outputs[batch_idx:batch_idx+1]
            batch_mask = attention_mask[batch_idx:batch_idx+1] if attention_mask is not None else None
            batch_schema = self._extract_batch_schema(schema, batch_idx)
            
            # Initialize beam
            sequences = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
            scores = torch.zeros(1, device=device)
            states = [{'position': 0, 'functional_used': set()}]
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
                    
                    # Apply schema guidance
                    guided_logits = self._apply_schema_guidance(
                        logits.squeeze(0), state, batch_schema, functional_tokens
                    )
                    
                    # Get probabilities
                    log_probs = F.log_softmax(guided_logits, dim=-1)
                    top_log_probs, top_tokens = log_probs.topk(self.beam_size)
                    
                    # Create candidates
                    for log_prob, token in zip(top_log_probs, top_tokens):
                        new_seq = torch.cat([seq.squeeze(0), token.unsqueeze(0)])
                        new_score = score + log_prob.item()
                        new_state = self._update_state(state.copy(), token.item(), functional_tokens)
                        
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
                best_seq = sequences[0].squeeze().tolist()
            
            results.append(best_seq)
        
        return results
    
    def _get_functional_tokens(self, tokenizer):
        """Get functional word token IDs"""
        functional_words = ['le', 'la', 'les', 'un', 'une', 'des', 'à', 'de', 'dans', 'sur', 'est', 'sont', 'et', 'ou']
        return set(tokenizer.convert_tokens_to_ids(word) for word in functional_words if word in tokenizer.vocab)
    
    def _extract_batch_schema(self, schema: Dict, batch_idx: int):
        """Extract schema for specific batch item"""
        batch_schema = {}
        for key, value in schema.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                batch_schema[key] = value[batch_idx] if value.size(0) > batch_idx else value[0]
            else:
                batch_schema[key] = value
        return batch_schema
    
    def _apply_schema_guidance(self, logits, state, schema, functional_tokens):
        """Apply schema-based guidance to logits"""
        guided_logits = logits.clone()
        
        # Simple guidance: boost functional words early in generation
        if state['position'] < 5:  # Early positions
            for token_id in functional_tokens:
                if token_id < len(guided_logits):
                    guided_logits[token_id] += 1.0
        
        return guided_logits
    
    def _update_state(self, state, token_id, functional_tokens):
        """Update beam state"""
        state['position'] += 1
        if token_id in functional_tokens:
            state['functional_used'].add(token_id)
        return state
    
    def _calculate_alignment(self, sequence, state, schema):
        """Calculate schema alignment bonus"""
        # Simple alignment: reward functional word usage
        functional_ratio = len(state['functional_used']) / max(state['position'], 1)
        return functional_ratio * 0.5