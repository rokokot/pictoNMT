import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

"""
Main contextualized decoding module, runs a beam search over defined functional categories, see docs

"""


# the first step is to define a beam for our search algorithm
class BeamHypothesis:

    def __init__(self, tokens: List[int], score: float, state: Dict[str, Any]):
        self.tokens = tokens
        self.score = score
        self.state = state
    
    def __len__(self):
        return len(self.tokens)
    
    def __lt__(self, other):
        return self.score < other.score



class CAsiBeamSearch:
    """induction search function"""
    
    def __init__(self, 
                 beam_size: int = 4,
                 max_length: int = 100,
                 length_penalty: float = 0.6,
                 schema_weight: float = 0.3,
                 early_stopping: bool = True):
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.schema_weight = schema_weight
        self.early_stopping = early_stopping
        
        self.functional_words = {
            'determiners': {
                'masculine_singular': [10, 11],  # le, un
                'feminine_singular': [12, 13],   # la, une
                'plural': [14, 15]               # les, des
            },
            'prepositions': {
                'location': [20, 21, 22],        # à, dans, sur
                'direction': [23, 24],           # vers, jusqu'à
                'possession': [25]               # de
            },
            'auxiliaries': {
                'être': [30, 31],                # est, sont
                'avoir': [32, 33]                # a, ont
            }
        }
    

    #   search algorithm

    def search(self, model, encoder_outputs: torch.Tensor, schema: Dict[str, torch.Tensor], tokenizer, attention_mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        run beam search
        
        Args:
            model = decoder
            encoder_outputs:  [batch_size, src_len, hidden_size]
            schema information from SchemaInducer
            tokenizer for special tokens
            attention_mask form source
            
        Outs
            list of sequences for each item in batch
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize beams for each batch item
        all_beams = []
        
        for batch_idx in range(batch_size):
            batch_schema = self._extract_batch_schema(schema, batch_idx)
            
            initial_beam = BeamHypothesis(tokens=[tokenizer.bos_token_id],score=0.0,state=self._initialize_beam_state(batch_schema))
            
            beams = [initial_beam]
            finished_beams = []
            
            for step in range(self.max_length):
                if len(finished_beams) >= self.beam_size and self.early_stopping:
                    break
                
                candidates = []
                
                for beam in beams:
                    if len(beam.tokens) >= self.max_length:
                        finished_beams.append(beam)
                        continue
                    
                    next_token_logits = self._get_next_token_logits(model, beam.tokens, encoder_outputs[batch_idx:batch_idx+1],attention_mask[batch_idx:batch_idx+1] if attention_mask is not None else None)
                    
                    #     work on schema guidance
                    guided_logits = self._apply_schema_guidance(
                        next_token_logits, beam.state, batch_schema, step
                    )
                    
                    # softmax probabilities
                    probs = F.softmax(guided_logits, dim=-1)
                    
                    #  top-k args
                    top_k = min(self.beam_size * 2, probs.size(-1))
                    topk_probs, topk_indices = torch.topk(probs, top_k)
                    
                    # generate successort
                    for prob, token_id in zip(topk_probs, topk_indices):
                        new_tokens = beam.tokens + [token_id.item()]
                        new_score = beam.score - math.log(prob.item())
                        
                        new_state = self._update_beam_state(beam.state.copy(), token_id.item(), batch_schema)
                        
                        alignment_bonus = self._calculate_schema_alignment(new_tokens, new_state, batch_schema)
                        
                        final_score = new_score - self.schema_weight * alignment_bonus
                        
                        candidate = BeamHypothesis(new_tokens, final_score, new_state)
                        
                        if token_id.item() == tokenizer.eos_token_id:
                            normalized_score = self._apply_length_penalty(final_score, len(new_tokens))
                            candidate.score = normalized_score
                            finished_beams.append(candidate)
                        else:
                            candidates.append(candidate)
                
                #     top beams for next iteration
                candidates.sort(key=lambda x: x.score)
                beams = candidates[:self.beam_size]
                
                if not beams:
                    break
            
            # best finished beam
            if finished_beams:
                best_beam = min(finished_beams, key=lambda x: x.score)
            else:
                best_beam = min(beams, key=lambda x: x.score) if beams else initial_beam
            
            all_beams.append(best_beam.tokens)
        
        return all_beams
    
    def _extract_batch_schema(self, schema: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Extract schema information for a specific batch item"""
        batch_schema = {}
        
        for key, value in schema.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 1:
                    batch_schema[key] = value[batch_idx]
                else:
                    batch_schema[key] = value
            else:
                batch_schema[key] = value
        
        return batch_schema
    
    def _initialize_beam_state(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize beam state with schema information"""
        return {
            'generated_content': set(),
            'used_functional': set(),
            'position': 0,
            'structure_type': schema.get('structure_type', 0),
            'expected_elements': self._get_expected_elements(schema),
            'complexity_score': schema.get('complexity_score', 0.5)
        }
    


    # linguistic elements in the decoding are hard coded with a custom rule set
    # this needs to be well-documented
    def _get_expected_elements(self, schema: Dict[str, Any]) -> List[str]:
        """Get expected functional elements based on schema"""
        structure_type = schema.get('structure_type', 0)
        
        structure_expectations = {
            0: ['determiner', 'noun', 'verb'],                    # Simple SV
            1: ['determiner', 'noun', 'verb', 'determiner', 'noun'],  # SVO
            2: ['adverb', 'determiner', 'noun', 'verb'],          # Time-SV
            3: ['determiner', 'noun', 'verb', 'preposition', 'noun'], # SV-Location
        }
        
        return structure_expectations.get(structure_type, ['determiner', 'noun', 'verb'])
    
    def _get_next_token_logits(self, model, tokens: List[int], encoder_outputs: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        input_ids = torch.tensor([tokens], device=encoder_outputs.device)
        
        logits = model.decoder.generate_step(input_ids=input_ids,encoder_outputs=encoder_outputs,encoder_mask=attention_mask)

        if logits is not None:
          return logits.squeeze(0) if logits.dim() > 1 else logits
        else:
            # Fallback: This should not happen in normal operation
            logger.warning("Decoder returned None for logits, using random logits")
            # Get vocab size from the model or use a default
            vocab_size = getattr(model.decoder, 'vocab_size', 32000)
            return torch.randn(vocab_size, device=encoder_outputs.device)

    def _apply_schema_guidance(self, logits: torch.Tensor, state: Dict[str, Any], schema: Dict[str, Any], step: int) -> torch.Tensor:
        guided_logits = logits.clone()
        
        expected_elements = state['expected_elements']
        position = state['position']
        
        if position < len(expected_elements):
            expected_element = expected_elements[position]
            
            # Boost appropriate functional words
            if expected_element == 'determiner':
                # Boost determiner tokens
                for det_type, token_ids in self.functional_words['determiners'].items():
                    for token_id in token_ids:
                        if token_id < len(guided_logits):
                            guided_logits[token_id] += 2.0
            
            elif expected_element == 'preposition':
                # Boost preposition tokens
                for prep_type, token_ids in self.functional_words['prepositions'].items():
                    for token_id in token_ids:
                        if token_id < len(guided_logits):
                            guided_logits[token_id] += 1.5
            
            elif expected_element == 'auxiliary':
                # Boost auxiliary verb tokens
                for aux_type, token_ids in self.functional_words['auxiliaries'].items():
                    for token_id in token_ids:
                        if token_id < len(guided_logits):
                            guided_logits[token_id] += 1.8
        
        return guided_logits
    
    def _update_beam_state(self, state: Dict[str, Any], token_id: int, schema: Dict[str, Any]) -> Dict[str, Any]:
       
        state['position'] += 1
        
        # Check if token is a functional word
        if self._is_functional_word(token_id):
            state['used_functional'].add(token_id)
        
        # Check if token represents content
        if self._is_content_word(token_id):
            state['generated_content'].add(token_id)
        
        return state
    
    def _is_functional_word(self, token_id: int) -> bool:
        all_functional = []
        for category in self.functional_words.values():
            for token_list in category.values():
                all_functional.extend(token_list)
        
        return token_id in all_functional
    
    def _is_content_word(self, token_id: int) -> bool:
        return not self._is_functional_word(token_id) and token_id > 100
    


    def _calculate_schema_alignment(self, tokens: List[int], state: Dict[str, Any], schema: Dict[str, Any]) -> float:
        alignment_score = 0.0
        
        expected_functional = len(state['expected_elements'])
        used_functional = len(state['used_functional'])
        
        if expected_functional > 0:
            functional_ratio = used_functional / expected_functional
            alignment_score += functional_ratio * 0.5
        
        if state['generated_content']:
            alignment_score += 0.3
        
        complexity = state['complexity_score']
        length_penalty = max(0, len(tokens) - complexity * 20) * 0.1
        alignment_score -= length_penalty
        
        return alignment_score
    
    # optional steering module
    def _apply_length_penalty(self, score: float, length: int) -> float:
        """Apply length penalty to final score"""
        return score / (length ** self.length_penalty)