"""
Claim Extractor - Uses spaCy NLP to identify verifiable claims in text.

This module analyzes text to find sentences that contain verifiable factual claims,
distinguishing them from opinions, questions, and general statements.
"""

import spacy
from spacy.tokens import Span, Doc
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re


class ClaimType(Enum):
    STATISTIC = "statistic"
    QUOTE = "quote"
    CAUSAL = "causal"
    COMPARISON = "comparison"
    HISTORICAL = "historical"
    ATTRIBUTION = "attribution"
    MEDICAL = "medical"
    SCIENTIFIC = "scientific"


@dataclass
class ExtractedClaim:
    text: str
    claim_type: ClaimType
    confidence: float
    entities: List[Dict[str, str]]
    evidence_keywords: List[str]
    sentence_index: int
    char_start: int
    char_end: int


class ClaimExtractor:
    """
    Extracts verifiable claims from text using spaCy NLP.
    """
    
    # Patterns that indicate a verifiable claim
    CLAIM_INDICATORS = {
        # Statistics and numbers
        'statistic': [
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\d{1,3}(?:,\d{3})+',  # Large numbers with commas
            r'\$\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?',  # Money
            r'\d+(?:\.\d+)?\s*(?:million|billion|trillion)',  # Large quantities
            r'(?:doubled|tripled|quadrupled|halved)',  # Multipliers
            r'(?:increase|decrease|rise|fall|drop|surge|spike|decline)\s+(?:of\s+)?\d+',
        ],
        # Causal claims
        'causal': [
            r'(?:causes?|caused|causing)',
            r'(?:leads?\s+to|led\s+to)',
            r'(?:results?\s+in|resulted\s+in)',
            r'(?:because\s+of|due\s+to)',
            r'(?:prevents?|prevented)',
            r'(?:increases?|decreases?)\s+(?:the\s+)?(?:risk|chance|likelihood)',
        ],
        # Comparisons
        'comparison': [
            r'more\s+(?:than|effective)',
            r'less\s+(?:than|effective)',
            r'(?:higher|lower|better|worse)\s+than',
            r'(?:compared\s+to|in\s+comparison)',
            r'(?:fastest|slowest|highest|lowest|best|worst)',
        ],
        # Quotes and attributions
        'quote': [
            r'(?:said|stated|claimed|argued|asserted|declared|announced)',
            r'according\s+to',
            r'"[^"]{20,}"',  # Quoted text
        ],
        # Medical/health claims
        'medical': [
            r'(?:treats?|cures?|prevents?|reduces?)\s+(?:the\s+)?(?:risk\s+of\s+)?(?:\w+\s+){0,3}(?:disease|cancer|diabetes|infection|illness)',
            r'(?:vaccine|medication|drug|treatment|therapy)\s+(?:is|was|has\s+been)',
            r'(?:study|research|trial|experiment)\s+(?:shows?|found|demonstrates?|proves?)',
            r'(?:FDA|CDC|WHO|NIH)\s+(?:approved|recommends?|warns?)',
            r'(?:symptoms?|side\s+effects?|adverse\s+(?:effects?|reactions?))',
        ],
        # Historical/factual events
        'historical': [
            r'in\s+(?:19|20)\d{2}',  # Year references
            r'(?:founded|established|created|invented)\s+(?:in|by)',
            r'(?:first|last|only)\s+(?:person|time|country|company)',
        ],
    }
    
    # Patterns that indicate NON-claims (opinions, questions, etc.)
    NON_CLAIM_PATTERNS = [
        r'^(?:I\s+)?(?:think|believe|feel|hope|wish|want)',  # Opinion indicators
        r'^(?:maybe|perhaps|possibly|probably)',  # Uncertainty
        r'\?$',  # Questions
        r'^(?:should|could|would|might)\s',  # Modal hedging at start
        r'^(?:if|when|unless)\s',  # Conditional statements
        r'(?:in\s+my\s+opinion|personally|I\s+guess)',  # Explicit opinions
        r'^(?:let\'s|let\s+us)',  # Suggestions
        r'^(?:please|kindly)',  # Requests
    ]
    
    # Entity types that increase claim likelihood
    HIGH_VALUE_ENTITIES = {
        'ORG': 1.5,      # Organizations
        'PERSON': 1.3,   # People
        'GPE': 1.2,      # Countries/cities
        'DATE': 1.3,     # Dates
        'MONEY': 1.5,    # Money amounts
        'PERCENT': 1.6,  # Percentages
        'QUANTITY': 1.4, # Quantities
        'CARDINAL': 1.2, # Numbers
    }
    
    # Minimum thresholds
    MIN_SENTENCE_LENGTH = 40
    MAX_SENTENCE_LENGTH = 500
    MIN_CONFIDENCE = 0.4
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize with spaCy model."""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Trying en_core_web_sm...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "No spaCy model found. Install with: python -m spacy download en_core_web_lg"
                )
    
    def extract_claims(self, text: str, max_claims: int = 50) -> List[ExtractedClaim]:
        """
        Extract verifiable claims from text.
        
        Args:
            text: Input text to analyze
            max_claims: Maximum number of claims to return
            
        Returns:
            List of ExtractedClaim objects sorted by confidence
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        claims = []
        
        for sent_idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            
            # Skip sentences that are too short or too long
            if len(sent_text) < self.MIN_SENTENCE_LENGTH:
                continue
            if len(sent_text) > self.MAX_SENTENCE_LENGTH:
                continue
            
            # Check if this looks like a claim
            claim_analysis = self._analyze_sentence(sent, sent_text)
            
            if claim_analysis['confidence'] >= self.MIN_CONFIDENCE:
                claim = ExtractedClaim(
                    text=sent_text,
                    claim_type=claim_analysis['claim_type'],
                    confidence=claim_analysis['confidence'],
                    entities=claim_analysis['entities'],
                    evidence_keywords=claim_analysis['evidence_keywords'],
                    sentence_index=sent_idx,
                    char_start=sent.start_char,
                    char_end=sent.end_char,
                )
                claims.append(claim)
        
        # Sort by confidence and limit
        claims.sort(key=lambda c: c.confidence, reverse=True)
        return claims[:max_claims]
    
    def _analyze_sentence(self, sent: Span, sent_text: str) -> Dict[str, Any]:
        """
        Analyze a sentence to determine if it's a verifiable claim.
        """
        result = {
            'confidence': 0.0,
            'claim_type': ClaimType.STATISTIC,
            'entities': [],
            'evidence_keywords': [],
            'reasons': [],
        }
        
        # Check for non-claim patterns (disqualifying)
        for pattern in self.NON_CLAIM_PATTERNS:
            if re.search(pattern, sent_text, re.IGNORECASE):
                result['reasons'].append(f"Non-claim pattern: {pattern}")
                return result
        
        score = 0.3  # Base score
        detected_type = None
        
        # Check for claim indicator patterns
        for claim_type, patterns in self.CLAIM_INDICATORS.items():
            for pattern in patterns:
                matches = re.findall(pattern, sent_text, re.IGNORECASE)
                if matches:
                    score += 0.15 * len(matches)
                    result['evidence_keywords'].extend(matches[:3])
                    if detected_type is None:
                        detected_type = claim_type
                    result['reasons'].append(f"Pattern match: {claim_type}")
        
        # Analyze named entities
        entities = []
        for ent in sent.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
            })
            
            # Boost score for high-value entity types
            if ent.label_ in self.HIGH_VALUE_ENTITIES:
                score += 0.1 * self.HIGH_VALUE_ENTITIES[ent.label_]
                result['reasons'].append(f"Entity: {ent.label_}")
        
        result['entities'] = entities
        
        # Analyze verb tenses and modality
        verb_analysis = self._analyze_verbs(sent)
        score += verb_analysis['score_modifier']
        result['reasons'].extend(verb_analysis['reasons'])
        
        # Check sentence structure
        structure_score = self._analyze_structure(sent, sent_text)
        score += structure_score
        
        # Assign claim type
        if detected_type:
            result['claim_type'] = ClaimType(detected_type)
        elif any(ent['label'] in ['PERCENT', 'MONEY', 'CARDINAL', 'QUANTITY'] for ent in entities):
            result['claim_type'] = ClaimType.STATISTIC
        
        # Cap confidence at 1.0
        result['confidence'] = min(1.0, max(0.0, score))
        
        return result
    
    def _analyze_verbs(self, sent: Span) -> Dict[str, Any]:
        """Analyze verbs for factual vs hedged language."""
        result = {'score_modifier': 0.0, 'reasons': []}
        
        for token in sent:
            if token.pos_ == 'VERB':
                # Past tense often indicates factual claims
                if token.tag_ in ['VBD', 'VBN']:  # Past tense verbs
                    result['score_modifier'] += 0.05
                    result['reasons'].append("Past tense verb")
                
                # Modal verbs reduce confidence
                if token.tag_ == 'MD':
                    result['score_modifier'] -= 0.1
                    result['reasons'].append("Modal verb (hedging)")
                
                # Assertive verbs increase confidence
                assertive_lemmas = {'prove', 'show', 'demonstrate', 'confirm', 'find', 
                                   'discover', 'reveal', 'establish', 'indicate'}
                if token.lemma_.lower() in assertive_lemmas:
                    result['score_modifier'] += 0.15
                    result['reasons'].append(f"Assertive verb: {token.lemma_}")
        
        return result
    
    def _analyze_structure(self, sent: Span, sent_text: str) -> float:
        """Analyze sentence structure for claim-like patterns."""
        score = 0.0
        
        # Subject-verb-object structure is common in claims
        has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in sent)
        has_object = any(token.dep_ in ['dobj', 'pobj', 'attr'] for token in sent)
        
        if has_subject and has_object:
            score += 0.1
        
        # Multiple clauses can indicate complex claims
        num_verbs = sum(1 for token in sent if token.pos_ == 'VERB')
        if num_verbs >= 2:
            score += 0.05
        
        # Presence of specific determiners
        if re.search(r'\b(?:the|this|that|these|those)\s+(?:study|research|report|data|evidence)', 
                    sent_text, re.IGNORECASE):
            score += 0.15
        
        return score


# Singleton instance for reuse
_extractor: Optional[ClaimExtractor] = None


def get_extractor(model_name: str = "en_core_web_lg") -> ClaimExtractor:
    """Get or create the claim extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = ClaimExtractor(model_name)
    return _extractor
