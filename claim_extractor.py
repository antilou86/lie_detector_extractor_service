"""
Claim Extractor - Uses NLTK to identify verifiable claims in text.

This module analyzes text to find sentences that contain verifiable factual claims,
distinguishing them from opinions, questions, and general statements.
"""

import nltk
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


# Download required NLTK data on import
def download_nltk_data():
    """Download required NLTK data packages."""
    packages = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('chunkers/maxent_ne_chunker_tab', 'maxent_ne_chunker_tab'),
        ('corpora/words', 'words'),
    ]
    for path, package in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass  # Some packages may not exist, that's ok


# Download on module load
download_nltk_data()


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
    Extracts verifiable claims from text using NLTK.
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
    
    # POS tags that indicate factual vs hedged language
    PAST_TENSE_TAGS = {'VBD', 'VBN'}  # Past tense verbs
    MODAL_TAGS = {'MD'}  # Modal verbs
    
    # Minimum thresholds
    MIN_SENTENCE_LENGTH = 40
    MAX_SENTENCE_LENGTH = 500
    MIN_CONFIDENCE = 0.4
    
    def __init__(self):
        """Initialize the extractor."""
        self.model_name = "nltk"
    
    def extract_claims(self, text: str, max_claims: int = 50) -> List[ExtractedClaim]:
        """
        Extract verifiable claims from text.
        
        Args:
            text: Input text to analyze
            max_claims: Maximum number of claims to return
            
        Returns:
            List of ExtractedClaim objects sorted by confidence
        """
        # Tokenize into sentences
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Fallback to simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        claims = []
        char_offset = 0
        
        for sent_idx, sent_text in enumerate(sentences):
            sent_text = sent_text.strip()
            
            # Track character positions
            char_start = text.find(sent_text, char_offset)
            if char_start == -1:
                char_start = char_offset
            char_end = char_start + len(sent_text)
            char_offset = char_end
            
            # Skip sentences that are too short or too long
            if len(sent_text) < self.MIN_SENTENCE_LENGTH:
                continue
            if len(sent_text) > self.MAX_SENTENCE_LENGTH:
                continue
            
            # Check if this looks like a claim
            claim_analysis = self._analyze_sentence(sent_text)
            
            if claim_analysis['confidence'] >= self.MIN_CONFIDENCE:
                claim = ExtractedClaim(
                    text=sent_text,
                    claim_type=claim_analysis['claim_type'],
                    confidence=claim_analysis['confidence'],
                    entities=claim_analysis['entities'],
                    evidence_keywords=claim_analysis['evidence_keywords'],
                    sentence_index=sent_idx,
                    char_start=char_start,
                    char_end=char_end,
                )
                claims.append(claim)
        
        # Sort by confidence and limit
        claims.sort(key=lambda c: c.confidence, reverse=True)
        return claims[:max_claims]
    
    def _analyze_sentence(self, sent_text: str) -> Dict[str, Any]:
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
        
        # Tokenize and POS tag
        try:
            tokens = nltk.word_tokenize(sent_text)
            pos_tags = nltk.pos_tag(tokens)
        except Exception:
            # Fallback: simple tokenization
            tokens = sent_text.split()
            pos_tags = [(t, 'NN') for t in tokens]
        
        # Extract named entities
        entities = self._extract_entities(pos_tags)
        result['entities'] = entities
        
        # Boost for entity types
        entity_labels = [e['label'] for e in entities]
        if 'ORGANIZATION' in entity_labels:
            score += 0.15
        if 'PERSON' in entity_labels:
            score += 0.13
        if 'GPE' in entity_labels:  # Geo-political entity
            score += 0.12
        
        # Boost for numbers (likely statistics)
        num_count = sum(1 for _, tag in pos_tags if tag == 'CD')  # Cardinal numbers
        if num_count > 0:
            score += 0.12 * min(num_count, 3)
            result['reasons'].append(f"Contains {num_count} numbers")
        
        # Analyze verb tenses and modality
        verb_analysis = self._analyze_verbs(pos_tags)
        score += verb_analysis['score_modifier']
        result['reasons'].extend(verb_analysis['reasons'])
        
        # Check sentence structure
        structure_score = self._analyze_structure(pos_tags, sent_text)
        score += structure_score
        
        # Assign claim type
        if detected_type:
            result['claim_type'] = ClaimType(detected_type)
        elif num_count > 0:
            result['claim_type'] = ClaimType.STATISTIC
        
        # Cap confidence at 1.0
        result['confidence'] = min(1.0, max(0.0, score))
        
        return result
    
    def _extract_entities(self, pos_tags: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """Extract named entities using NLTK's chunker."""
        entities = []
        
        try:
            # Use NLTK's named entity chunker
            chunks = nltk.ne_chunk(pos_tags)
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join(c[0] for c in chunk)
                    entities.append({
                        'text': entity_text,
                        'label': chunk.label(),
                    })
        except Exception:
            # Fallback: look for capitalized sequences (simple NER)
            current_entity = []
            for token, tag in pos_tags:
                if tag.startswith('NNP'):  # Proper noun
                    current_entity.append(token)
                elif current_entity:
                    entities.append({
                        'text': ' '.join(current_entity),
                        'label': 'ENTITY',
                    })
                    current_entity = []
            
            if current_entity:
                entities.append({
                    'text': ' '.join(current_entity),
                    'label': 'ENTITY',
                })
        
        return entities
    
    def _analyze_verbs(self, pos_tags: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze verbs for factual vs hedged language."""
        result = {'score_modifier': 0.0, 'reasons': []}
        
        # Check for assertive verbs
        assertive_verbs = {'prove', 'show', 'demonstrate', 'confirm', 'find', 
                          'discover', 'reveal', 'establish', 'indicate',
                          'proved', 'showed', 'demonstrated', 'confirmed', 'found',
                          'discovered', 'revealed', 'established', 'indicated'}
        
        for token, tag in pos_tags:
            # Past tense often indicates factual claims
            if tag in self.PAST_TENSE_TAGS:
                result['score_modifier'] += 0.05
                result['reasons'].append("Past tense verb")
            
            # Modal verbs reduce confidence
            if tag in self.MODAL_TAGS:
                result['score_modifier'] -= 0.1
                result['reasons'].append("Modal verb (hedging)")
            
            # Assertive verbs increase confidence
            if token.lower() in assertive_verbs:
                result['score_modifier'] += 0.15
                result['reasons'].append(f"Assertive verb: {token}")
        
        return result
    
    def _analyze_structure(self, pos_tags: List[Tuple[str, str]], sent_text: str) -> float:
        """Analyze sentence structure for claim-like patterns."""
        score = 0.0
        
        # Check for subject-verb structure
        has_noun = any(tag.startswith('NN') for _, tag in pos_tags)
        has_verb = any(tag.startswith('VB') for _, tag in pos_tags)
        
        if has_noun and has_verb:
            score += 0.1
        
        # Multiple verbs can indicate complex claims
        num_verbs = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
        if num_verbs >= 2:
            score += 0.05
        
        # Presence of specific determiners
        if re.search(r'\b(?:the|this|that|these|those)\s+(?:study|research|report|data|evidence)', 
                    sent_text, re.IGNORECASE):
            score += 0.15
        
        return score


# Singleton instance for reuse
_extractor: Optional[ClaimExtractor] = None


def get_extractor() -> ClaimExtractor:
    """Get or create the claim extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = ClaimExtractor()
    return _extractor
