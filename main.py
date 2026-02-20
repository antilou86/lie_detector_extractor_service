"""
NLP Service - Flask server for claim extraction.

This service provides NLP-based claim extraction using NLTK.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging

from claim_extractor import get_extractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for local development


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        extractor = get_extractor()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "model_name": extractor.model_name
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "model_name": "unknown"
        }), 500


@app.route('/extract', methods=['POST'])
def extract_claims():
    """
    Extract verifiable claims from text.
    
    This endpoint analyzes the input text using NLP to identify sentences
    that contain verifiable factual claims.
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        max_claims = data.get('max_claims', 20)
        
        if len(text) < 10:
            return jsonify({"error": "Text too short (min 10 chars)"}), 400
        
        if len(text) > 100000:
            return jsonify({"error": "Text too long (max 100000 chars)"}), 400
        
        extractor = get_extractor()
        
        # Extract claims
        claims = extractor.extract_claims(text, max_claims=max_claims)
        
        # Count total sentences (for stats)
        import nltk
        try:
            total_sentences = len(nltk.sent_tokenize(text))
        except Exception:
            total_sentences = len(text.split('. '))
        
        # Convert to response format
        claim_responses = [
            {
                "text": claim.text,
                "claim_type": claim.claim_type.value,
                "confidence": round(claim.confidence, 3),
                "entities": claim.entities,
                "evidence_keywords": claim.evidence_keywords[:5],
                "sentence_index": claim.sentence_index,
                "char_start": claim.char_start,
                "char_end": claim.char_end,
            }
            for claim in claims
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Extracted {len(claims)} claims from {len(text)} chars "
            f"in {processing_time:.0f}ms"
        )
        
        return jsonify({
            "claims": claim_responses,
            "processing_time_ms": round(processing_time, 2),
            "text_length": len(text),
            "total_sentences": total_sentences,
        })
        
    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-sentence', methods=['POST'])
def analyze_sentence():
    """
    Analyze a single sentence for claim likelihood.
    Useful for debugging and testing.
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        extractor = get_extractor()
        
        # Analyze the text directly
        analysis = extractor._analyze_sentence(text.strip())
        
        return jsonify({
            "text": text.strip(),
            "confidence": round(analysis['confidence'], 3),
            "claim_type": analysis['claim_type'].value,
            "entities": analysis['entities'],
            "evidence_keywords": analysis['evidence_keywords'],
            "reasons": analysis['reasons'],
        })
        
    except Exception as e:
        logger.error(f"Sentence analysis failed: {e}")
        return jsonify({"error": str(e)}), 500


# Initialize extractor on startup
with app.app_context():
    logger.info("Loading NLP model...")
    start = time.time()
    try:
        extractor = get_extractor()
        elapsed = (time.time() - start) * 1000
        logger.info(f"NLP model loaded in {elapsed:.0f}ms")
    except Exception as e:
        logger.error(f"Failed to load NLP model: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3002, debug=False)
