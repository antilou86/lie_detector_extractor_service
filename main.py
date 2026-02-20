"""
NLP Service - FastAPI server for claim extraction.

This service provides NLP-based claim extraction using spaCy.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

from claim_extractor import get_extractor, ExtractedClaim, ClaimType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fact Checker NLP Service",
    description="NLP-based claim extraction for the fact verification platform",
    version="1.0.0"
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class ExtractClaimsRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=100000)
    url: Optional[str] = None
    max_claims: int = Field(default=20, ge=1, le=100)


class EntityResponse(BaseModel):
    text: str
    label: str


class ClaimResponse(BaseModel):
    text: str
    claim_type: str
    confidence: float
    entities: List[EntityResponse]
    evidence_keywords: List[str]
    sentence_index: int
    char_start: int
    char_end: int


class ExtractClaimsResponse(BaseModel):
    claims: List[ClaimResponse]
    processing_time_ms: float
    text_length: int
    total_sentences: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


@app.on_event("startup")
async def startup_event():
    """Load the NLP model on startup."""
    logger.info("Loading NLP model...")
    start = time.time()
    try:
        extractor = get_extractor()
        elapsed = (time.time() - start) * 1000
        logger.info(f"NLP model loaded in {elapsed:.0f}ms")
    except Exception as e:
        logger.error(f"Failed to load NLP model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        extractor = get_extractor()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=extractor.nlp.meta["name"]
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name="unknown"
        )


@app.post("/extract", response_model=ExtractClaimsResponse)
async def extract_claims(request: ExtractClaimsRequest):
    """
    Extract verifiable claims from text.
    
    This endpoint analyzes the input text using NLP to identify sentences
    that contain verifiable factual claims.
    """
    start_time = time.time()
    
    try:
        extractor = get_extractor()
        
        # Extract claims
        claims = extractor.extract_claims(request.text, max_claims=request.max_claims)
        
        # Count total sentences (for stats)
        doc = extractor.nlp(request.text)
        total_sentences = len(list(doc.sents))
        
        # Convert to response format
        claim_responses = [
            ClaimResponse(
                text=claim.text,
                claim_type=claim.claim_type.value,
                confidence=round(claim.confidence, 3),
                entities=[EntityResponse(**e) for e in claim.entities],
                evidence_keywords=claim.evidence_keywords[:5],
                sentence_index=claim.sentence_index,
                char_start=claim.char_start,
                char_end=claim.char_end,
            )
            for claim in claims
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Extracted {len(claims)} claims from {len(request.text)} chars "
            f"in {processing_time:.0f}ms"
        )
        
        return ExtractClaimsResponse(
            claims=claim_responses,
            processing_time_ms=round(processing_time, 2),
            text_length=len(request.text),
            total_sentences=total_sentences,
        )
        
    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-sentence")
async def analyze_sentence(text: str):
    """
    Analyze a single sentence for claim likelihood.
    Useful for debugging and testing.
    """
    try:
        extractor = get_extractor()
        doc = extractor.nlp(text)
        
        # Get the first sentence
        sents = list(doc.sents)
        if not sents:
            return {"error": "No sentence found"}
        
        sent = sents[0]
        analysis = extractor._analyze_sentence(sent, sent.text.strip())
        
        return {
            "text": sent.text.strip(),
            "confidence": round(analysis['confidence'], 3),
            "claim_type": analysis['claim_type'].value,
            "entities": analysis['entities'],
            "evidence_keywords": analysis['evidence_keywords'],
            "reasons": analysis['reasons'],
        }
        
    except Exception as e:
        logger.error(f"Sentence analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002)
