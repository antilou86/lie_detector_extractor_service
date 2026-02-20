# NLP Service

Python microservice for NLP-based claim extraction using spaCy.

## Setup

### 1. Create virtual environment
```bash
cd nlp-service
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy model
```bash
# Large model (recommended, ~560MB)
python -m spacy download en_core_web_lg

# Or small model (faster download, less accurate, ~12MB)
python -m spacy download en_core_web_sm
```

### 4. Run the service
```bash
# Development
uvicorn main:app --reload --port 3002

# Production
uvicorn main:app --host 0.0.0.0 --port 3002
```

## API Endpoints

### `GET /health`
Health check endpoint.

### `POST /extract`
Extract verifiable claims from text.

**Request:**
```json
{
  "text": "The study found that 75% of participants showed improvement. According to the CDC, this treatment reduces risk by 40%.",
  "url": "https://example.com/article",
  "max_claims": 20
}
```

**Response:**
```json
{
  "claims": [
    {
      "text": "The study found that 75% of participants showed improvement.",
      "claim_type": "statistic",
      "confidence": 0.85,
      "entities": [
        {"text": "75%", "label": "PERCENT"}
      ],
      "evidence_keywords": ["75%", "found"],
      "sentence_index": 0,
      "char_start": 0,
      "char_end": 60
    }
  ],
  "processing_time_ms": 45.2,
  "text_length": 120,
  "total_sentences": 2
}
```

### `POST /analyze-sentence?text=...`
Analyze a single sentence (for debugging).

## Claim Types

- `statistic` - Contains numbers, percentages, or quantitative data
- `quote` - Attribution to a person or organization
- `causal` - Claims about cause and effect
- `comparison` - Comparative claims
- `medical` - Health and medical claims
- `scientific` - Scientific findings
- `historical` - Historical events or facts
- `attribution` - Claims attributed to sources

## Configuration

The extractor uses these thresholds (configurable in `claim_extractor.py`):

- `MIN_SENTENCE_LENGTH`: 40 characters
- `MAX_SENTENCE_LENGTH`: 500 characters  
- `MIN_CONFIDENCE`: 0.4 (out of 1.0)

## Testing

```bash
# Test health endpoint
curl http://localhost:3002/health

# Test extraction
curl -X POST http://localhost:3002/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Studies show that vaccines reduce COVID-19 hospitalizations by 90%."}'
```
