from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cohere
from pinecone import Pinecone
from anthropic import Client
import os
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import defaultdict

# Load environment variables
load_dotenv()

# Verify environment variables
required_env_vars = ['COHERE_API_KEY', 'PINECONE_API_KEY', 'ANTHROPIC_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients using environment variables
cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
anthropic_client = Client(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Connect to existing Pinecone index
index = pc.Index("orwell-chunks")

# Rate limiting setup
RATE_LIMIT_MINUTES = 30
RATE_LIMIT_REQUESTS = 3

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
    
    def _clean_old_requests(self, ip: str):
        """Remove requests older than the rate limit window"""
        cutoff = datetime.now() - timedelta(minutes=RATE_LIMIT_MINUTES)
        self.requests[ip] = [req for req in self.requests[ip] if req > cutoff]
    
    def is_rate_limited(self, ip: str) -> bool:
        """Check if an IP has exceeded the rate limit"""
        self._clean_old_requests(ip)
        return len(self.requests[ip]) >= RATE_LIMIT_REQUESTS
    
    def add_request(self, ip: str):
        """Record a new request"""
        self._clean_old_requests(ip)
        self.requests[ip].append(datetime.now())
    
    def get_remaining_requests(self, ip: str) -> dict:
        """Get remaining requests and time until reset"""
        self._clean_old_requests(ip)
        requests_made = len(self.requests[ip])
        requests_remaining = RATE_LIMIT_REQUESTS - requests_made
        
        if requests_made > 0:
            oldest_request = min(self.requests[ip])
            reset_time = oldest_request + timedelta(minutes=RATE_LIMIT_MINUTES)
            seconds_until_reset = max(0, (reset_time - datetime.now()).total_seconds())
        else:
            seconds_until_reset = 0
            
        return {
            "requests_remaining": requests_remaining,
            "seconds_until_reset": int(seconds_until_reset)
        }

rate_limiter = RateLimiter()

class Query(BaseModel):
    question: str
    style: str = "scholarly"  # Default to scholarly style if not specified

class Response(BaseModel):
    answer: str
    rate_limit_info: dict

# Example for creative storytelling style
STORYTELLING_EXAMPLES = """
Example creative responses:

Q: What does Room 101 represent?
A: Let me take you inside Winston's mind as he faces Room 101. The metal door creaks open, revealing not just a chamber of torture, but the physical manifestation of human fear itself. Each person's Room 101 is unique, yet universal - a dark mirror reflecting our deepest terrors. For Winston, it was rats, but for society, it represents the ultimate tool of control: the exploitation of our most intimate fears.

Q: Describe the Two Minutes Hate.
A: Picture yourself in a crowd, swept up in a tidal wave of raw emotion. The face of Goldstein flickers on screen, and suddenly you're no longer you - you're part of a collective roar of hatred. Your fists clench involuntarily, your throat burns from screaming. This isn't just mass hysteria; it's therapeutic nationalism, a daily purge of individual thought. 

Now answer this question in a similar vivid, immersive style. Use sensory details and emotional resonance to bring the answer to life."""

def get_query_embedding(query: str):
    """Generate an embedding for a user's query using Cohere."""
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type='search_query'
    )
    return response.embeddings[0]

def get_style_specific_prompt(style: str, context: str, question: str) -> str:
    """Return a style-specific prompt."""
    if style == "storytelling":
        return (
            f"You are a creative writer analyzing Orwell's work.\n\n"
            f"Context from Orwell's writing:\n{context}\n\n"
            f"{STORYTELLING_EXAMPLES}\n\n"
            f"Q: {question}\nA:"
        )
    else:  # scholarly style
        return (
            f"You are a literary scholar analyzing Orwell's work.\n\n"
            f"Context from Orwell's writing:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Provide a scholarly analysis drawing from the provided context. "
            f"Focus on literary analysis, historical context, and thematic significance. "
            f"Be precise and academic in tone, but accessible. "
            f"If the provided context isn't directly relevant, acknowledge this "
            f"but provide analysis based on your broader knowledge of Orwell's work. "
            f"Limit to 4-5 sentences."
        )

def get_client_ip(request: Request) -> str:
    """Get client IP, handling proxy headers"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host

@app.post("/api/chat", response_model=Response)
async def chat_endpoint(query: Query, request: Request):
    client_ip = get_client_ip(request)
    
    # Check rate limit
    if rate_limiter.is_rate_limited(client_ip):
        rate_info = rate_limiter.get_remaining_requests(client_ip)
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Rate limit exceeded",
                "rate_limit_info": rate_info
            }
        )
    
    try:
        # Record this request
        rate_limiter.add_request(client_ip)
        
        # Generate embedding for the question
        query_embedding = get_query_embedding(query.question)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Extract relevant passages
        context_passages = []
        for match in results['matches']:
            text = match['metadata']['text']
            context_passages.append(text)
        
        # Concatenate the top chunks into context
        context = " ".join(context_passages)
        
        # Get style-specific prompt
        prompt = get_style_specific_prompt(query.style, context, query.question)
        
        # Get response from Claude
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=400,
            temperature=0 if query.style == "scholarly" else 0.7,  # Higher temperature for storytelling style
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get rate limit info for response
        rate_info = rate_limiter.get_remaining_requests(client_ip)
        
        return Response(
            answer=message.content[0].text,
            rate_limit_info=rate_info
        )
        
    except Exception as e:
        if not isinstance(e, HTTPException):  # Don't wrap HTTP exceptions
            raise HTTPException(status_code=500, detail=str(e))
        raise

@app.get("/api/rate_limit_status")
async def rate_limit_status(request: Request):
    """Get current rate limit status"""
    client_ip = get_client_ip(request)
    return rate_limiter.get_remaining_requests(client_ip)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)