#!/usr/bin/env python3
"""
Part 4 (Bonus): Inference server for the fine-tuned Qwen model.

This script creates a FastAPI server for serving the fine-tuned model using vLLM for efficient inference.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Fallback to transformers
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for code generation."""
    problem: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    stop_sequences: Optional[List[str]] = None

class GenerationResponse(BaseModel):
    """Response model for code generation."""
    solution: str
    metadata: Dict

@dataclass
class ServerConfig:
    """Configuration for inference server."""
    model_path: str = "models/qwen-7b-finetuned"
    base_model_name: str = "Qwen/Qwen-7B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

class VLLMInferenceEngine:
    """Inference engine using vLLM."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Install with: pip install vllm")
        
        logger.info(f"Loading model with vLLM: {config.model_path}")
        
        # Initialize vLLM model
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True
        )
        
        logger.info("vLLM model loaded successfully")
    
    def generate(self, request: GenerationRequest) -> str:
        """Generate code solution using vLLM."""
        # Format prompt
        prompt = self._format_prompt(request.problem)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop_sequences or ["<|im_end|>", "```\n\n"]
        )
        
        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Extract code
        solution = self._extract_code(generated_text)
        
        return solution
    
    def _format_prompt(self, problem: str) -> str:
        """Format problem as prompt."""
        return f"""<|im_start|>system
You are an expert competitive programmer. Solve the following problem with a complete Python solution.<|im_end|>
<|im_start|>user
{problem}

Write a complete Python solution. Your code should read from standard input and write to standard output.
<|im_end|>
<|im_start|>assistant
I'll solve this step by step.

```python
"""
    
    def _extract_code(self, generated_text: str) -> str:
        """Extract code from generated text."""
        # Look for code blocks
        if "```python" in generated_text:
            start = generated_text.find("```python") + 9
            end = generated_text.find("```", start)
            if end != -1:
                return generated_text[start:end].strip()
        
        # If no explicit code block, take everything up to a stop sequence
        for stop in ["<|im_end|>", "\n\n```"]:
            if stop in generated_text:
                generated_text = generated_text.split(stop)[0]
        
        return generated_text.strip()

class TransformersInferenceEngine:
    """Fallback inference engine using transformers."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        
        logger.info(f"Loading model with transformers: {config.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load fine-tuned weights
        self.model = PeftModel.from_pretrained(base_model, config.model_path)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Transformers model loaded successfully")
    
    def generate(self, request: GenerationRequest) -> str:
        """Generate code solution using transformers."""
        # Format prompt
        prompt = self._format_prompt(request.problem)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract code
        solution = self._extract_code(generated_text)
        
        return solution
    
    def _format_prompt(self, problem: str) -> str:
        """Format problem as prompt."""
        return f"""<|im_start|>system
You are an expert competitive programmer. Solve the following problem with a complete Python solution.<|im_end|>
<|im_start|>user
{problem}

Write a complete Python solution. Your code should read from standard input and write to standard output.
<|im_end|>
<|im_start|>assistant
I'll solve this step by step.

```python
"""
    
    def _extract_code(self, generated_text: str) -> str:
        """Extract code from generated text."""
        # Look for code blocks
        if "```python" in generated_text:
            start = generated_text.find("```python") + 9
            end = generated_text.find("```", start)
            if end != -1:
                return generated_text[start:end].strip()
        elif "```" in generated_text:
            # Look for any code block
            start = generated_text.find("```") + 3
            end = generated_text.find("```", start)
            if end != -1:
                return generated_text[start:end].strip()
        
        # If no code block, return everything up to stop sequences
        for stop in ["<|im_end|>", "\n\nProblem", "\n\n#"]:
            if stop in generated_text:
                generated_text = generated_text.split(stop)[0]
        
        return generated_text.strip()

# Initialize FastAPI app
app = FastAPI(
    title="Qwen-7B Code Generation API",
    description="API for generating code solutions using fine-tuned Qwen-7B model",
    version="1.0.0"
)

# Global inference engine
inference_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup."""
    global inference_engine
    
    config = ServerConfig()
    
    # Check if model exists
    if not Path(config.model_path).exists():
        logger.error(f"Model not found at {config.model_path}")
        raise RuntimeError("Fine-tuned model not found")
    
    # Try to load with vLLM first, fallback to transformers
    try:
        if VLLM_AVAILABLE:
            inference_engine = VLLMInferenceEngine(config)
            logger.info("Using vLLM for inference")
        else:
            inference_engine = TransformersInferenceEngine(config)
            logger.info("Using transformers for inference")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Try fallback
        try:
            inference_engine = TransformersInferenceEngine(config)
            logger.info("Using transformers fallback for inference")
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            raise RuntimeError("Could not initialize inference engine")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Qwen-7B Code Generation API",
        "status": "running",
        "engine": "vLLM" if VLLM_AVAILABLE and isinstance(inference_engine, VLLMInferenceEngine) else "transformers"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_code(request: GenerationRequest):
    """Generate code solution for a given problem."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate solution
        solution = inference_engine.generate(request)
        
        # Prepare response
        response = GenerationResponse(
            solution=solution,
            metadata={
                "problem_length": len(request.problem),
                "solution_length": len(solution),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "engine": "vLLM" if isinstance(inference_engine, VLLMInferenceEngine) else "transformers"
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/batch")
async def generate_batch(requests: List[GenerationRequest]):
    """Generate solutions for multiple problems."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(requests) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 10)")
    
    try:
        responses = []
        for request in requests:
            solution = inference_engine.generate(request)
            responses.append(GenerationResponse(
                solution=solution,
                metadata={
                    "problem_length": len(request.problem),
                    "solution_length": len(solution),
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "engine": "vLLM" if isinstance(inference_engine, VLLMInferenceEngine) else "transformers"
                }
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

def main():
    """Run the inference server."""
    config = ServerConfig()
    
    logger.info(f"Starting server on {config.host}:{config.port}")
    logger.info(f"Model path: {config.model_path}")
    
    uvicorn.run(
        "inference_server:app",
        host=config.host,
        port=config.port,
        reload=False,
        workers=1  # Important: only use 1 worker with GPU models
    )

if __name__ == "__main__":
    main()