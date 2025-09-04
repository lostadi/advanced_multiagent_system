#!/usr/bin/env python3
"""
Ultra-Fast Multi-Agent System
Maximum speed optimizations for limited hardware
"""

import asyncio
import time
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
import re

class UltraFastSystem:
    """Ultra-optimized system for maximum speed"""
    
    def __init__(self, model: str = "huihui_ai/qwen3-abliterated:0.6b"):
        """Initialize with optimizations for speed"""
        self.llm = ChatOllama(
            model=model,
            temperature=0.3,  # Lower temperature for faster, more deterministic responses
            num_ctx=1024,     # Minimal context window
            num_thread=8,
            timeout=20,       # 20 second timeout
            top_k=10,        # Limit sampling for speed
            top_p=0.9,       # Focused sampling
            repeat_penalty=1.1,
            num_predict=256   # Limit response length
        )
    
    def clean_response(self, text: str) -> str:
        """Remove thinking tags and clean up response"""
        # Remove <think>...</think> tags and content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    async def quick_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Get a quick, direct response"""
        messages = [
            SystemMessage(content="You are a direct, concise assistant. Answer in 1-2 sentences maximum. No explanations unless asked."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=10.0  # 10 second hard timeout
            )
            return self.clean_response(response.content)[:max_tokens]
        except asyncio.TimeoutError:
            return "Response timed out. Try a simpler question."
        except Exception as e:
            return f"Error: {str(e)[:100]}"
    
    async def code_snippet(self, task: str) -> str:
        """Generate code quickly"""
        prompt = f"""Write only the code, no explanations:
{task}

Code:"""
        
        messages = [
            SystemMessage(content="You are a code generator. Output only clean, working code. No comments unless essential."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=15.0
            )
            return self.clean_response(response.content)
        except asyncio.TimeoutError:
            return "# Code generation timed out"
        except Exception as e:
            return f"# Error: {str(e)}"

async def speed_test():
    """Run a speed test"""
    system = UltraFastSystem()
    
    tests = [
        ("Simple Math", "What is 5 * 7?"),
        ("Definition", "Define recursion in one sentence"),
        ("Code", "Write a Python function to reverse a string")
    ]
    
    print("\n🚀 SPEED TEST RESULTS\n" + "="*40)
    
    for test_name, prompt in tests:
        start = time.time()
        
        if "Code" in test_name:
            result = await system.code_snippet(prompt)
        else:
            result = await system.quick_response(prompt)
        
        elapsed = time.time() - start
        
        print(f"\n{test_name} ({elapsed:.1f}s):")
        print(f"  {result[:200]}")
        if len(result) > 200:
            print("  ...")

async def interactive_fast():
    """Ultra-fast interactive mode"""
    system = UltraFastSystem()
    
    print("\n⚡ ULTRA-FAST MODE ⚡")
    print("Type 'exit' to quit\n")
    
    while True:
        prompt = input("You: ").strip()
        
        if prompt.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not prompt:
            continue
        
        start = time.time()
        
        # Detect if it's a code request
        if any(word in prompt.lower() for word in ['code', 'function', 'write', 'program']):
            response = await system.code_snippet(prompt)
        else:
            response = await system.quick_response(prompt)
        
        elapsed = time.time() - start
        
        print(f"\nAI ({elapsed:.1f}s): {response}\n")

async def benchmark_models():
    """Benchmark different models"""
    models = [
        "huihui_ai/qwen3-abliterated:0.6b",
        "huihui_ai/gemma3-abliterated:1b",
        "huihui_ai/llama3.2-abliterate:1b"
    ]
    
    test_prompt = "What is the capital of France?"
    
    print("\n📊 MODEL BENCHMARK\n" + "="*40)
    
    for model in models:
        print(f"\nTesting {model}...")
        system = UltraFastSystem(model)
        
        start = time.time()
        result = await system.quick_response(test_prompt)
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Response: {result[:100]}")

async def main():
    """Main menu"""
    print("""
╔══════════════════════════════════════╗
║     ULTRA-FAST AI SYSTEM             ║
║     Optimized for Speed              ║
╚══════════════════════════════════════╝
    """)
    
    while True:
        print("\nOptions:")
        print("1. Speed Test (run benchmarks)")
        print("2. Interactive Mode (chat)")
        print("3. Model Comparison")
        print("0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            await speed_test()
        elif choice == "2":
            await interactive_fast()
        elif choice == "3":
            await benchmark_models()
        else:
            print("Invalid choice")

if __name__ == "__main__":
    try:
        # Check if Ollama is running
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Starting Ollama service...")
            subprocess.run(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
