#!/usr/bin/env python3
"""
Fast Quickstart for Multi-Agent System
Optimized for quick responses and limited hardware
"""

import asyncio
import time
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

class FastMultiAgentSystem:
    """Simplified, fast multi-agent system for quick demos"""
    
    def __init__(self, model: str = "huihui_ai/qwen3-abliterated:0.6b"):
        """Initialize with the smallest, fastest model by default"""
        self.llm = ChatOllama(
            model=model,
            temperature=0.7,
            num_ctx=2048,  # Smaller context for speed
            num_thread=8,
            timeout=30  # 30 second timeout
        )
        self.max_iterations = 3  # Limit iterations to prevent long runs
        
    async def run_simple(self, task: str) -> str:
        """Direct execution without complex routing"""
        system_prompt = """You are a helpful AI assistant. Provide clear, concise answers.
        Be direct and to the point. No need for extensive explanations unless asked."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"input": task})
        return response.content
    
    async def run_with_agents(self, task: str, agent_type: str = "general") -> Dict[str, Any]:
        """Run with a specific agent type, but limited iterations"""
        
        agent_prompts = {
            "coder": """You are a coding expert. Write clean, efficient code.
            Provide working examples with brief explanations.
            Focus on practical solutions.""",
            
            "creative": """You are a creative thinker. Generate innovative ideas.
            Be imaginative but practical. Keep responses engaging but concise.""",
            
            "analyst": """You are a data analyst. Provide insights and analysis.
            Use clear reasoning. Be thorough but concise.""",
            
            "general": """You are a helpful assistant. Answer clearly and directly.
            Provide accurate information efficiently."""
        }
        
        system_prompt = agent_prompts.get(agent_type, agent_prompts["general"])
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task)
        ]
        
        result = {
            "task": task,
            "agent": agent_type,
            "response": "",
            "iterations": 0,
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Single pass execution - no routing loops
            response = await self.llm.ainvoke(messages)
            result["response"] = response.content
            result["iterations"] = 1
            
        except asyncio.TimeoutError:
            result["response"] = "Task timed out. Please try a simpler query."
        except Exception as e:
            result["response"] = f"Error: {str(e)}"
        
        result["execution_time"] = time.time() - start_time
        return result
    
    async def run_debate_fast(self, topic: str) -> str:
        """Quick debate simulation without multiple agent hops"""
        debate_prompt = f"""Provide a brief, balanced discussion on: {topic}
        
        Present two contrasting viewpoints in a structured format:
        1. Argument FOR (2-3 points)
        2. Argument AGAINST (2-3 points)
        3. Brief conclusion
        
        Keep it concise and engaging."""
        
        return await self.run_simple(debate_prompt)

async def test_code_generation():
    """Test code generation"""
    console.print("\n[yellow]Test: Code Generation[/yellow]")
    system = FastMultiAgentSystem()
    
    task = "Write a Python function to calculate factorial"
    result = await system.run_with_agents(task, "coder")
    
    console.print(f"Task: {task}")
    console.print(f"Response time: {result['execution_time']:.2f}s")
    console.print(Panel(result['response'][:500], title="Code Output"))

async def test_creative():
    """Test creative writing"""
    console.print("\n[yellow]Test: Creative Writing[/yellow]")
    system = FastMultiAgentSystem()
    
    task = "Write a haiku about artificial intelligence"
    result = await system.run_with_agents(task, "creative")
    
    console.print(f"Task: {task}")
    console.print(f"Response time: {result['execution_time']:.2f}s")
    console.print(Panel(result['response'], title="Creative Output"))

async def test_debate():
    """Test quick debate"""
    console.print("\n[yellow]Test: Quick Debate[/yellow]")
    system = FastMultiAgentSystem()
    
    topic = "Is AI beneficial for humanity?"
    response = await system.run_debate_fast(topic)
    
    console.print(f"Topic: {topic}")
    console.print(Panel(response[:800], title="Debate Output"))

async def interactive_mode():
    """Interactive mode for custom queries"""
    console.print("\n[cyan]Interactive Mode - Fast Responses[/cyan]")
    console.print("Type 'exit' to quit\n")
    
    system = FastMultiAgentSystem()
    
    while True:
        task = console.input("[green]Enter your task: [/green]")
        
        if task.lower() in ['exit', 'quit']:
            break
        
        # Determine agent type based on keywords
        agent_type = "general"
        if any(word in task.lower() for word in ['code', 'program', 'function', 'script']):
            agent_type = "coder"
        elif any(word in task.lower() for word in ['story', 'creative', 'imagine', 'poem']):
            agent_type = "creative"
        elif any(word in task.lower() for word in ['analyze', 'data', 'statistics']):
            agent_type = "analyst"
        
        console.print(f"\n[dim]Using {agent_type} agent...[/dim]")
        
        with console.status("Processing...", spinner="dots"):
            start = time.time()
            result = await system.run_with_agents(task, agent_type)
            elapsed = time.time() - start
        
        console.print(f"\n[green]Response ({elapsed:.1f}s):[/green]")
        console.print(Panel(result['response'], border_style="blue"))

async def main():
    """Main entry point"""
    console.print(Panel.fit(
        "[bold cyan]Fast Multi-Agent System[/bold cyan]\n"
        "Optimized for quick responses",
        border_style="blue"
    ))
    
    # Show available models
    table = Table(title="Available Models (Fastest First)")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Speed", style="yellow")
    
    table.add_row("huihui_ai/qwen3-abliterated:0.6b", "0.6B", "⚡⚡⚡ Fastest")
    table.add_row("huihui_ai/gemma3-abliterated:1b", "1B", "⚡⚡ Very Fast")
    table.add_row("huihui_ai/llama3.2-abliterate:1b", "1B", "⚡⚡ Fast")
    table.add_row("huihui_ai/falcon3-abliterated:1b", "1B", "⚡⚡ Fast")
    table.add_row("huihui_ai/qwen3-abliterated:1.7b", "1.7B", "⚡ Good")
    
    console.print(table)
    
    while True:
        console.print("\n[yellow]Choose an option:[/yellow]")
        console.print("1. Quick Test (Simple task)")
        console.print("2. Code Generation Demo")
        console.print("3. Creative Writing Demo")
        console.print("4. Quick Debate Demo")
        console.print("5. Interactive Mode")
        console.print("0. Exit")
        
        choice = console.input("\n[cyan]Enter choice: [/cyan]")
        
        if choice == "0":
            console.print("[green]Goodbye! 👋[/green]")
            break
        elif choice == "1":
            system = FastMultiAgentSystem()
            task = "What is 2+2 and why?"
            console.print(f"\nTask: {task}")
            with console.status("Processing...", spinner="dots"):
                start = time.time()
                response = await system.run_simple(task)
                elapsed = time.time() - start
            console.print(f"Response ({elapsed:.1f}s): {response}")
        elif choice == "2":
            await test_code_generation()
        elif choice == "3":
            await test_creative()
        elif choice == "4":
            await test_debate()
        elif choice == "5":
            await interactive_mode()
        else:
            console.print("[red]Invalid choice[/red]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
