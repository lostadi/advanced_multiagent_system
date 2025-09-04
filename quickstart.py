#!/usr/bin/env python3
"""
Quick start script for testing the Advanced Multi-Agent System
Run this to see the system in action with various examples
"""

import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import time

# Import our multi-agent system
from multi_agent_system import (
    AdvancedMultiAgentSystem,
    CreativeInteractionManager,
    EntropyCollector
)

console = Console()

def print_header():
    """Print a nice header"""
    console.print(Panel.fit(
        "[bold cyan]Advanced LangChain + Ollama Multi-Agent System[/bold cyan]\n"
        "[dim]Production-ready implementation with code execution and human-in-the-loop[/dim]",
        border_style="cyan"
    ))

async def test_simple_task(system):
    """Test a simple task"""
    console.print("\n[bold green]Test 1: Simple Task[/bold green]")
    console.print("[dim]Asking the system to explain recursion...[/dim]\n")
    
    result = await system.run(
        "Explain recursion with a simple Python example"
    )
    
    console.print(f"✅ Completed with {len(result['messages'])} messages")
    
    # Show last message
    if result['messages']:
        last_msg = result['messages'][-1].content
        console.print(Panel(
            last_msg[:500] + "..." if len(last_msg) > 500 else last_msg,
            title="Agent Response",
            border_style="green"
        ))

async def test_code_generation(system):
    """Test code generation with execution"""
    console.print("\n[bold yellow]Test 2: Code Generation[/bold yellow]")
    console.print("[dim]Creating a simple calculator function...[/dim]\n")
    
    result = await system.run(
        "Write a Python function that calculates the factorial of a number. "
        "Include error handling and a test example."
    )
    
    console.print(f"✅ Completed with {len(result['messages'])} messages")
    
    # Check if code was executed
    if result.get('execution_history'):
        console.print(f"🖥️  Executed {len(result['execution_history'])} code blocks")
        for exec_result in result['execution_history']:
            status = "✓" if exec_result.get('success') else "✗"
            console.print(f"   {status} Execution time: {exec_result.get('execution_time', 0):.2f}s")

async def test_debate(system):
    """Test debate interaction pattern"""
    console.print("\n[bold magenta]Test 3: Agent Debate[/bold magenta]")
    console.print("[dim]Running a debate on programming paradigms...[/dim]\n")
    
    interaction_manager = CreativeInteractionManager(system)
    
    result = await interaction_manager.run_debate(
        topic="Object-Oriented vs Functional Programming",
        positions=[
            "OOP is better for large-scale applications",
            "FP leads to more maintainable and testable code"
        ]
    )
    
    console.print(f"✅ Debate completed with {len(result['messages'])} exchanges")

async def test_storytelling(system):
    """Test creative storytelling"""
    console.print("\n[bold blue]Test 4: Creative Storytelling[/bold blue]")
    console.print("[dim]Generating a short story collaboratively...[/dim]\n")
    
    interaction_manager = CreativeInteractionManager(system)
    
    result = await interaction_manager.run_storytelling(
        premise="A programmer discovers their code is sentient",
        genre="tech thriller"
    )
    
    console.print(f"✅ Story created with {len(result['messages'])} contributions")

def test_entropy_collection():
    """Test entropy collection system"""
    console.print("\n[bold cyan]Test 5: Entropy Collection[/bold cyan]")
    console.print("[dim]Testing entropy generation from various sources...[/dim]\n")
    
    collector = EntropyCollector()
    
    # Simulate mouse movements
    mock_mouse_events = [
        (100, 100, 0.0),
        (150, 120, 0.1),
        (200, 180, 0.2),
        (180, 200, 0.3),
    ]
    
    mouse_entropy = collector.collect_mouse_entropy(mock_mouse_events)
    console.print(f"🎲 Mouse entropy: {mouse_entropy:.4f}")
    
    # Test circadian entropy
    circadian_entropy = collector.get_circadian_entropy()
    console.print(f"🕐 Circadian entropy: {circadian_entropy:.4f}")
    
    # Simulate keystroke dynamics
    mock_keystrokes = [
        {'press_time': 0.0, 'release_time': 0.05},
        {'press_time': 0.1, 'release_time': 0.18},
        {'press_time': 0.25, 'release_time': 0.30},
    ]
    
    keystroke_dynamics = collector.collect_keystroke_dynamics(mock_keystrokes)
    console.print(f"⌨️  Keystroke entropy: {keystroke_dynamics['entropy_value']:.4f}")

async def interactive_mode(system):
    """Interactive mode for custom tasks"""
    console.print("\n[bold white]Interactive Mode[/bold white]")
    console.print("[dim]Enter your own task for the multi-agent system[/dim]\n")
    
    task = console.input("[cyan]Enter your task:[/cyan] ")
    
    if task.strip():
        console.print("\n[dim]Processing your request...[/dim]")
        result = await system.run(task)
        
        console.print(f"\n✅ Task completed with {len(result['messages'])} messages")
        
        # Show the final response
        if result['messages']:
            last_msg = result['messages'][-1].content
            console.print(Panel(
                last_msg[:800] + "..." if len(last_msg) > 800 else last_msg,
                title="Final Response",
                border_style="cyan"
            ))
            
        # Show collected entropy
        if result.get('entropy_pool'):
            avg_entropy = sum(result['entropy_pool']) / len(result['entropy_pool'])
            console.print(f"\n🎲 Collected {len(result['entropy_pool'])} entropy values")
            console.print(f"   Average: {avg_entropy:.4f}")

async def main():
    """Main execution"""
    print_header()
    
    # Model selection for limited hardware
    console.print("\n[bold]Available Small Models for Limited Hardware:[/bold]")
    models = [
        ("huihui_ai/qwen3-abliterated:1.7b", "1.7B params - Best balance"),
        ("huihui_ai/llama3.2-abliterate:1b", "1B params - Good quality"),
        ("huihui_ai/falcon3-abliterated:1b", "1B params - Fast"),
        ("huihui_ai/gemma3-abliterated:1b", "1B params - Efficient"),
        ("huihui_ai/qwen3-abliterated:0.6b", "0.6B params - Fastest")
    ]
    
    console.print("\n[cyan]Select a model:[/cyan]")
    for i, (model, desc) in enumerate(models, 1):
        console.print(f"{i}. {model} - {desc}")
    
    model_choice = console.input("\n[cyan]Enter choice (1-5, default=1):[/cyan] ").strip() or "1"
    
    try:
        model_index = int(model_choice) - 1
        if 0 <= model_index < len(models):
            selected_model = models[model_index][0]
        else:
            selected_model = models[0][0]
    except:
        selected_model = models[0][0]
    
    # Check if Ollama is available
    console.print(f"\n[bold]Initializing System with {selected_model}...[/bold]")
    
    try:
        # Initialize the multi-agent system
        console.print("📦 Loading Ollama model (this may take a moment)...")
        system = AdvancedMultiAgentSystem(model=selected_model, use_local=True)
        console.print("✅ System initialized successfully!\n")
        
    except Exception as e:
        console.print(f"[red]❌ Error initializing system: {e}[/red]")
        console.print("\n[yellow]Make sure:[/yellow]")
        console.print("1. Ollama is installed and running: [cyan]ollama serve[/cyan]")
        console.print(f"2. You've pulled the model: [cyan]ollama pull {selected_model}[/cyan]")
        console.print("3. All dependencies are installed: [cyan]pip install -r requirements.txt[/cyan]")
        console.print("\n[dim]To pull all small models, run:[/dim]")
        for model, _ in models:
            console.print(f"[dim]ollama pull {model}[/dim]")
        sys.exit(1)
    
    # Menu
    while True:
        console.print("\n[bold]Choose an option:[/bold]")
        console.print("1. Run simple task test")
        console.print("2. Test code generation")
        console.print("3. Run agent debate")
        console.print("4. Create collaborative story")
        console.print("5. Test entropy collection")
        console.print("6. Interactive mode (custom task)")
        console.print("0. Exit")
        
        choice = console.input("\n[cyan]Enter choice (0-6):[/cyan] ")
        
        try:
            if choice == "0":
                console.print("\n[bold green]Goodbye! 👋[/bold green]")
                break
            elif choice == "1":
                await test_simple_task(system)
            elif choice == "2":
                await test_code_generation(system)
            elif choice == "3":
                await test_debate(system)
            elif choice == "4":
                await test_storytelling(system)
            elif choice == "5":
                test_entropy_collection()
            elif choice == "6":
                await interactive_mode(system)
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Program interrupted[/yellow]")
        sys.exit(0)
