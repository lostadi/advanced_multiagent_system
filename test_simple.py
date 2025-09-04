#!/usr/bin/env python3
"""
Simple test script to verify the multi-agent system works with small models
"""

import asyncio
from multi_agent_system import AdvancedMultiAgentSystem
from rich.console import Console

console = Console()

async def test_basic():
    """Test basic functionality with small models"""
    
    console.print("[bold cyan]Testing Advanced Multi-Agent System with Small Models[/bold cyan]\n")
    
    # Test different small models
    models_to_test = [
        "huihui_ai/qwen3-abliterated:1.7b",  # Best balance
        "huihui_ai/qwen3-abliterated:0.6b",  # Fastest
    ]
    
    for model in models_to_test:
        console.print(f"\n[yellow]Testing model: {model}[/yellow]")
        console.print("-" * 50)
        
        try:
            # Initialize system with small model
            system = AdvancedMultiAgentSystem(model=model, use_local=True)
            console.print(f"✅ System initialized with {model}")
            
            # Simple test task
            task = "Write a simple Python function to add two numbers"
            console.print(f"\n[cyan]Task:[/cyan] {task}")
            
            # Run the task
            result = await system.run(task)
            
            # Display results
            console.print(f"\n[green]✅ Task completed![/green]")
            console.print(f"Messages exchanged: {len(result['messages'])}")
            
            # Show last response
            if result['messages']:
                last_msg = result['messages'][-1].content
                console.print(f"\n[dim]Last response preview:[/dim]")
                console.print(last_msg[:300] + "..." if len(last_msg) > 300 else last_msg)
            
            # Show entropy if collected
            if result.get('entropy_pool'):
                console.print(f"\nEntropy values collected: {len(result['entropy_pool'])}")
                
        except Exception as e:
            console.print(f"[red]❌ Error with model {model}: {e}[/red]")
            continue
            
        console.print("\n" + "="*50)

if __name__ == "__main__":
    try:
        console.print("[bold]Starting test...[/bold]\n")
        asyncio.run(test_basic())
        console.print("\n[bold green]All tests completed! 🎉[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Test failed: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
