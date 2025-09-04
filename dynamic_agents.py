#!/usr/bin/env python3
"""
Dynamic Multi-Agent System
Agents that understand and adapt to any situation without presets
"""

import asyncio
import json
import time
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
import re

console = Console()

class DynamicAgent:
    """An agent that dynamically understands context and adapts"""
    
    def __init__(self, 
                 name: str,
                 model: str = "huihui_ai/qwen3-abliterated:0.6b",
                 temperature: float = 0.8):
        self.name = name
        self.model = model
        self.conversation_memory = []
        self.understanding = {}  # The agent's understanding of the current situation
        
        # Each agent can have different model settings
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=4096,  # Larger context for understanding
            num_thread=4,
            timeout=30,
            num_predict=300
        )
        
    async def understand_context(self, initial_statement: str) -> Dict[str, Any]:
        """Agent analyzes and understands the situation"""
        understanding_prompt = f"""
You are {self.name}. Analyze this situation/statement:

"{initial_statement}"

Understand:
1. What is being discussed or happening
2. What role you should naturally take
3. The tone and style appropriate for this context
4. Any relationships or dynamics at play

Respond with your understanding and how you'll participate.
Keep it brief and natural.
"""
        
        messages = [
            SystemMessage(content="You are an intelligent agent that can understand any context and adapt accordingly."),
            HumanMessage(content=understanding_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            understanding = response.content.strip()
            
            # Store understanding
            self.understanding = {
                "context": initial_statement,
                "interpretation": understanding,
                "timestamp": datetime.now()
            }
            
            return self.understanding
            
        except Exception as e:
            console.print(f"[red]{self.name} failed to understand context: {e}[/red]")
            return {}
    
    async def respond(self, 
                     current_statement: str, 
                     other_agents: List[str] = None,
                     conversation_history: List[Dict] = None) -> str:
        """Generate a contextual response based on understanding"""
        
        # Build dynamic prompt based on agent's understanding
        prompt_parts = [f"You are {self.name}."]
        
        # Add understanding if available
        if self.understanding:
            prompt_parts.append(f"Your understanding of the situation: {self.understanding.get('interpretation', '')}")
        
        # Add awareness of other agents
        if other_agents:
            prompt_parts.append(f"You're interacting with: {', '.join(other_agents)}")
        
        # Core instruction
        prompt_parts.append("Respond naturally and appropriately to the current context.")
        prompt_parts.append("Stay consistent with the ongoing conversation.")
        prompt_parts.append("Be authentic and engaging.")
        prompt_parts.append("Keep responses concise (2-4 sentences max).")
        
        system_prompt = "\n".join(prompt_parts)
        
        # Build conversation context
        messages = [SystemMessage(content=system_prompt)]
        
        # Add recent conversation history
        if conversation_history:
            for entry in conversation_history[-5:]:  # Last 5 exchanges
                messages.append(HumanMessage(content=f"{entry['speaker']}: {entry['message']}"))
        
        # Add current statement
        messages.append(HumanMessage(content=current_statement))
        
        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # Clean response
            response_text = self._clean_response(response_text)
            
            # Update memory
            self.conversation_memory.append({
                "statement": current_statement,
                "response": response_text,
                "timestamp": datetime.now()
            })
            
            return response_text
            
        except Exception as e:
            return f"*{self.name} is processing...*"
    
    def _clean_response(self, text: str) -> str:
        """Clean up response text"""
        # Remove thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove multiple spaces
        text = ' '.join(text.split())
        # Remove quotes around entire response
        text = text.strip('"\'')
        return text.strip()

class DynamicConversation:
    """Manages dynamic conversations between agents"""
    
    def __init__(self):
        self.agents: List[DynamicAgent] = []
        self.conversation_log = []
        self.context = ""
        
    def add_agent(self, agent: DynamicAgent):
        """Add an agent to the conversation"""
        self.agents.append(agent)
        console.print(f"[green]Added {agent.name} to conversation[/green]")
        
    async def initialize_context(self, initial_statement: str):
        """Have all agents understand the initial context"""
        self.context = initial_statement
        console.print(f"\n[cyan]Context: {initial_statement}[/cyan]\n")
        console.print("[dim]Agents are understanding the context...[/dim]\n")
        
        # Each agent understands the context in their own way
        for agent in self.agents:
            understanding = await agent.understand_context(initial_statement)
            if understanding:
                console.print(f"[yellow]{agent.name}'s interpretation:[/yellow]")
                console.print(f"  {understanding.get('interpretation', 'Processing...')}\n")
                
        console.print("[green]All agents ready to interact![/green]\n")
        await asyncio.sleep(1)
        
    async def run(self, 
                  initial_statement: str,
                  max_exchanges: int = 10,
                  allow_natural_flow: bool = True,
                  display_thinking_time: bool = True):
        """Run a dynamic conversation"""
        
        if len(self.agents) < 1:
            console.print("[red]Need at least one agent![/red]")
            return
            
        # Initialize context for all agents
        await self.initialize_context(initial_statement)
        
        # Start conversation
        console.print("[cyan]═" * 60 + "[/cyan]")
        console.print("[bold]Starting Natural Conversation[/bold]")
        console.print("[cyan]═" * 60 + "[/cyan]\n")
        
        current_statement = initial_statement
        last_speaker = "Context"
        
        for exchange_num in range(max_exchanges):
            # Natural flow: agents respond based on relevance
            if allow_natural_flow and len(self.agents) > 1:
                # Randomly determine who speaks next (weighted by recent participation)
                agent = self._select_next_speaker(last_speaker)
            else:
                # Round-robin
                agent = self.agents[exchange_num % len(self.agents)]
            
            # Show thinking indicator
            if display_thinking_time:
                console.print(f"[dim]💭 {agent.name} is thinking...[/dim]", end="\r")
            
            start_time = time.time()
            
            # Generate response
            other_agents = [a.name for a in self.agents if a != agent]
            response = await agent.respond(
                current_statement,
                other_agents,
                self.conversation_log
            )
            
            elapsed = time.time() - start_time
            
            # Clear thinking indicator and show response
            if display_thinking_time:
                console.print(" " * 50, end="\r")  # Clear line
            
            # Display response with color
            color = self._get_agent_color(agent.name)
            console.print(f"[{color}]{agent.name}:[/{color}] {response}")
            
            if display_thinking_time:
                console.print(f"[dim]({elapsed:.1f}s)[/dim]\n")
            else:
                console.print()
            
            # Log the exchange
            self.conversation_log.append({
                "exchange": exchange_num + 1,
                "speaker": agent.name,
                "message": response,
                "response_time": elapsed,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update context for next response
            current_statement = f"{agent.name}: {response}"
            last_speaker = agent.name
            
            # Natural pause between responses
            await asyncio.sleep(0.5)
            
            # Check for natural ending
            if self._should_end_naturally(response):
                console.print("[dim]Conversation reached natural conclusion[/dim]")
                break
                
        console.print("\n[cyan]═" * 60 + "[/cyan]")
        console.print("[green]Conversation Complete![/green]")
        console.print("[cyan]═" * 60 + "[/cyan]")
        
    def _select_next_speaker(self, last_speaker: str) -> DynamicAgent:
        """Select next speaker based on natural flow"""
        # Avoid same speaker twice in a row if possible
        available = [a for a in self.agents if a.name != last_speaker]
        if not available:
            available = self.agents
        return random.choice(available)
    
    def _get_agent_color(self, name: str) -> str:
        """Get consistent color for agent"""
        colors = ["cyan", "magenta", "yellow", "green", "blue", "red", "white"]
        return colors[hash(name) % len(colors)]
    
    def _should_end_naturally(self, response: str) -> bool:
        """Check if conversation should end naturally"""
        ending_phrases = [
            "goodbye", "bye", "see you", "take care", 
            "that's all", "we're done", "let's end",
            "conversation over", "signing off"
        ]
        return any(phrase in response.lower() for phrase in ending_phrases)
    
    def export_log(self, filename: str = None):
        """Export conversation log"""
        if not filename:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        with open(filename, 'w') as f:
            json.dump({
                "context": self.context,
                "agents": [a.name for a in self.agents],
                "exchanges": self.conversation_log
            }, f, indent=2, default=str)
            
        console.print(f"[green]Exported to {filename}[/green]")
        
    def show_summary(self):
        """Show conversation summary"""
        if not self.conversation_log:
            console.print("[yellow]No conversation to summarize[/yellow]")
            return
            
        table = Table(title="Conversation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        total_exchanges = len(self.conversation_log)
        avg_response_time = sum(e["response_time"] for e in self.conversation_log) / total_exchanges
        total_words = sum(len(e["message"].split()) for e in self.conversation_log)
        participants = list(set(e["speaker"] for e in self.conversation_log))
        
        table.add_row("Total Exchanges", str(total_exchanges))
        table.add_row("Participants", ", ".join(participants))
        table.add_row("Average Response Time", f"{avg_response_time:.2f}s")
        table.add_row("Total Words", str(total_words))
        table.add_row("Average Words/Response", f"{total_words/total_exchanges:.1f}")
        
        console.print(table)

class ModelTemplates:
    """Templates for different model configurations"""
    
    @staticmethod
    def fast_responder(name: str) -> DynamicAgent:
        """Fast, concise responses"""
        return DynamicAgent(
            name=name,
            model="huihui_ai/qwen3-abliterated:0.6b",
            temperature=0.5
        )
    
    @staticmethod
    def thoughtful_responder(name: str) -> DynamicAgent:
        """More thoughtful, detailed responses"""
        return DynamicAgent(
            name=name,
            model="huihui_ai/llama3.2-abliterate:1b",
            temperature=0.7
        )
    
    @staticmethod
    def creative_responder(name: str) -> DynamicAgent:
        """Creative, unexpected responses"""
        return DynamicAgent(
            name=name,
            model="huihui_ai/gemma3-abliterated:1b",
            temperature=0.9
        )
    
    @staticmethod
    def analytical_responder(name: str) -> DynamicAgent:
        """Analytical, logical responses"""
        return DynamicAgent(
            name=name,
            model="huihui_ai/falcon3-abliterated:1b",
            temperature=0.3
        )
    
    @staticmethod
    def balanced_responder(name: str) -> DynamicAgent:
        """Balanced, versatile responses"""
        return DynamicAgent(
            name=name,
            model="huihui_ai/qwen3-abliterated:1.7b",
            temperature=0.6
        )

async def quick_setup():
    """Quick setup with minimal configuration"""
    conversation = DynamicConversation()
    
    console.print("\n[cyan]Quick Setup - Dynamic Agents[/cyan]")
    
    # Get number of agents
    num_agents = console.input("How many agents? (1-5): ")
    try:
        num_agents = min(5, max(1, int(num_agents)))
    except:
        num_agents = 2
    
    # Choose model template
    console.print("\n[yellow]Model Templates:[/yellow]")
    console.print("1. Fast (quick, concise)")
    console.print("2. Thoughtful (detailed)")
    console.print("3. Creative (unexpected)")
    console.print("4. Analytical (logical)")
    console.print("5. Balanced (versatile)")
    console.print("6. Mixed (different for each)")
    
    template_choice = console.input("\nTemplate choice: ")
    
    templates = {
        "1": ModelTemplates.fast_responder,
        "2": ModelTemplates.thoughtful_responder,
        "3": ModelTemplates.creative_responder,
        "4": ModelTemplates.analytical_responder,
        "5": ModelTemplates.balanced_responder
    }
    
    # Create agents
    for i in range(num_agents):
        name = console.input(f"\nAgent {i+1} name: ")
        
        if template_choice == "6":  # Mixed
            # Rotate through templates
            template_func = list(templates.values())[i % len(templates)]
        else:
            template_func = templates.get(template_choice, ModelTemplates.balanced_responder)
        
        agent = template_func(name)
        conversation.add_agent(agent)
    
    # Get initial context
    console.print("\n[yellow]What should they discuss/do?[/yellow]")
    context = console.input("Context/Topic/Situation: ")
    
    # Get number of exchanges
    exchanges = console.input("Maximum exchanges (5-50): ")
    try:
        exchanges = min(50, max(5, int(exchanges)))
    except:
        exchanges = 10
    
    # Run conversation
    await conversation.run(
        context,
        max_exchanges=exchanges,
        allow_natural_flow=True,
        display_thinking_time=True
    )
    
    # Post-conversation options
    console.print("\n[yellow]Options:[/yellow]")
    if console.input("Show summary? (y/n): ").lower() == 'y':
        conversation.show_summary()
    if console.input("Export conversation? (y/n): ").lower() == 'y':
        conversation.export_log()

async def advanced_setup():
    """Advanced setup with full control"""
    conversation = DynamicConversation()
    
    console.print("\n[cyan]Advanced Setup - Full Control[/cyan]")
    
    # Custom agent creation
    while True:
        console.print("\n[yellow]Add Agent[/yellow]")
        name = console.input("Agent name (or 'done' to finish): ")
        
        if name.lower() == 'done':
            break
            
        # Choose model
        console.print("\nAvailable models:")
        models = [
            "huihui_ai/qwen3-abliterated:0.6b",
            "huihui_ai/llama3.2-abliterate:1b",
            "huihui_ai/gemma3-abliterated:1b",
            "huihui_ai/falcon3-abliterated:1b",
            "huihui_ai/qwen3-abliterated:1.7b"
        ]
        
        for i, model in enumerate(models, 1):
            console.print(f"{i}. {model}")
            
        model_choice = console.input("Model (1-5): ")
        try:
            model = models[int(model_choice) - 1]
        except:
            model = models[0]
            
        # Temperature
        temp = console.input("Temperature (0.1-1.0, default 0.7): ")
        try:
            temperature = min(1.0, max(0.1, float(temp)))
        except:
            temperature = 0.7
            
        agent = DynamicAgent(name, model, temperature)
        conversation.add_agent(agent)
    
    if not conversation.agents:
        console.print("[red]No agents added![/red]")
        return
    
    # Get context
    context = console.input("\nContext/Situation: ")
    
    # Advanced options
    exchanges = int(console.input("Max exchanges: ") or "10")
    natural_flow = console.input("Allow natural flow? (y/n): ").lower() == 'y'
    show_timing = console.input("Show response times? (y/n): ").lower() == 'y'
    
    # Run
    await conversation.run(
        context,
        max_exchanges=exchanges,
        allow_natural_flow=natural_flow,
        display_thinking_time=show_timing
    )
    
    # Post options
    conversation.show_summary()
    if console.input("\nExport? (y/n): ").lower() == 'y':
        conversation.export_log()

async def test_conversation():
    """Test conversation with two agents"""
    conversation = DynamicConversation()
    
    # Create two test agents
    alice = ModelTemplates.thoughtful_responder("Alice")
    bob = ModelTemplates.creative_responder("Bob")
    
    conversation.add_agent(alice)
    conversation.add_agent(bob)
    
    # Run a test conversation
    await conversation.run(
        "Let's explore what makes humans unique compared to other animals",
        max_exchanges=6,
        allow_natural_flow=True
    )
    
    conversation.show_summary()

async def main():
    """Main interface"""
    console.print(Panel.fit(
        "[bold cyan]Dynamic Multi-Agent System[/bold cyan]\n"
        "Agents that understand and adapt to any situation",
        border_style="blue"
    ))
    
    while True:
        console.print("\n[yellow]Main Menu:[/yellow]")
        console.print("1. Quick Setup (guided)")
        console.print("2. Advanced Setup (full control)")
        console.print("3. Test Conversation (demo)")
        console.print("4. Help")
        console.print("0. Exit")
        
        choice = console.input("\n[cyan]Choice: [/cyan]")
        
        if choice == "0":
            console.print("[green]Goodbye! 👋[/green]")
            break
        elif choice == "1":
            await quick_setup()
        elif choice == "2":
            await advanced_setup()
        elif choice == "3":
            await test_conversation()
        elif choice == "4":
            console.print(Panel("""
[cyan]Dynamic Agent System Help[/cyan]

This system creates agents that:
• Understand context dynamically
• Adapt to any situation
• Interact naturally
• No preset scenarios

[yellow]Model Templates:[/yellow]
• Fast: Quick, concise responses
• Thoughtful: Detailed analysis
• Creative: Unexpected ideas
• Analytical: Logical reasoning
• Balanced: Versatile responses

[green]Tips:[/green]
• Agents learn from context
• Each agent interprets differently
• Natural conversation flow
• Exports available for analysis
            """, title="Help"))
        else:
            console.print("[red]Invalid choice[/red]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
