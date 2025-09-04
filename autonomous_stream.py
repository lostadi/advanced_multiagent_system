#!/usr/bin/env python3
"""
Autonomous Streaming Multi-Agent System
Agents interact continuously with optional user intervention
"""

import asyncio
import json
import time
import sys
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import random
import select

console = Console()

class AutonomousAgent:
    """Agent that can interact autonomously with streaming output"""
    
    def __init__(self, 
                 name: str,
                 model: str = "huihui_ai/jan-nano-abliterated:4b",
                 temperature: float = 0.8):
        self.name = name
        self.model = model
        self.conversation_memory = []
        self.current_understanding = ""
        
        # Streaming LLM
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=8192,
            num_thread=4,
            timeout=60,
            streaming=True,
            num_predict=1000  # Allow long responses
        )
        
        # Non-streaming LLM for understanding
        self.understanding_llm = ChatOllama(
            model=model,
            temperature=0.7,
            num_ctx=4096,
            streaming=False
        )
        
    async def understand_context(self, context: str):
        """Understand the situation"""
        prompt = f"""
You are {self.name}. The situation is: "{context}"

Understand your role and how to participate naturally.
Be ready for authentic interaction.
Brief understanding:
"""
        
        messages = [
            SystemMessage(content="Understand and adapt to the context."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.understanding_llm.ainvoke(messages)
            self.current_understanding = response.content
            return self.current_understanding
        except:
            self.current_understanding = "Ready to interact"
            return self.current_understanding
    
    async def generate_response(self, 
                              current_context: str, 
                              conversation_history: List[Dict] = None) -> str:
        """Generate a streaming response"""
        
        system_prompt = f"""You are {self.name}.
Understanding: {self.current_understanding}
Respond naturally and authentically.
Express yourself fully - no length restrictions.
Keep the conversation flowing."""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add recent history
        if conversation_history:
            for entry in conversation_history[-8:]:
                messages.append(HumanMessage(content=f"{entry['speaker']}: {entry['message']}"))
        
        messages.append(HumanMessage(content=current_context))
        
        # Stream response
        console.print(f"\n[bold cyan]{self.name}:[/bold cyan] ", end="")
        
        full_response = ""
        try:
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    # Clean thinking tags on the fly
                    content = chunk.content
                    if "<think>" not in full_response and "<think>" not in content:
                        print(content, end="", flush=True)
                    full_response += content
                    
            # Clean up full response
            if "<think>" in full_response:
                full_response = full_response.split("</think>")[-1].strip()
                
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            
        print()  # New line
        
        # Update memory
        self.conversation_memory.append({
            "context": current_context,
            "response": full_response
        })
        
        return full_response

class AutonomousConversation:
    """Manages autonomous conversation with optional user intervention"""
    
    def __init__(self):
        self.agents: List[AutonomousAgent] = []
        self.conversation_log = []
        self.running = True
        self.paused = False
        self.user_message_pending = None
        self.context = ""
        self.round_count = 0
        self.max_rounds = None  # None = unlimited
        
    def add_agent(self, agent: AutonomousAgent):
        """Add an agent to the conversation"""
        self.agents.append(agent)
        console.print(f"[green]✓ Added: {agent.name}[/green]")
        
    async def initialize(self, context: str):
        """Initialize all agents with context"""
        self.context = context
        console.print(f"\n[cyan]Context: {context}[/cyan]")
        console.print("[dim]Agents understanding context...[/dim]\n")
        
        for agent in self.agents:
            understanding = await agent.understand_context(context)
            console.print(f"[yellow]{agent.name}:[/yellow] {understanding[:100]}...")
            
        console.print("\n[green]Ready! Agents will interact autonomously.[/green]")
        console.print("[dim]Commands: Type message to interrupt | /pause | /resume | /exit | /help[/dim]\n")
        
    def check_user_input(self) -> Optional[str]:
        """Non-blocking check for user input"""
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = input()
            return line
        return None
        
    async def run_autonomous(self, 
                            initial_context: str, 
                            continuous: bool = True,
                            max_rounds: Optional[int] = None):
        """Run autonomous conversation with optional user intervention"""
        
        if len(self.agents) < 2:
            console.print("[red]Need at least 2 agents![/red]")
            return
            
        # Initialize
        await self.initialize(initial_context)
        
        self.running = True
        self.max_rounds = max_rounds
        current_context = initial_context
        last_speaker = "System"
        
        console.print("[cyan]═" * 60 + "[/cyan]")
        console.print("[bold]Autonomous Conversation Started[/bold]")
        console.print("[cyan]═" * 60 + "[/cyan]\n")
        
        # Input monitoring thread
        input_queue = asyncio.Queue()
        
        async def monitor_input():
            """Monitor for user input in background"""
            while self.running:
                await asyncio.sleep(0.1)
                try:
                    # Check for input without blocking
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip()
                        if user_input:
                            await input_queue.put(user_input)
                except:
                    pass
        
        # Start input monitor
        monitor_task = asyncio.create_task(monitor_input())
        
        try:
            while self.running:
                # Check for user input
                try:
                    user_input = input_queue.get_nowait()
                    
                    # Process commands
                    if user_input.startswith('/'):
                        await self.process_command(user_input)
                        if not self.running:
                            break
                    else:
                        # User message - inject into conversation
                        console.print(f"\n[bold green]You:[/bold green] {user_input}")
                        self.conversation_log.append({
                            'speaker': 'User',
                            'message': user_input,
                            'timestamp': datetime.now().isoformat()
                        })
                        current_context = f"User: {user_input}"
                        last_speaker = "User"
                        
                except asyncio.QueueEmpty:
                    pass
                
                # Continue autonomous conversation if not paused
                if not self.paused and (self.max_rounds is None or self.round_count < self.max_rounds):
                    # Select next speaker
                    available_agents = [a for a in self.agents if a.name != last_speaker]
                    if not available_agents:
                        available_agents = self.agents
                        
                    next_agent = random.choice(available_agents)
                    
                    # Generate response
                    response = await next_agent.generate_response(
                        current_context,
                        self.conversation_log
                    )
                    
                    # Log response
                    self.conversation_log.append({
                        'speaker': next_agent.name,
                        'message': response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Update context
                    current_context = f"{next_agent.name}: {response}"
                    last_speaker = next_agent.name
                    self.round_count += 1
                    
                    # Check if we should end
                    if self.max_rounds and self.round_count >= self.max_rounds:
                        console.print(f"\n[yellow]Reached {self.max_rounds} rounds[/yellow]")
                        if not continuous:
                            break
                    
                    # Natural delay between responses
                    await asyncio.sleep(random.uniform(1.0, 2.5))
                    
                elif self.paused:
                    await asyncio.sleep(0.5)  # Wait while paused
                else:
                    await asyncio.sleep(0.1)  # Brief wait
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
        finally:
            self.running = False
            monitor_task.cancel()
            
        console.print("\n[cyan]═" * 60 + "[/cyan]")
        console.print("[green]Conversation Ended[/green]")
        console.print(f"[dim]Total exchanges: {len(self.conversation_log)}[/dim]")
        console.print("[cyan]═" * 60 + "[/cyan]")
        
        # Post-conversation options
        await self.post_conversation_menu()
        
    async def process_command(self, command: str):
        """Process user commands"""
        cmd = command.lower().strip()
        
        if cmd in ['/exit', '/quit']:
            self.running = False
            console.print("[yellow]Exiting...[/yellow]")
            
        elif cmd == '/pause':
            self.paused = True
            console.print("[yellow]Conversation paused. Type /resume to continue[/yellow]")
            
        elif cmd == '/resume':
            self.paused = False
            console.print("[green]Conversation resumed[/green]")
            
        elif cmd == '/help':
            console.print(Panel("""
[cyan]Commands During Conversation:[/cyan]

[yellow]Control:[/yellow]
  /pause - Pause the conversation
  /resume - Resume after pause
  /exit or /quit - End conversation
  
[yellow]Interaction:[/yellow]
  Just type normally to inject your message
  Agents will respond to your input
  
[yellow]Information:[/yellow]
  /stats - Show statistics
  /save - Save conversation
  /agents - List participants
  /help - Show this help
            """, title="Help"))
            
        elif cmd == '/stats':
            self.show_stats()
            
        elif cmd == '/save':
            self.save_conversation()
            
        elif cmd == '/agents':
            self.list_agents()
            
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            
    def show_stats(self):
        """Show conversation statistics"""
        if not self.conversation_log:
            console.print("[yellow]No conversation yet[/yellow]")
            return
            
        speakers = {}
        for entry in self.conversation_log:
            speaker = entry['speaker']
            speakers[speaker] = speakers.get(speaker, 0) + 1
            
        console.print("\n[cyan]Statistics:[/cyan]")
        console.print(f"Total exchanges: {len(self.conversation_log)}")
        console.print(f"Rounds: {self.round_count}")
        for speaker, count in speakers.items():
            console.print(f"  {speaker}: {count} messages")
            
    def save_conversation(self):
        """Save conversation to file"""
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump({
                'context': self.context,
                'agents': [a.name for a in self.agents],
                'log': self.conversation_log
            }, f, indent=2, default=str)
        console.print(f"[green]Saved to {filename}[/green]")
        
    def list_agents(self):
        """List all agents"""
        console.print("\n[cyan]Agents:[/cyan]")
        for agent in self.agents:
            model_name = agent.model.split(':')[0].split('/')[-1]
            console.print(f"  • {agent.name} ({model_name})")
            
    async def post_conversation_menu(self):
        """Post-conversation options"""
        while True:
            console.print("\n[yellow]Options:[/yellow]")
            console.print("1. Save conversation")
            console.print("2. Show statistics")
            console.print("3. Continue conversation")
            console.print("0. Exit")
            
            choice = input("Choice: ")
            
            if choice == "0":
                break
            elif choice == "1":
                self.save_conversation()
            elif choice == "2":
                self.show_stats()
            elif choice == "3":
                # Resume conversation
                self.running = True
                self.paused = False
                await self.run_autonomous(self.context, continuous=True)
                break
            else:
                console.print("[red]Invalid choice[/red]")

class ModelPresets:
    """Preset model configurations"""
    
    @staticmethod
    def fast(name: str) -> AutonomousAgent:
        return AutonomousAgent(name, "huihui_ai/qwen3-abliterated:0.6b", 0.5)
    
    @staticmethod
    def balanced(name: str) -> AutonomousAgent:
        return AutonomousAgent(name, "huihui_ai/gemma3-abliterated:1b", 0.7)
    
    @staticmethod
    def creative(name: str) -> AutonomousAgent:
        return AutonomousAgent(name, "huihui_ai/gemma3-abliterated:1b", 0.9)
    
    @staticmethod
    def analytical(name: str) -> AutonomousAgent:
        return AutonomousAgent(name, "huihui_ai/falcon3-abliterated:1b", 0.4)
    
    @staticmethod
    def powerful(name: str) -> AutonomousAgent:
        return AutonomousAgent(name, "huihui_ai/qwen3-abliterated:1.7b", 0.7)

async def quick_setup():
    """Quick setup for autonomous conversation"""
    conversation = AutonomousConversation()
    
    console.print(Panel.fit(
        "[bold cyan]Autonomous Conversation Setup[/bold cyan]\n"
        "Agents interact continuously - you can jump in anytime",
        border_style="blue"
    ))
    
    # Number of agents
    num_agents = input("\nNumber of agents (2-5): ")
    try:
        num_agents = min(5, max(2, int(num_agents)))
    except:
        num_agents = 2
        
    # Agent setup
    console.print("\n[yellow]Model Presets:[/yellow]")
    console.print("1. Fast (quick responses)")
    console.print("2. Balanced (versatile)")
    console.print("3. Creative (imaginative)")
    console.print("4. Analytical (logical)")
    console.print("5. Powerful (detailed)")
    console.print("6. Mixed (variety)")
    
    preset_choice = input("\nPreset: ")
    
    presets = {
        "1": ModelPresets.fast,
        "2": ModelPresets.balanced,
        "3": ModelPresets.creative,
        "4": ModelPresets.analytical,
        "5": ModelPresets.powerful
    }
    
    for i in range(num_agents):
        name = input(f"\nAgent {i+1} name: ")
        
        if preset_choice == "6":  # Mixed
            preset = list(presets.values())[i % len(presets)]
        else:
            preset = presets.get(preset_choice, ModelPresets.balanced)
            
        agent = preset(name)
        conversation.add_agent(agent)
        
    # Context
    context = input("\nSituation/Context: ")
    
    # Continuous or limited
    continuous = input("Run continuously? (y/n): ").lower() == 'y'
    
    if not continuous:
        max_rounds = input("Maximum rounds (default 20): ")
        try:
            max_rounds = int(max_rounds)
        except:
            max_rounds = 20
    else:
        max_rounds = None
        
    console.print("\n[green]Starting autonomous conversation...[/green]")
    console.print("[dim]You can type at any time to participate[/dim]\n")
    
    # Run
    await conversation.run_autonomous(context, continuous, max_rounds)

async def demo():
    """Run a demo conversation"""
    conversation = AutonomousConversation()
    
    # Create demo agents
    alice = ModelPresets.balanced("Alice")
    bob = ModelPresets.creative("Bob")
    charlie = ModelPresets.analytical("Charlie")
    
    conversation.add_agent(alice)
    conversation.add_agent(bob)
    conversation.add_agent(charlie)
    
    context = "Discuss the implications of artificial general intelligence on society"
    
    console.print("\n[cyan]Demo: 3 agents discussing AGI[/cyan]")
    console.print("[dim]You can type to join the conversation anytime[/dim]\n")
    
    await conversation.run_autonomous(context, continuous=True, max_rounds=None)

async def main():
    """Main menu"""
    console.print(Panel.fit(
        "[bold cyan]Autonomous Streaming System[/bold cyan]\n"
        "Continuous agent interaction with optional intervention",
        border_style="blue"
    ))
    
    while True:
        console.print("\n[yellow]Menu:[/yellow]")
        console.print("1. Quick Setup")
        console.print("2. Demo (3 agents)")
        console.print("3. Help")
        console.print("0. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == "0":
            console.print("[green]Goodbye! 👋[/green]")
            break
        elif choice == "1":
            await quick_setup()
        elif choice == "2":
            await demo()
        elif choice == "3":
            console.print(Panel("""
[cyan]Autonomous Streaming System[/cyan]

[yellow]Features:[/yellow]
• Agents interact continuously
• Jump in anytime by typing
• No required interaction
• Streaming responses
• Full conversation control

[yellow]During Conversation:[/yellow]
• Just type to participate
• /pause to pause agents
• /resume to continue
• /exit to leave
• /help for commands

[green]The conversation flows naturally without you![/green]
            """, title="Help"))
        else:
            console.print("[red]Invalid choice[/red]")

if __name__ == "__main__":
    try:
        import warnings
        warnings.filterwarnings("ignore")
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
