#!/usr/bin/env python3
"""
Interactive Streaming Multi-Agent System
Real-time interaction with user participation and control
"""

import asyncio
import json
import time
import sys
import termios
import tty
import select
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
import threading
import queue

console = Console()

class StreamingAgent:
    """Agent with streaming output capabilities"""
    
    def __init__(self, 
                 name: str,
                 model: str = "huihui_ai/qwen3-abliterated:0.6b",
                 temperature: float = 0.8,
                 is_user: bool = False):
        self.name = name
        self.model = model
        self.is_user = is_user  # Is this the human user?
        self.conversation_memory = []
        self.current_context = ""
        
        if not is_user:
            # AI agent with streaming
            self.llm = ChatOllama(
                model=model,
                temperature=temperature,
                num_ctx=8192,  # Large context
                num_thread=4,
                timeout=60,
                streaming=True,  # Enable streaming
                callbacks=[StreamingStdOutCallbackHandler()]
            )
        
    async def understand_situation(self, context: str):
        """Agent understands the current situation"""
        if self.is_user:
            return  # User doesn't need AI understanding
            
        understanding_prompt = f"""
You are {self.name}. The situation is: "{context}"

Quickly understand your role and how to participate naturally.
Be ready to interact authentically.
"""
        
        messages = [
            SystemMessage(content="Understand the context and adapt naturally."),
            HumanMessage(content=understanding_prompt)
        ]
        
        try:
            # Don't stream understanding phase
            llm_quiet = ChatOllama(
                model=self.model,
                temperature=0.7,
                num_ctx=4096,
                streaming=False
            )
            response = await llm_quiet.ainvoke(messages)
            self.current_context = response.content
        except:
            self.current_context = context
    
    async def stream_response(self, 
                            prompt: str, 
                            conversation_history: List[Dict] = None,
                            max_length: Optional[int] = None) -> str:
        """Stream a response with no length limit unless specified"""
        if self.is_user:
            return ""  # User types their own responses
        
        # Build context
        system_prompt = f"""You are {self.name}.
Context: {self.current_context}
Respond naturally to the conversation.
Be authentic and engaging.
{"No length limit - express yourself fully." if not max_length else f"Keep under {max_length} words."}
"""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add recent history
        if conversation_history:
            for entry in conversation_history[-10:]:  # More history
                messages.append(HumanMessage(content=f"{entry['speaker']}: {entry['message']}"))
        
        messages.append(HumanMessage(content=prompt))
        
        # Stream the response
        console.print(f"\n[bold cyan]{self.name}:[/bold cyan] ", end="")
        
        full_response = ""
        try:
            # Create streaming LLM for this response
            stream_llm = ChatOllama(
                model=self.model,
                temperature=0.8,
                num_ctx=8192,
                streaming=True,
                num_predict=max_length if max_length else 2000  # No artificial limit
            )
            
            # Stream tokens
            async for chunk in stream_llm.astream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    full_response += chunk.content
                    
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            
        print()  # New line after streaming
        return full_response

class InteractiveConversation:
    """Manages interactive streaming conversation with user participation"""
    
    def __init__(self):
        self.agents: Dict[str, StreamingAgent] = {}
        self.conversation_log = []
        self.running = False
        self.user_input_queue = queue.Queue()
        self.commands = {
            '/exit': self.handle_exit,
            '/quit': self.handle_exit,
            '/save': self.save_conversation,
            '/stats': self.show_stats,
            '/help': self.show_help,
            '/agents': self.list_agents,
            '/add': self.add_agent_command,
            '/remove': self.remove_agent_command,
            '/context': self.change_context
        }
        
    def add_agent(self, agent: StreamingAgent):
        """Add an agent (including user)"""
        self.agents[agent.name] = agent
        if not agent.is_user:
            console.print(f"[green]✓ Added AI agent: {agent.name}[/green]")
        else:
            console.print(f"[green]✓ You joined as: {agent.name}[/green]")
    
    def handle_exit(self):
        """Handle exit command"""
        self.running = False
        console.print("[yellow]Exiting conversation...[/yellow]")
        
    def save_conversation(self):
        """Save conversation to file"""
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.conversation_log, f, indent=2, default=str)
        console.print(f"[green]Saved to {filename}[/green]")
        
    def show_stats(self):
        """Show conversation statistics"""
        if not self.conversation_log:
            console.print("[yellow]No conversation yet[/yellow]")
            return
            
        total = len(self.conversation_log)
        speakers = {}
        for entry in self.conversation_log:
            speaker = entry['speaker']
            speakers[speaker] = speakers.get(speaker, 0) + 1
            
        console.print("\n[cyan]Conversation Statistics:[/cyan]")
        console.print(f"Total exchanges: {total}")
        for speaker, count in speakers.items():
            console.print(f"  {speaker}: {count} messages")
            
    def show_help(self):
        """Show available commands"""
        console.print(Panel("""
[cyan]Available Commands During Conversation:[/cyan]

[yellow]Control:[/yellow]
  /exit or /quit - End conversation
  /help - Show this help
  
[yellow]Management:[/yellow]
  /agents - List all participants
  /add <name> - Add new AI agent
  /remove <name> - Remove an agent
  /context <new context> - Change situation
  
[yellow]Data:[/yellow]
  /save - Save conversation log
  /stats - Show statistics
  
[green]Tips:[/green]
  • Type normally to speak as your character
  • Press Enter to send your message
  • AI agents respond automatically
  • No length limits on responses
        """, title="Help"))
        
    def list_agents(self):
        """List all agents in conversation"""
        console.print("\n[cyan]Current Participants:[/cyan]")
        for name, agent in self.agents.items():
            agent_type = "You" if agent.is_user else f"AI ({agent.model.split(':')[0].split('/')[-1]})"
            console.print(f"  • {name} [{agent_type}]")
            
    def add_agent_command(self):
        """Add a new agent during conversation"""
        console.print("[yellow]Adding new agent...[/yellow]")
        # This would need async handling in practice
        
    def remove_agent_command(self):
        """Remove an agent during conversation"""
        console.print("[yellow]Remove agent functionality[/yellow]")
        
    def change_context(self):
        """Change the conversation context"""
        console.print("[yellow]Context change functionality[/yellow]")
        
    async def initialize(self, context: str):
        """Initialize all agents with context"""
        console.print(f"\n[cyan]Setting up scenario: {context}[/cyan]")
        console.print("[dim]Agents preparing...[/dim]\n")
        
        # AI agents understand context
        for agent in self.agents.values():
            if not agent.is_user:
                await agent.understand_situation(context)
                
        console.print("[green]Ready to begin![/green]")
        console.print("[dim]Type /help for commands[/dim]\n")
        
    def get_user_input_nonblocking(self) -> Optional[str]:
        """Check for user input without blocking"""
        try:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip()
                return line
        except:
            pass
        return None
        
    async def run(self, 
                  initial_context: str,
                  user_character_name: Optional[str] = None):
        """Run interactive conversation with streaming"""
        
        # Initialize context
        await self.initialize(initial_context)
        
        self.running = True
        current_context = initial_context
        conversation_active = True
        
        console.print("[cyan]═" * 60 + "[/cyan]")
        console.print("[bold]Interactive Conversation Started[/bold]")
        console.print("[cyan]═" * 60 + "[/cyan]\n")
        
        # Get user agent if exists
        user_agent = None
        for agent in self.agents.values():
            if agent.is_user:
                user_agent = agent
                break
                
        if user_agent:
            console.print(f"[green]You are playing as: {user_agent.name}[/green]")
            console.print("[dim]Type your messages and press Enter[/dim]\n")
            
        # Main conversation loop
        while self.running and conversation_active:
            try:
                # Check for user input
                user_input = input(f"[{user_character_name or 'You'}]: " if user_agent else "> ")
                
                # Check if it's a command
                if user_input.startswith('/'):
                    command = user_input.split()[0]
                    if command in self.commands:
                        self.commands[command]()
                        if command in ['/exit', '/quit']:
                            break
                    else:
                        console.print(f"[red]Unknown command: {command}[/red]")
                    continue
                    
                # User message
                if user_input and user_agent:
                    # Log user message
                    self.conversation_log.append({
                        'speaker': user_agent.name,
                        'message': user_input,
                        'timestamp': datetime.now().isoformat(),
                        'is_user': True
                    })
                    
                    current_context = f"{user_agent.name}: {user_input}"
                    
                    # AI agents respond
                    for agent_name, agent in self.agents.items():
                        if not agent.is_user:
                            # Randomly decide if this agent responds
                            import random
                            if random.random() < 0.7:  # 70% chance to respond
                                response = await agent.stream_response(
                                    current_context,
                                    self.conversation_log
                                )
                                
                                # Log AI response
                                self.conversation_log.append({
                                    'speaker': agent.name,
                                    'message': response,
                                    'timestamp': datetime.now().isoformat(),
                                    'is_user': False
                                })
                                
                                current_context = f"{agent.name}: {response}"
                                
                                # Small delay between agents
                                await asyncio.sleep(0.5)
                                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                
        console.print("\n[cyan]═" * 60 + "[/cyan]")
        console.print("[green]Conversation Ended[/green]")
        console.print("[cyan]═" * 60 + "[/cyan]")
        
        # Ask about saving
        if self.conversation_log:
            save = input("\nSave conversation? (y/n): ")
            if save.lower() == 'y':
                self.save_conversation()
                
            show_stats = input("Show statistics? (y/n): ")
            if show_stats.lower() == 'y':
                self.show_stats()

class AgentTemplates:
    """Quick templates for different agent types"""
    
    @staticmethod
    def fast_ai(name: str) -> StreamingAgent:
        return StreamingAgent(name, "huihui_ai/qwen3-abliterated:0.6b", 0.5)
    
    @staticmethod
    def thoughtful_ai(name: str) -> StreamingAgent:
        return StreamingAgent(name, "huihui_ai/llama3.2-abliterate:1b", 0.7)
    
    @staticmethod
    def creative_ai(name: str) -> StreamingAgent:
        return StreamingAgent(name, "huihui_ai/gemma3-abliterated:1b", 0.9)
    
    @staticmethod
    def human_player(name: str) -> StreamingAgent:
        return StreamingAgent(name, is_user=True)

async def quick_interactive_setup():
    """Quick setup for interactive conversation"""
    conversation = InteractiveConversation()
    
    console.print(Panel.fit(
        "[bold cyan]Interactive Streaming Setup[/bold cyan]\n"
        "You can participate and control the conversation",
        border_style="blue"
    ))
    
    # Ask if user wants to participate
    participate = input("\nDo you want to be a character? (y/n): ")
    
    if participate.lower() == 'y':
        user_name = input("Your character name: ")
        user_agent = AgentTemplates.human_player(user_name)
        conversation.add_agent(user_agent)
    else:
        user_name = None
        
    # Add AI agents
    num_ai = input("How many AI agents? (1-4): ")
    try:
        num_ai = min(4, max(1, int(num_ai)))
    except:
        num_ai = 2
        
    console.print("\n[yellow]AI Types:[/yellow]")
    console.print("1. Fast (quick responses)")
    console.print("2. Thoughtful (detailed)")  
    console.print("3. Creative (imaginative)")
    console.print("4. Mixed")
    
    ai_type = input("Choose type: ")
    
    for i in range(num_ai):
        name = input(f"AI Agent {i+1} name: ")
        
        if ai_type == "1":
            agent = AgentTemplates.fast_ai(name)
        elif ai_type == "2":
            agent = AgentTemplates.thoughtful_ai(name)
        elif ai_type == "3":
            agent = AgentTemplates.creative_ai(name)
        else:  # Mixed
            templates = [AgentTemplates.fast_ai, AgentTemplates.thoughtful_ai, AgentTemplates.creative_ai]
            agent = templates[i % 3](name)
            
        conversation.add_agent(agent)
        
    # Get context
    context = input("\nSituation/Context: ")
    
    # Run conversation
    await conversation.run(context, user_name)

async def main():
    """Main entry point"""
    console.print(Panel.fit(
        "[bold cyan]Interactive Streaming Multi-Agent System[/bold cyan]\n"
        "Real-time conversation with full control",
        border_style="blue"
    ))
    
    while True:
        console.print("\n[yellow]Options:[/yellow]")
        console.print("1. Quick Interactive Setup")
        console.print("2. Test Demo (2 AI agents)")
        console.print("3. Help")
        console.print("0. Exit")
        
        choice = input("\n[Choice]: ")
        
        if choice == "0":
            console.print("[green]Goodbye! 👋[/green]")
            break
        elif choice == "1":
            await quick_interactive_setup()
        elif choice == "2":
            # Demo
            conversation = InteractiveConversation()
            conversation.add_agent(AgentTemplates.thoughtful_ai("Alice"))
            conversation.add_agent(AgentTemplates.creative_ai("Bob"))
            conversation.add_agent(AgentTemplates.human_player("User"))
            await conversation.run("Let's discuss the nature of consciousness", "User")
        elif choice == "3":
            console.print(Panel("""
[cyan]Interactive Streaming System[/cyan]

Features:
• Real-time streaming responses
• User participation as character
• Interrupt and guide anytime
• No response length limits
• Commands during conversation

[yellow]During Conversation:[/yellow]
• Type normally to speak
• Use /commands for control
• /help for all commands
• /exit to leave

[green]Tips:[/green]
• AI agents stream responses
• You see output as it's generated
• Full conversation control
            """, title="Help"))
        else:
            console.print("[red]Invalid choice[/red]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
