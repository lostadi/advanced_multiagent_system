#!/usr/bin/env python3
"""
Autonomous Multi-Agent Interaction System
Multiple AI agents interact with each other without human intervention
"""

import asyncio
import time
import random
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

console = Console()

class AutonomousAgent:
    """An autonomous agent that can interact with other agents"""
    
    def __init__(self, name: str, personality: str, model: str = "huihui_ai/qwen3-abliterated:0.6b", color: str = "white"):
        self.name = name
        self.personality = personality
        self.color = color
        self.conversation_history = []
        
        # Each agent gets its own LLM instance
        self.llm = ChatOllama(
            model=model,
            temperature=0.7,
            num_ctx=2048,
            num_thread=4,  # Fewer threads per agent for parallel processing
            timeout=30,
            num_predict=150  # Keep responses concise
        )
        
    async def respond_to(self, message: str, sender: str = "User") -> str:
        """Generate a response to a message"""
        # Build context from conversation history
        context = f"You are {self.name}. {self.personality}\n"
        context += f"You are having a conversation with {sender}.\n"
        context += "Respond naturally, stay in character, and keep responses under 3 sentences.\n"
        
        # Add recent conversation history
        recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
        
        messages = [
            SystemMessage(content=context),
        ]
        
        # Add history
        for hist in recent_history:
            messages.append(HumanMessage(content=f"{hist['sender']}: {hist['message']}"))
        
        # Add current message
        messages.append(HumanMessage(content=f"{sender}: {message}"))
        
        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # Clean up response (remove thinking tags if present)
            if "<think>" in response_text:
                response_text = response_text.split("</think>")[-1].strip()
            
            # Store in history
            self.conversation_history.append({
                "sender": sender,
                "message": message,
                "timestamp": datetime.now()
            })
            self.conversation_history.append({
                "sender": self.name,
                "message": response_text,
                "timestamp": datetime.now()
            })
            
            return response_text
            
        except Exception as e:
            return f"*{self.name} seems confused* (Error: {str(e)[:50]})"

class MultiAgentConversation:
    """Manages autonomous conversations between multiple agents"""
    
    def __init__(self):
        self.agents: List[AutonomousAgent] = []
        self.conversation_log = []
        
    def add_agent(self, agent: AutonomousAgent):
        """Add an agent to the conversation"""
        self.agents.append(agent)
        
    async def run_conversation(self, topic: str, rounds: int = 5, display_live: bool = True):
        """Run an autonomous conversation between agents"""
        if len(self.agents) < 2:
            console.print("[red]Need at least 2 agents for a conversation![/red]")
            return
        
        console.print(f"\n[cyan]Starting conversation about: {topic}[/cyan]")
        console.print(f"[dim]Participants: {', '.join([a.name for a in self.agents])}[/dim]\n")
        
        # Start the conversation with the topic
        current_message = topic
        current_sender = "Moderator"
        
        for round_num in range(rounds):
            # Each agent responds in sequence
            for i, agent in enumerate(self.agents):
                # Display who's thinking
                if display_live:
                    console.print(f"[dim]💭 {agent.name} is thinking...[/dim]", end="\r")
                
                # Get response
                start_time = time.time()
                response = await agent.respond_to(current_message, current_sender)
                response_time = time.time() - start_time
                
                # Clear the thinking message and display response
                if display_live:
                    console.print(" " * 50, end="\r")  # Clear line
                    console.print(f"[{agent.color}]{agent.name}[/{agent.color}]: {response}")
                    console.print(f"[dim](Response time: {response_time:.1f}s)[/dim]\n")
                
                # Log the conversation
                self.conversation_log.append({
                    "round": round_num + 1,
                    "speaker": agent.name,
                    "message": response,
                    "response_time": response_time
                })
                
                # Update for next agent
                current_message = response
                current_sender = agent.name
                
                # Small delay to make conversation feel natural
                await asyncio.sleep(0.5)
        
        console.print("[green]Conversation complete![/green]")
        
    async def run_debate(self, topic: str, rounds: int = 3):
        """Run a structured debate between agents"""
        if len(self.agents) < 2:
            console.print("[red]Need at least 2 agents for a debate![/red]")
            return
        
        console.print(f"\n[yellow]🎭 DEBATE: {topic}[/yellow]")
        console.print("="*60)
        
        # Assign positions
        for i, agent in enumerate(self.agents):
            position = "FOR" if i % 2 == 0 else "AGAINST"
            agent.personality += f" You are arguing {position} the topic."
            console.print(f"{agent.name} is arguing [{agent.color}]{position}[/{agent.color}]")
        
        console.print("="*60 + "\n")
        
        # Run the debate
        await self.run_conversation(f"Debate topic: {topic}", rounds * 2, display_live=True)
        
    async def run_collaboration(self, task: str, rounds: int = 4):
        """Run a collaborative task between agents"""
        console.print(f"\n[green]🤝 COLLABORATION: {task}[/green]")
        console.print("="*60 + "\n")
        
        # Modify personalities for collaboration
        for agent in self.agents:
            agent.personality += " You are working collaboratively. Build on others' ideas."
        
        await self.run_conversation(f"Let's work together on: {task}", rounds, display_live=True)

class AgentOrchestrator:
    """Orchestrates complex multi-agent scenarios"""
    
    def __init__(self):
        self.scenarios = {
            "philosophical_debate": self.philosophical_debate,
            "creative_storytelling": self.creative_storytelling,
            "problem_solving": self.problem_solving,
            "comedy_show": self.comedy_show,
            "technical_discussion": self.technical_discussion
        }
        
    async def philosophical_debate(self):
        """Run a philosophical debate"""
        conversation = MultiAgentConversation()
        
        # Create philosophers
        socrates = AutonomousAgent(
            "Socrates",
            "You are Socrates, the ancient Greek philosopher. You ask probing questions and seek truth through dialogue.",
            color="cyan"
        )
        
        nietzsche = AutonomousAgent(
            "Nietzsche",
            "You are Friedrich Nietzsche. You challenge conventional morality and speak of the will to power.",
            color="red"
        )
        
        buddha = AutonomousAgent(
            "Buddha",
            "You are Buddha. You speak of suffering, impermanence, and the middle way with compassion.",
            color="yellow"
        )
        
        conversation.add_agent(socrates)
        conversation.add_agent(nietzsche)
        conversation.add_agent(buddha)
        
        await conversation.run_debate("What is the meaning of existence?", rounds=3)
        
    async def creative_storytelling(self):
        """Collaborative storytelling"""
        conversation = MultiAgentConversation()
        
        narrator = AutonomousAgent(
            "Narrator",
            "You are a master storyteller. Set scenes and advance the plot dramatically.",
            color="green"
        )
        
        character = AutonomousAgent(
            "Hero",
            "You are the brave hero of the story. Speak in first person about your adventures.",
            color="blue"
        )
        
        villain = AutonomousAgent(
            "Villain",
            "You are the cunning antagonist. Create conflict and challenge the hero.",
            color="red"
        )
        
        conversation.add_agent(narrator)
        conversation.add_agent(character)
        conversation.add_agent(villain)
        
        await conversation.run_collaboration(
            "Create a story about a mysterious artifact that grants wishes but at a terrible cost",
            rounds=4
        )
        
    async def problem_solving(self):
        """Technical problem solving session"""
        conversation = MultiAgentConversation()
        
        engineer = AutonomousAgent(
            "Engineer",
            "You are a practical engineer. Focus on feasibility and implementation details.",
            color="blue"
        )
        
        scientist = AutonomousAgent(
            "Scientist",
            "You are a research scientist. Provide theoretical insights and innovative approaches.",
            color="cyan"
        )
        
        analyst = AutonomousAgent(
            "Analyst",
            "You are a systems analyst. Consider edge cases, risks, and optimization.",
            color="green"
        )
        
        conversation.add_agent(engineer)
        conversation.add_agent(scientist)
        conversation.add_agent(analyst)
        
        await conversation.run_collaboration(
            "Design a system for sustainable energy storage using common materials",
            rounds=3
        )
        
    async def comedy_show(self):
        """Comedy improvisation"""
        conversation = MultiAgentConversation()
        
        comedian1 = AutonomousAgent(
            "Setup",
            "You're a comedian who sets up jokes. Be witty and create funny scenarios.",
            color="yellow"
        )
        
        comedian2 = AutonomousAgent(
            "Punchline",
            "You deliver punchlines and witty responses. Be surprising and hilarious.",
            color="magenta"
        )
        
        heckler = AutonomousAgent(
            "Heckler",
            "You're a friendly heckler. Make funny interruptions and silly observations.",
            color="red"
        )
        
        conversation.add_agent(comedian1)
        conversation.add_agent(comedian2)
        conversation.add_agent(heckler)
        
        await conversation.run_conversation(
            "Let's do improv comedy about AI taking over the world",
            rounds=4
        )
        
    async def technical_discussion(self):
        """Technical architecture discussion"""
        conversation = MultiAgentConversation()
        
        backend_dev = AutonomousAgent(
            "BackendDev",
            "You're a backend developer. Focus on APIs, databases, and server architecture.",
            color="green"
        )
        
        frontend_dev = AutonomousAgent(
            "FrontendDev",
            "You're a frontend developer. Care about user experience and interface design.",
            color="blue"
        )
        
        devops = AutonomousAgent(
            "DevOps",
            "You're a DevOps engineer. Focus on deployment, scaling, and monitoring.",
            color="yellow"
        )
        
        conversation.add_agent(backend_dev)
        conversation.add_agent(frontend_dev)
        conversation.add_agent(devops)
        
        await conversation.run_collaboration(
            "Design the architecture for a real-time collaborative document editor",
            rounds=3
        )

async def custom_conversation():
    """Create a custom conversation with user-defined agents"""
    console.print("\n[cyan]Create Your Custom Conversation[/cyan]")
    
    conversation = MultiAgentConversation()
    
    # Get number of agents
    num_agents = console.input("How many agents? (2-4): ")
    try:
        num_agents = min(4, max(2, int(num_agents)))
    except:
        num_agents = 2
    
    # Create agents
    colors = ["red", "blue", "green", "yellow"]
    for i in range(num_agents):
        name = console.input(f"Agent {i+1} name: ")
        personality = console.input(f"Agent {i+1} personality: ")
        
        agent = AutonomousAgent(name, personality, color=colors[i % len(colors)])
        conversation.add_agent(agent)
    
    # Get topic
    topic = console.input("\nConversation topic: ")
    rounds = console.input("Number of rounds (1-10): ")
    
    try:
        rounds = min(10, max(1, int(rounds)))
    except:
        rounds = 3
    
    # Run conversation
    await conversation.run_conversation(topic, rounds)

async def main():
    """Main menu for autonomous agents"""
    console.print(Panel.fit(
        "[bold cyan]Autonomous Multi-Agent System[/bold cyan]\n"
        "Watch AI agents interact without human intervention!",
        border_style="blue"
    ))
    
    orchestrator = AgentOrchestrator()
    
    while True:
        console.print("\n[yellow]Choose a scenario:[/yellow]")
        console.print("1. Philosophical Debate (Socrates vs Nietzsche vs Buddha)")
        console.print("2. Creative Storytelling (Narrator, Hero, Villain)")
        console.print("3. Problem Solving (Engineer, Scientist, Analyst)")
        console.print("4. Comedy Show (Comedians and Heckler)")
        console.print("5. Technical Discussion (Backend, Frontend, DevOps)")
        console.print("6. Custom Conversation (Define your own agents)")
        console.print("0. Exit")
        
        choice = console.input("\n[cyan]Choice: [/cyan]")
        
        if choice == "0":
            console.print("[green]Goodbye! 👋[/green]")
            break
        elif choice == "1":
            await orchestrator.philosophical_debate()
        elif choice == "2":
            await orchestrator.creative_storytelling()
        elif choice == "3":
            await orchestrator.problem_solving()
        elif choice == "4":
            await orchestrator.comedy_show()
        elif choice == "5":
            await orchestrator.technical_discussion()
        elif choice == "6":
            await custom_conversation()
        else:
            console.print("[red]Invalid choice[/red]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
