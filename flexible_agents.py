#!/usr/bin/env python3
"""
Flexible Multi-Agent System
A general-purpose system for various agent interactions and tasks
"""

import asyncio
import json
import time
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import print as rprint

console = Console()

class InteractionMode(Enum):
    """Different interaction modes for agents"""
    CONVERSATION = "conversation"
    DEBATE = "debate"
    BRAINSTORM = "brainstorm"
    ROLEPLAY = "roleplay"
    INTERVIEW = "interview"
    TEACHING = "teaching"
    COLLABORATION = "collaboration"
    COMPETITION = "competition"
    NEGOTIATION = "negotiation"
    THERAPY = "therapy"
    GAME = "game"
    SIMULATION = "simulation"

class FlexibleAgent:
    """A flexible agent that can adapt to any role or task"""
    
    def __init__(self, 
                 name: str, 
                 base_personality: str = "",
                 model: str = "huihui_ai/qwen3-abliterated:0.6b",
                 temperature: float = 0.7,
                 max_tokens: int = 200):
        self.name = name
        self.base_personality = base_personality
        self.current_role = base_personality
        self.memory = []
        self.relationships = {}  # Track relationships with other agents
        self.emotional_state = "neutral"
        self.goals = []
        self.knowledge_base = {}
        
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=2048,
            num_thread=4,
            timeout=30,
            num_predict=max_tokens
        )
        
    def set_role(self, role: str):
        """Dynamically change the agent's role"""
        self.current_role = role
        
    def add_goal(self, goal: str):
        """Add a goal for the agent to pursue"""
        self.goals.append(goal)
        
    def update_relationship(self, other_agent: str, sentiment: str):
        """Update relationship with another agent"""
        self.relationships[other_agent] = sentiment
        
    def set_emotional_state(self, state: str):
        """Set the agent's emotional state"""
        self.emotional_state = state
        
    async def generate_response(self, 
                               context: str, 
                               mode: InteractionMode = InteractionMode.CONVERSATION,
                               additional_instructions: str = "") -> str:
        """Generate a contextual response based on mode and situation"""
        
        # Build comprehensive prompt
        prompt_parts = []
        
        # Base identity
        if self.current_role:
            prompt_parts.append(f"You are {self.name}. {self.current_role}")
        else:
            prompt_parts.append(f"You are {self.name}.")
            
        # Emotional state
        if self.emotional_state != "neutral":
            prompt_parts.append(f"Your current emotional state: {self.emotional_state}")
            
        # Goals
        if self.goals:
            prompt_parts.append(f"Your goals: {', '.join(self.goals)}")
            
        # Relationships
        if self.relationships:
            rel_str = ", ".join([f"{k}: {v}" for k, v in self.relationships.items()])
            prompt_parts.append(f"Your relationships: {rel_str}")
            
        # Mode-specific instructions
        mode_instructions = {
            InteractionMode.DEBATE: "Argue your position strongly but respectfully.",
            InteractionMode.BRAINSTORM: "Be creative and build on ideas. Think outside the box.",
            InteractionMode.ROLEPLAY: "Stay completely in character. Be immersive.",
            InteractionMode.INTERVIEW: "Ask insightful questions or give thoughtful answers.",
            InteractionMode.TEACHING: "Explain clearly and check for understanding.",
            InteractionMode.COLLABORATION: "Work together constructively. Support others' ideas.",
            InteractionMode.COMPETITION: "Try to win but maintain sportsmanship.",
            InteractionMode.NEGOTIATION: "Seek mutual benefit while advocating for your interests.",
            InteractionMode.THERAPY: "Be empathetic and supportive. Listen actively.",
            InteractionMode.GAME: "Play strategically and have fun.",
            InteractionMode.SIMULATION: "Act realistically within the simulation parameters."
        }
        
        if mode in mode_instructions:
            prompt_parts.append(mode_instructions[mode])
            
        # Additional custom instructions
        if additional_instructions:
            prompt_parts.append(additional_instructions)
            
        # Response format
        prompt_parts.append("Keep responses concise (2-3 sentences max) and natural.")
        
        # Build messages
        system_prompt = "\n".join(prompt_parts)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]
        
        # Add recent memory for context
        for memory_item in self.memory[-3:]:
            messages.append(HumanMessage(content=memory_item))
        
        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # Clean up response
            if "<think>" in response_text:
                response_text = response_text.split("</think>")[-1].strip()
                
            # Store in memory
            self.memory.append(f"{self.name}: {response_text}")
            if len(self.memory) > 10:  # Keep memory limited
                self.memory.pop(0)
                
            return response_text
            
        except Exception as e:
            return f"*{self.name} is having technical difficulties*"

class FlexibleMultiAgentSystem:
    """A flexible system for managing multiple agents in various scenarios"""
    
    def __init__(self):
        self.agents: Dict[str, FlexibleAgent] = {}
        self.conversation_log = []
        self.current_mode = InteractionMode.CONVERSATION
        self.scenario_state = {}
        
    def add_agent(self, agent: FlexibleAgent):
        """Add an agent to the system"""
        self.agents[agent.name] = agent
        
    def remove_agent(self, name: str):
        """Remove an agent from the system"""
        if name in self.agents:
            del self.agents[name]
            
    def set_mode(self, mode: InteractionMode):
        """Set the interaction mode"""
        self.current_mode = mode
        
    def setup_scenario(self, scenario: Dict[str, Any]):
        """Setup a custom scenario"""
        self.scenario_state = scenario
        
        # Apply scenario settings to agents
        if "agent_roles" in scenario:
            for agent_name, role in scenario["agent_roles"].items():
                if agent_name in self.agents:
                    self.agents[agent_name].set_role(role)
                    
        if "agent_goals" in scenario:
            for agent_name, goals in scenario["agent_goals"].items():
                if agent_name in self.agents:
                    for goal in goals:
                        self.agents[agent_name].add_goal(goal)
                        
        if "relationships" in scenario:
            for agent_name, rels in scenario["relationships"].items():
                if agent_name in self.agents:
                    for other, sentiment in rels.items():
                        self.agents[agent_name].update_relationship(other, sentiment)
                        
    async def run_interaction(self, 
                            initial_context: str,
                            rounds: int = 5,
                            mode: Optional[InteractionMode] = None,
                            custom_instructions: str = "",
                            allow_interruptions: bool = False):
        """Run a flexible interaction between agents"""
        
        if not self.agents:
            console.print("[red]No agents in the system![/red]")
            return
            
        if mode:
            self.current_mode = mode
            
        console.print(f"\n[cyan]Starting {self.current_mode.value} interaction[/cyan]")
        console.print(f"[dim]Participants: {', '.join(self.agents.keys())}[/dim]\n")
        
        current_context = initial_context
        agent_list = list(self.agents.values())
        
        for round_num in range(rounds):
            console.print(f"[dim]Round {round_num + 1}/{rounds}[/dim]\n")
            
            # Randomize order if in certain modes
            if self.current_mode in [InteractionMode.BRAINSTORM, InteractionMode.GAME]:
                random.shuffle(agent_list)
                
            for agent in agent_list:
                # Check for interruptions
                if allow_interruptions and random.random() < 0.2:
                    # Another agent might interrupt
                    interruptor = random.choice([a for a in agent_list if a != agent])
                    interrupt_response = await interruptor.generate_response(
                        f"You want to interrupt with a quick comment about: {current_context}",
                        self.current_mode,
                        "Make a brief interruption (1 sentence max)."
                    )
                    console.print(f"[yellow]{interruptor.name} (interrupting):[/yellow] {interrupt_response}")
                    
                # Generate main response
                response = await agent.generate_response(
                    current_context,
                    self.current_mode,
                    custom_instructions
                )
                
                # Display with color coding based on mode
                color = self._get_color_for_mode(agent.name)
                console.print(f"[{color}]{agent.name}:[/{color}] {response}")
                
                # Log interaction
                self.conversation_log.append({
                    "round": round_num + 1,
                    "speaker": agent.name,
                    "message": response,
                    "mode": self.current_mode.value,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update context for next agent
                current_context = response
                
                # Small delay for natural flow
                await asyncio.sleep(0.3)
                
            console.print()  # Space between rounds
            
        console.print("[green]Interaction complete![/green]")
        
    def _get_color_for_mode(self, agent_name: str) -> str:
        """Get color based on mode and agent"""
        colors = ["cyan", "magenta", "yellow", "green", "blue", "red"]
        return colors[hash(agent_name) % len(colors)]
        
    async def run_parallel_conversations(self, topics: List[str], rounds: int = 3):
        """Run multiple parallel conversations"""
        console.print("\n[cyan]Running Parallel Conversations[/cyan]")
        
        agent_list = list(self.agents.values())
        if len(agent_list) < 2:
            console.print("[red]Need at least 2 agents![/red]")
            return
            
        tasks = []
        for topic in topics:
            # Randomly pair agents for each topic
            selected_agents = random.sample(agent_list, min(2, len(agent_list)))
            
            async def converse(topic, agents):
                console.print(f"\n[yellow]Topic: {topic}[/yellow]")
                console.print(f"Agents: {', '.join([a.name for a in agents])}")
                
                context = topic
                for _ in range(rounds):
                    for agent in agents:
                        response = await agent.generate_response(context, InteractionMode.CONVERSATION)
                        console.print(f"  {agent.name}: {response}")
                        context = response
                        
            tasks.append(converse(topic, selected_agents))
            
        await asyncio.gather(*tasks)
        
    def export_conversation(self, filename: str = "conversation_log.json"):
        """Export conversation log to file"""
        with open(filename, 'w') as f:
            json.dump(self.conversation_log, f, indent=2, default=str)
        console.print(f"[green]Conversation exported to {filename}[/green]")
        
    def get_statistics(self):
        """Get interaction statistics"""
        stats = {
            "total_interactions": len(self.conversation_log),
            "agents": list(self.agents.keys()),
            "modes_used": list(set(log["mode"] for log in self.conversation_log)),
            "average_response_length": sum(len(log["message"].split()) for log in self.conversation_log) / len(self.conversation_log) if self.conversation_log else 0
        }
        
        table = Table(title="Interaction Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))
            
        console.print(table)

class InteractionTemplates:
    """Pre-built templates for common interaction scenarios"""
    
    @staticmethod
    def business_meeting():
        return {
            "agent_roles": {
                "CEO": "You are the CEO, focused on company vision and strategy.",
                "CFO": "You are the CFO, concerned with financial implications.",
                "CTO": "You are the CTO, focused on technical feasibility.",
                "Marketing": "You are the Marketing head, thinking about customer impact."
            },
            "mode": InteractionMode.COLLABORATION,
            "context": "Discussing the launch of a new AI product line"
        }
        
    @staticmethod
    def creative_workshop():
        return {
            "agent_roles": {
                "Artist": "You are a visionary artist with bold ideas.",
                "Writer": "You are a creative writer who loves narrative.",
                "Designer": "You are a designer focused on aesthetics and function.",
                "Producer": "You are a producer thinking about feasibility."
            },
            "mode": InteractionMode.BRAINSTORM,
            "context": "Creating a new multimedia art installation"
        }
        
    @staticmethod
    def debate_club():
        return {
            "agent_roles": {
                "Proponent": "You strongly support the proposition.",
                "Opponent": "You strongly oppose the proposition.",
                "Moderator": "You are a neutral moderator ensuring fair debate."
            },
            "mode": InteractionMode.DEBATE,
            "context": "Should AI have rights?"
        }
        
    @staticmethod
    def therapy_session():
        return {
            "agent_roles": {
                "Therapist": "You are an empathetic therapist using active listening.",
                "Client": "You are seeking help with stress and anxiety.",
                "Observer": "You are a training therapist observing the session."
            },
            "mode": InteractionMode.THERAPY,
            "context": "Exploring work-life balance challenges"
        }
        
    @staticmethod
    def game_night():
        return {
            "agent_roles": {
                "Player1": "You are competitive but fair.",
                "Player2": "You love strategy and planning.",
                "Player3": "You're here for fun and jokes.",
                "GameMaster": "You manage the game and keep it fun."
            },
            "mode": InteractionMode.GAME,
            "context": "Playing a word association game"
        }

async def interactive_setup():
    """Interactive setup for custom scenarios"""
    system = FlexibleMultiAgentSystem()
    
    console.print("\n[cyan]Interactive Agent Setup[/cyan]")
    
    # Get number of agents
    num_agents = console.input("Number of agents (1-6): ")
    try:
        num_agents = min(6, max(1, int(num_agents)))
    except:
        num_agents = 2
        
    # Create agents
    for i in range(num_agents):
        name = console.input(f"\nAgent {i+1} name: ")
        personality = console.input(f"Agent {i+1} personality/role: ")
        
        # Optional: emotional state
        emotion = console.input(f"Agent {i+1} emotional state (or press enter for neutral): ").strip()
        
        agent = FlexibleAgent(name, personality)
        if emotion:
            agent.set_emotional_state(emotion)
            
        # Optional: goals
        goal = console.input(f"Agent {i+1} goal (or press enter to skip): ").strip()
        if goal:
            agent.add_goal(goal)
            
        system.add_agent(agent)
        
    # Choose mode
    console.print("\n[yellow]Interaction Modes:[/yellow]")
    modes = list(InteractionMode)
    for i, mode in enumerate(modes, 1):
        console.print(f"{i}. {mode.value}")
        
    mode_choice = console.input("\nChoose mode (number): ")
    try:
        selected_mode = modes[int(mode_choice) - 1]
    except:
        selected_mode = InteractionMode.CONVERSATION
        
    # Get context
    context = console.input("\nInitial context or topic: ")
    
    # Get rounds
    rounds = console.input("Number of rounds (1-20): ")
    try:
        rounds = min(20, max(1, int(rounds)))
    except:
        rounds = 5
        
    # Custom instructions
    custom = console.input("Any custom instructions (or press enter): ").strip()
    
    # Run interaction
    await system.run_interaction(
        context,
        rounds=rounds,
        mode=selected_mode,
        custom_instructions=custom,
        allow_interruptions=(selected_mode in [InteractionMode.DEBATE, InteractionMode.BRAINSTORM])
    )
    
    # Ask about export
    if console.input("\nExport conversation? (y/n): ").lower() == 'y':
        system.export_conversation()
        
    # Show stats
    if console.input("Show statistics? (y/n): ").lower() == 'y':
        system.get_statistics()

async def quick_scenario(template_name: str):
    """Run a quick scenario from template"""
    templates = {
        "business": InteractionTemplates.business_meeting(),
        "creative": InteractionTemplates.creative_workshop(),
        "debate": InteractionTemplates.debate_club(),
        "therapy": InteractionTemplates.therapy_session(),
        "game": InteractionTemplates.game_night()
    }
    
    if template_name not in templates:
        console.print(f"[red]Unknown template: {template_name}[/red]")
        return
        
    template = templates[template_name]
    system = FlexibleMultiAgentSystem()
    
    # Create agents from template
    for name, role in template["agent_roles"].items():
        agent = FlexibleAgent(name, role)
        system.add_agent(agent)
        
    # Run interaction
    await system.run_interaction(
        template["context"],
        rounds=5,
        mode=template["mode"]
    )

async def main():
    """Main menu"""
    console.print(Panel.fit(
        "[bold cyan]Flexible Multi-Agent System[/bold cyan]\n"
        "General-purpose agent interactions for any scenario",
        border_style="blue"
    ))
    
    while True:
        console.print("\n[yellow]Options:[/yellow]")
        console.print("1. Interactive Setup (create custom scenario)")
        console.print("2. Business Meeting Template")
        console.print("3. Creative Workshop Template")
        console.print("4. Debate Club Template")
        console.print("5. Therapy Session Template")
        console.print("6. Game Night Template")
        console.print("7. Quick Test (2 agents, simple conversation)")
        console.print("0. Exit")
        
        choice = console.input("\n[cyan]Choice: [/cyan]")
        
        if choice == "0":
            console.print("[green]Goodbye! 👋[/green]")
            break
        elif choice == "1":
            await interactive_setup()
        elif choice == "2":
            await quick_scenario("business")
        elif choice == "3":
            await quick_scenario("creative")
        elif choice == "4":
            await quick_scenario("debate")
        elif choice == "5":
            await quick_scenario("therapy")
        elif choice == "6":
            await quick_scenario("game")
        elif choice == "7":
            # Quick test
            system = FlexibleMultiAgentSystem()
            agent1 = FlexibleAgent("Alice", "You are friendly and curious.")
            agent2 = FlexibleAgent("Bob", "You are analytical and thoughtful.")
            system.add_agent(agent1)
            system.add_agent(agent2)
            await system.run_interaction("Let's discuss the future of technology", rounds=3)
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
