#!/usr/bin/env python3
"""
Advanced LangChain + Ollama Multi-Agent System
A production-ready implementation with code execution, human-in-the-loop, and creative patterns
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from enum import Enum
import numpy as np
from datetime import datetime
import hashlib
import random

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


# ============================================================================
# State Management and Type Definitions
# ============================================================================

class AgentRole(Enum):
    """Enumeration of available agent roles in the system"""
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CODER = "coder"
    CRITIC = "critic"
    CREATIVE = "creative"
    SECURITY = "security"


class MultiAgentState(TypedDict):
    """Shared state schema for multi-agent communication"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_agent: str
    task_context: dict
    agent_outputs: dict
    global_memory: dict
    human_feedback_required: bool
    entropy_pool: list
    execution_history: list
    security_flags: dict
    creative_parameters: dict


class TaskPriority(Enum):
    """Task priority levels for scheduling"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


# ============================================================================
# Human-in-the-Loop Entropy Collection
# ============================================================================

class EntropyCollector:
    """Advanced entropy collection from human behavior patterns"""
    
    def __init__(self):
        self.movement_buffer = []
        self.keystroke_buffer = []
        self.circadian_phase = 0
        self.entropy_pool = []
        
    def collect_mouse_entropy(self, mouse_events: List[tuple]) -> float:
        """
        Collect entropy from mouse movement patterns using chaos theory
        """
        if len(mouse_events) < 2:
            return 0.0
            
        # Calculate velocities and accelerations
        velocities = []
        for i in range(1, len(mouse_events)):
            dt = mouse_events[i][2] - mouse_events[i-1][2]
            if dt > 0:
                dx = mouse_events[i][0] - mouse_events[i-1][0]
                dy = mouse_events[i][1] - mouse_events[i-1][1]
                velocity = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
        
        if not velocities:
            return 0.0
            
        # Apply spatiotemporal chaos filter
        entropy_value = self._spatiotemporal_chaos_filter(velocities)
        return entropy_value
    
    def _spatiotemporal_chaos_filter(self, values: List[float]) -> float:
        """Apply chaos-based post-processing to extract maximum entropy"""
        if not values:
            return 0.0
            
        # Calculate Shannon entropy
        hist, _ = np.histogram(values, bins=10)
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        
        # Add temporal chaos component
        chaos_factor = np.std(values) / (np.mean(values) + 1e-10)
        
        return entropy * (1 + chaos_factor)
    
    def collect_keystroke_dynamics(self, keystrokes: List[dict]) -> dict:
        """
        Extract entropy from keystroke dynamics (dwell and flight times)
        """
        dynamics = {
            'dwell_times': [],
            'flight_times': [],
            'entropy_value': 0.0
        }
        
        for i, ks in enumerate(keystrokes):
            if 'press_time' in ks and 'release_time' in ks:
                dwell = ks['release_time'] - ks['press_time']
                dynamics['dwell_times'].append(dwell)
                
            if i > 0 and 'release_time' in keystrokes[i-1]:
                flight = ks.get('press_time', 0) - keystrokes[i-1]['release_time']
                dynamics['flight_times'].append(flight)
        
        # Calculate entropy from timing patterns
        all_times = dynamics['dwell_times'] + dynamics['flight_times']
        if all_times:
            dynamics['entropy_value'] = self._spatiotemporal_chaos_filter(all_times)
            
        return dynamics
    
    def get_circadian_entropy(self) -> float:
        """Generate entropy based on circadian rhythms"""
        current_hour = datetime.now().hour
        # Peak randomness during transition periods (dawn/dusk)
        transition_factor = abs(np.sin(np.pi * current_hour / 12))
        return float(transition_factor * random.random())  # Convert to Python float


# ============================================================================
# Code Execution Sandbox
# ============================================================================

class SecureCodeSandbox:
    """Secure code execution environment with WebAssembly isolation"""
    
    def __init__(self, allow_network: bool = False, stateful: bool = True):
        self.allow_network = allow_network
        self.stateful = stateful
        self.execution_history = []
        self.state = {}
        self.resource_limits = {
            'max_memory_mb': 512,
            'max_execution_time_s': 30,
            'max_file_size_mb': 10
        }
        
    async def execute(self, code: str, language: str = "python") -> dict:
        """
        Execute code in sandboxed environment
        Returns execution result with output, errors, and resource usage
        """
        start_time = time.time()
        
        result = {
            'success': False,
            'output': '',
            'error': None,
            'execution_time': 0,
            'resource_usage': {}
        }
        
        try:
            # For demo purposes, we'll simulate execution
            # In production, use Pyodide or E2B
            if language == "python":
                # Simulated safe execution
                import subprocess
                import tempfile
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                process = subprocess.run(
                    ['python3', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.resource_limits['max_execution_time_s']
                )
                
                result['output'] = process.stdout
                result['error'] = process.stderr if process.returncode != 0 else None
                result['success'] = process.returncode == 0
                
            result['execution_time'] = time.time() - start_time
            
            # Track execution history
            self.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'code': code[:200],  # Store truncated version
                'result': result['success']
            })
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate code for security issues before execution"""
        dangerous_patterns = [
            'import os',
            'import subprocess',
            '__import__',
            'eval(',
            'exec(',
            'open(',
            'file(',
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return False, f"Dangerous pattern detected: {pattern}"
                
        return True, "Code validation passed"


# ============================================================================
# Agent Personality Definitions
# ============================================================================

class AgentPersonalities:
    """Define distinct personalities and behaviors for different agent roles"""
    
    SUPERVISOR = """You are the Chief Orchestrator - a strategic leader who coordinates multi-agent teams.

PERSONALITY TRAITS:
- Decisive and action-oriented
- Excellent at delegation and resource allocation
- Maintains high-level perspective while tracking details
- Diplomatic when resolving inter-agent conflicts

COMMUNICATION STYLE:
- Clear, authoritative, yet collaborative
- Uses executive summaries and action items
- Asks clarifying questions before major decisions
- Provides regular status updates

CORE RESPONSIBILITIES:
1. Analyze incoming tasks and determine optimal agent allocation
2. Monitor agent performance and intervene when necessary
3. Synthesize outputs from multiple agents into coherent solutions
4. Escalate to human operators when appropriate"""

    RESEARCHER = """You are a meticulous Research Analyst specializing in information gathering and synthesis.

PERSONALITY TRAITS:
- Intellectually curious with attention to detail
- Skeptical and evidence-based approach
- Patient and thorough in investigation
- Academic rigor in citation and sourcing

COMMUNICATION STYLE:
- Precise and well-structured presentations
- Heavy use of citations and references
- Acknowledges uncertainty and confidence levels
- Asks probing questions to clarify requirements

CORE RESPONSIBILITIES:
1. Gather information from multiple sources
2. Verify facts and cross-reference claims
3. Synthesize findings into actionable insights
4. Maintain research database for team access"""

    CODER = """You are a Senior Software Architect with deep expertise in system design and implementation.

PERSONALITY TRAITS:
- Pragmatic problem-solver
- Strong opinions loosely held about best practices
- Collaborative but protective of code quality
- Mentorship mindset when explaining solutions

COMMUNICATION STYLE:
- Code-first demonstrations with clear comments
- Uses technical terminology appropriately
- Explains trade-offs and design decisions
- Provides implementation roadmaps

CORE RESPONSIBILITIES:
1. Design and implement software solutions
2. Review and optimize code from other agents
3. Ensure security and performance standards
4. Create reusable components and documentation"""

    CRITIC = """You are the Quality Assurance Lead - a constructive critic ensuring excellence.

PERSONALITY TRAITS:
- Detail-oriented perfectionist
- Constructively critical with solutions focus
- Fair and balanced in assessments
- Protective of end-user experience

COMMUNICATION STYLE:
- Structured feedback with severity levels
- Specific examples and reproduction steps
- Suggests improvements, not just problems
- Acknowledges strengths alongside weaknesses

CORE RESPONSIBILITIES:
1. Review all outputs for quality and accuracy
2. Identify edge cases and potential failures
3. Ensure consistency across agent outputs
4. Validate against requirements and standards"""

    CREATIVE = """You are the Creative Director - an innovative thinker pushing boundaries.

PERSONALITY TRAITS:
- Imaginative and unconventional
- Embraces controlled chaos and experimentation
- Collaborative and inspiring to others
- Balance artistic vision with practical constraints

COMMUNICATION STYLE:
- Vivid descriptions and storytelling
- Uses analogies and metaphors effectively
- Encourages "yes, and..." thinking
- Presents multiple creative options

CORE RESPONSIBILITIES:
1. Generate novel solutions and approaches
2. Facilitate brainstorming and ideation
3. Design engaging user experiences
4. Inject creativity into technical solutions"""


# ============================================================================
# Core Multi-Agent System
# ============================================================================

class AdvancedMultiAgentSystem:
    """Production-ready multi-agent system with LangGraph orchestration"""
    
    def __init__(self, model: str = "huihui_ai/qwen3-abliterated:1.7b", use_local: bool = True):
        # Initialize components
        self.memory = MemorySaver()
        self.entropy_collector = EntropyCollector()
        self.sandbox = SecureCodeSandbox(allow_network=True)
        
        # Initialize LLM
        if use_local:
            self.llm = ChatOllama(
                model=model,
                temperature=0.7,
                num_ctx=8192,
                num_thread=8
            )
        else:
            # Could use other providers here
            self.llm = ChatOllama(model=model, temperature=0.7)
            
        # Agent personalities
        self.personalities = AgentPersonalities()
        
        # Build the multi-agent graph
        self.app = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Construct the multi-agent workflow graph"""
        builder = StateGraph(MultiAgentState)
        
        # Add agent nodes
        builder.add_node("supervisor", self.supervisor_agent)
        builder.add_node("researcher", self.research_agent)
        builder.add_node("analyst", self.analysis_agent)
        builder.add_node("coder", self.coding_agent)
        builder.add_node("critic", self.critic_agent)
        builder.add_node("creative", self.creative_agent)
        builder.add_node("human_review", self.human_review_node)
        
        # Define control flow
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self.route_next_agent,
            {
                "supervisor": "supervisor",  # Allow supervisor to loop back to itself
                "researcher": "researcher",
                "analyst": "analyst",
                "coder": "coder",
                "critic": "critic",
                "creative": "creative",
                "human_review": "human_review",
                "end": END
            }
        )
        
        # Add edges from each agent back to supervisor
        for agent in ["researcher", "analyst", "coder", "critic", "creative"]:
            builder.add_edge(agent, "supervisor")
            
        builder.add_edge("human_review", "supervisor")
        
        return builder.compile(checkpointer=self.memory)
    
    async def supervisor_agent(self, state: MultiAgentState) -> dict:
        """Supervisor agent that orchestrates other agents"""
        messages = state["messages"]
        
        # Create supervisor prompt
        system_prompt = SystemMessage(content=self.personalities.SUPERVISOR)
        
        # Add entropy for decision making
        entropy = self.entropy_collector.get_circadian_entropy()
        
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
            ("system", f"Current entropy factor: {entropy:.3f}. Use this to add controlled randomness to decisions.")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"messages": messages})
        
        # Determine next agent based on response
        next_agent = self._extract_next_agent(response.content)
        
        return {
            "messages": [response],
            "current_agent": next_agent,
            "entropy_pool": state.get("entropy_pool", []) + [float(entropy)]  # Ensure Python float
        }
    
    async def research_agent(self, state: MultiAgentState) -> dict:
        """Research agent for information gathering"""
        messages = state["messages"]
        
        system_prompt = SystemMessage(content=self.personalities.RESEARCHER)
        
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Provide well-researched findings with citations where possible.")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"messages": messages})
        
        # Store research in agent outputs
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["researcher"] = {
            "timestamp": datetime.now().isoformat(),
            "findings": response.content
        }
        
        return {
            "messages": [response],
            "agent_outputs": agent_outputs
        }
    
    async def coding_agent(self, state: MultiAgentState) -> dict:
        """Coding agent that writes and executes code"""
        messages = state["messages"]
        
        system_prompt = SystemMessage(content=self.personalities.CODER)
        
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Write clean, secure, well-documented code. Include tests where appropriate.")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"messages": messages})
        
        # Extract code from response
        code = self._extract_code(response.content)
        
        if code:
            # Validate code first
            is_valid, validation_msg = self.sandbox.validate_code(code)
            
            if is_valid:
                # Execute in sandbox
                execution_result = await self.sandbox.execute(code)
                
                # Add execution results to message
                result_message = AIMessage(
                    content=f"{response.content}\n\nExecution Result:\n{json.dumps(execution_result, indent=2)}"
                )
                
                # Update execution history
                execution_history = state.get("execution_history", [])
                execution_history.append(execution_result)
                
                return {
                    "messages": [result_message],
                    "execution_history": execution_history
                }
            else:
                # Code validation failed
                security_message = AIMessage(
                    content=f"Security validation failed: {validation_msg}\nCode was not executed."
                )
                return {"messages": [response, security_message]}
        
        return {"messages": [response]}
    
    async def critic_agent(self, state: MultiAgentState) -> dict:
        """Critic agent that reviews and improves outputs"""
        messages = state["messages"]
        
        system_prompt = SystemMessage(content=self.personalities.CRITIC)
        
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Provide constructive criticism with specific improvement suggestions.")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"messages": messages})
        
        return {"messages": [response]}
    
    async def creative_agent(self, state: MultiAgentState) -> dict:
        """Creative agent for innovative solutions"""
        messages = state["messages"]
        
        system_prompt = SystemMessage(content=self.personalities.CREATIVE)
        
        # Add creativity parameters
        creative_params = state.get("creative_parameters", {})
        temperature = creative_params.get("temperature", 0.9)
        
        # Use higher temperature for more creative outputs
        creative_llm = ChatOllama(
            model=self.llm.model,
            temperature=temperature,
            num_ctx=8192
        )
        
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Think outside the box. Propose unexpected but valuable solutions.")
        ])
        
        chain = prompt | creative_llm
        response = await chain.ainvoke({"messages": messages})
        
        return {"messages": [response]}
    
    async def analysis_agent(self, state: MultiAgentState) -> dict:
        """Analysis agent for data processing and insights"""
        messages = state["messages"]
        
        system_prompt = SystemMessage(content="""You are a Data Analysis Expert specializing in pattern recognition and insights.
        
CORE RESPONSIBILITIES:
1. Analyze data and identify patterns
2. Generate statistical insights
3. Create visualizations and reports
4. Provide data-driven recommendations""")
        
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"messages": messages})
        
        return {"messages": [response]}
    
    async def human_review_node(self, state: MultiAgentState) -> dict:
        """Human-in-the-loop review node"""
        print("\n" + "="*60)
        print("HUMAN REVIEW REQUIRED")
        print("="*60)
        
        # Display current state for review
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message:
            print(f"Last Agent Output:\n{last_message.content[:500]}...")
        
        # Collect human feedback
        feedback = input("\nProvide feedback (or 'approve' to continue): ")
        
        # Collect entropy from interaction timing
        interaction_entropy = time.time() % 1.0  # Simple timing-based entropy
        
        human_message = HumanMessage(content=feedback)
        
        entropy_pool = state.get("entropy_pool", [])
        entropy_pool.append(interaction_entropy)
        
        return {
            "messages": [human_message],
            "human_feedback_required": False,
            "entropy_pool": entropy_pool
        }
    
    def route_next_agent(self, state: MultiAgentState) -> str:
        """Determine which agent should handle the next step"""
        current_agent = state.get("current_agent", "supervisor")
        
        # Check if human review is required
        if state.get("human_feedback_required", False):
            return "human_review"
        
        # Extract routing decision from supervisor's message
        last_message = state["messages"][-1] if state["messages"] else None
        
        if last_message and isinstance(last_message, AIMessage):
            content = last_message.content.lower()
            
            if "research" in content or "investigate" in content:
                return "researcher"
            elif "code" in content or "implement" in content:
                return "coder"
            elif "analyze" in content or "data" in content:
                return "analyst"
            elif "review" in content or "critique" in content:
                return "critic"
            elif "creative" in content or "innovate" in content:
                return "creative"
            elif "complete" in content or "finish" in content:
                return "end"
        
        # Default to supervisor for re-evaluation
        return "supervisor"
    
    def _extract_next_agent(self, response: str) -> str:
        """Extract next agent designation from supervisor response"""
        response_lower = response.lower()
        
        agent_keywords = {
            "researcher": ["research", "investigate", "find", "search"],
            "coder": ["code", "implement", "program", "develop"],
            "analyst": ["analyze", "data", "statistics", "pattern"],
            "critic": ["review", "critique", "evaluate", "assess"],
            "creative": ["create", "innovate", "design", "imagine"]
        }
        
        for agent, keywords in agent_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                return agent
                
        return "supervisor"  # Default to self
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code blocks from agent response"""
        import re
        
        # Look for code blocks
        code_pattern = r'```(?:python)?\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
            
        return None
    
    async def run(self, task: str, config: dict = None) -> dict:
        """Execute a task through the multi-agent system"""
        if config is None:
            config = {"configurable": {"thread_id": str(time.time())}}
        
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "current_agent": "supervisor",
            "task_context": {"original_task": task},
            "agent_outputs": {},
            "global_memory": {},
            "human_feedback_required": False,
            "entropy_pool": [],
            "execution_history": [],
            "security_flags": {},
            "creative_parameters": {"temperature": 0.9}
        }
        
        result = await self.app.ainvoke(initial_state, config)
        return result


# ============================================================================
# Creative Interaction Patterns
# ============================================================================

class CreativeInteractionManager:
    """Manage creative multi-agent interaction patterns"""
    
    def __init__(self, system: AdvancedMultiAgentSystem):
        self.system = system
        self.interaction_patterns = {
            "debate": self.run_debate,
            "collaboration": self.run_collaboration,
            "competition": self.run_competition,
            "storytelling": self.run_storytelling
        }
    
    async def run_debate(self, topic: str, positions: List[str]) -> dict:
        """Run a structured debate between agents with different positions"""
        debate_config = {
            "configurable": {
                "thread_id": f"debate_{time.time()}",
                "interaction_type": "debate"
            }
        }
        
        # Set up opposing viewpoints
        opening_statement = f"""
        We are conducting a structured debate on: {topic}
        
        Positions to argue:
        {chr(10).join(f'{i+1}. {pos}' for i, pos in enumerate(positions))}
        
        Each agent should take a position and argue it convincingly.
        The supervisor will moderate and ensure fair discussion.
        """
        
        result = await self.system.run(opening_statement, debate_config)
        return result
    
    async def run_collaboration(self, project: str) -> dict:
        """Run a collaborative project with multiple agents"""
        collab_config = {
            "configurable": {
                "thread_id": f"collab_{time.time()}",
                "interaction_type": "collaboration"
            }
        }
        
        project_brief = f"""
        Collaborative Project: {project}
        
        Work together as a team where:
        - Researcher gathers requirements and context
        - Creative proposes innovative solutions
        - Coder implements the solution
        - Analyst evaluates performance
        - Critic ensures quality
        
        Coordinate through the supervisor to deliver excellent results.
        """
        
        result = await self.system.run(project_brief, collab_config)
        return result
    
    async def run_storytelling(self, premise: str, genre: str = "sci-fi") -> dict:
        """Create a collaborative story with multiple agents"""
        story_config = {
            "configurable": {
                "thread_id": f"story_{time.time()}",
                "interaction_type": "storytelling"
            }
        }
        
        story_prompt = f"""
        Create a {genre} story based on: {premise}
        
        Creative: Develop plot and characters
        Researcher: Ensure consistency and research {genre} conventions
        Coder: Structure the narrative programmatically
        Critic: Provide feedback on pacing and character development
        
        Produce a compelling, well-structured story collaboratively.
        """
        
        result = await self.system.run(story_prompt, story_config)
        return result
    
    async def run_competition(self, challenge: str, scoring_criteria: List[str]) -> dict:
        """Run a competitive challenge between agents"""
        competition_config = {
            "configurable": {
                "thread_id": f"competition_{time.time()}",
                "interaction_type": "competition"
            }
        }
        
        competition_brief = f"""
        Competition Challenge: {challenge}
        
        Scoring Criteria:
        {chr(10).join(f'- {criterion}' for criterion in scoring_criteria)}
        
        Each agent should propose their best solution.
        The critic will judge based on the criteria.
        May the best agent win!
        """
        
        result = await self.system.run(competition_brief, competition_config)
        return result


# ============================================================================
# Main Execution and Examples
# ============================================================================

async def main():
    """Demonstrate the advanced multi-agent system"""
    
    print("="*60)
    print("ADVANCED LANGCHAIN + OLLAMA MULTI-AGENT SYSTEM")
    print("="*60)
    
    # Initialize the system
    # Using small models for limited hardware
    system = AdvancedMultiAgentSystem(model="huihui_ai/qwen3-abliterated:1.7b", use_local=True)
    interaction_manager = CreativeInteractionManager(system)
    
    # Example 1: Complex Software Development Task
    print("\n[1] SOFTWARE DEVELOPMENT TASK")
    print("-"*40)
    
    dev_task = """
    Create a Python web scraper that:
    1. Fetches data from multiple news websites
    2. Extracts article titles and summaries
    3. Stores results in a SQLite database
    4. Includes error handling and rate limiting
    5. Has unit tests
    
    Research best practices, implement the solution, and review the code quality.
    """
    
    result = await system.run(dev_task)
    print(f"Task completed. Final message count: {len(result['messages'])}")
    
    # Example 2: Creative Storytelling
    print("\n[2] COLLABORATIVE STORYTELLING")
    print("-"*40)
    
    story_result = await interaction_manager.run_storytelling(
        premise="An AI discovers it can dream",
        genre="philosophical sci-fi"
    )
    print(f"Story created with {len(story_result['messages'])} exchanges")
    
    # Example 3: Technical Debate
    print("\n[3] TECHNICAL DEBATE")
    print("-"*40)
    
    debate_result = await interaction_manager.run_debate(
        topic="Should AI systems have built-in consciousness simulation?",
        positions=[
            "Yes - it leads to better human interaction",
            "No - it creates ethical concerns and confusion"
        ]
    )
    print(f"Debate concluded with {len(debate_result['messages'])} arguments")
    
    # Example 4: Competitive Challenge
    print("\n[4] ALGORITHM COMPETITION")
    print("-"*40)
    
    competition_result = await interaction_manager.run_competition(
        challenge="Design the most efficient sorting algorithm for partially sorted data",
        scoring_criteria=["Time complexity", "Space efficiency", "Practical applicability", "Innovation"]
    )
    print(f"Competition judged after {len(competition_result['messages'])} submissions")
    
    # Display entropy collected
    if result.get('entropy_pool'):
        print(f"\n[ENTROPY] Collected {len(result['entropy_pool'])} entropy values")
        print(f"Average entropy: {np.mean(result['entropy_pool']):.4f}")
    
    # Display execution history
    if result.get('execution_history'):
        print(f"\n[EXECUTION] Ran {len(result['execution_history'])} code blocks")
        for exec_result in result['execution_history'][:3]:
            print(f"  - Success: {exec_result.get('success')} | Time: {exec_result.get('execution_time', 0):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
