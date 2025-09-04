# Advanced LangChain + Ollama Multi-Agent System

A production-ready implementation of a sophisticated multi-agent system combining LangChain and Ollama with advanced features including code execution, human-in-the-loop controls, entropy-based randomness, and creative interaction patterns.

## 🚀 Features

### Core Capabilities
- **Multi-Agent Orchestration**: Supervisor architecture with specialized agents (Researcher, Coder, Analyst, Critic, Creative)
- **LangGraph Integration**: State-based workflow management with persistence and checkpointing
- **Secure Code Execution**: Sandboxed environment with security validation
- **Human-in-the-Loop**: Interactive review points with entropy collection from human behavior
- **Creative Patterns**: Debates, collaborations, competitions, and storytelling modes

### Advanced Features
- **Entropy Collection System**:
  - Mouse movement entropy using chaos theory
  - Keystroke dynamics (dwell and flight times)
  - Circadian rhythm-based randomness
  - Time-based entropy from human interactions

- **Agent Personalities**:
  - Distinct communication styles
  - Role-specific expertise
  - Collaborative and competitive behaviors
  - Self-modifying capabilities

- **Security & Safety**:
  - Code validation before execution
  - Resource limits and timeouts
  - Sandbox isolation
  - Human approval gates

## 📋 Prerequisites

- Python 3.9+
- Ollama installed and running locally
- SQLite for state persistence
- 8GB+ RAM recommended for running local LLMs

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository>
cd advanced_multiagent_system
```

2. **Install Ollama** (if not already installed):
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

3. **Pull required models**:
```bash
ollama pull llama3.1
ollama pull mistral  # Optional alternative
```

4. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

5. **Set up environment** (optional):
```bash
cp .env.example .env
# Edit .env with your configurations
```

## 🎮 Usage

### Basic Usage

```python
import asyncio
from multi_agent_system import AdvancedMultiAgentSystem

async def main():
    # Initialize the system
    system = AdvancedMultiAgentSystem(model="llama3.1")
    
    # Run a task
    result = await system.run("Create a web scraper for news articles")
    
    # Access results
    print(f"Messages: {len(result['messages'])}")
    print(f"Entropy collected: {result.get('entropy_pool', [])}")

asyncio.run(main())
```

### Creative Interaction Patterns

#### 1. Debates
```python
from multi_agent_system import CreativeInteractionManager

interaction_manager = CreativeInteractionManager(system)

result = await interaction_manager.run_debate(
    topic="Should AI have consciousness?",
    positions=[
        "Yes - enhances interaction",
        "No - creates ethical issues"
    ]
)
```

#### 2. Collaborative Projects
```python
result = await interaction_manager.run_collaboration(
    project="Build a recommendation system"
)
```

#### 3. Storytelling
```python
result = await interaction_manager.run_storytelling(
    premise="A robot learns to paint",
    genre="philosophical fiction"
)
```

#### 4. Competitions
```python
result = await interaction_manager.run_competition(
    challenge="Optimize database queries",
    scoring_criteria=["Speed", "Efficiency", "Scalability"]
)
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────┐
│           SUPERVISOR AGENT              │
│         (Orchestration Layer)           │
└────────────┬───────────────────────────┘
             │
    ┌────────┴────────┬────────┬────────┬────────┐
    ▼                 ▼        ▼        ▼        ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│RESEARCHER│  │  CODER   │  │ ANALYST  │  │ CREATIVE │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
                    │
                    ▼
            ┌──────────────┐
            │   SANDBOX    │
            │  EXECUTION   │
            └──────────────┘
```

### State Management

The system uses LangGraph's state management with:
- **Message History**: Full conversation tracking
- **Agent Outputs**: Individual agent results
- **Execution History**: Code execution logs
- **Entropy Pool**: Collected randomness values
- **Security Flags**: Safety indicators

### Agent Roles

| Agent | Personality | Primary Functions |
|-------|-------------|-------------------|
| **Supervisor** | Strategic leader | Task allocation, orchestration |
| **Researcher** | Meticulous analyst | Information gathering, verification |
| **Coder** | Software architect | Implementation, testing |
| **Analyst** | Data expert | Pattern recognition, insights |
| **Critic** | Quality guardian | Review, improvement suggestions |
| **Creative** | Innovation catalyst | Novel solutions, brainstorming |

## 🔒 Security Considerations

### Code Execution Safety
- Pre-execution validation for dangerous patterns
- Configurable resource limits
- Timeout mechanisms
- Sandboxed execution environment

### Human-in-the-Loop Controls
- Approval gates for critical operations
- Review points before major decisions
- Feedback collection mechanisms
- Emergency stop capabilities

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=multi_agent_system tests/
```

## 📊 Performance Optimization

### Model Selection
- **llama3.1**: Best balance of speed and quality
- **mistral**: Faster responses, good for simple tasks
- **llama2:70b**: Higher quality for complex reasoning (requires more resources)

### Resource Management
```python
# Configure for your system
system = AdvancedMultiAgentSystem(
    model="llama3.1",
    use_local=True  # Use local Ollama
)

# Adjust LLM parameters
llm = ChatOllama(
    model="llama3.1",
    temperature=0.7,
    num_ctx=4096,    # Context window
    num_gpu=1,        # GPU layers
    num_thread=8      # CPU threads
)
```

## 🎯 Use Cases

### Software Development
- Code generation with testing
- Architecture design
- Code review and optimization
- Documentation generation

### Research & Analysis
- Literature review
- Data analysis pipelines
- Market research
- Competitive analysis

### Creative Applications
- Story generation
- Game design
- Content creation
- Brainstorming sessions

### Educational
- Interactive tutorials
- Problem-solving demonstrations
- Concept exploration
- Socratic dialogues

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📚 Advanced Topics

### Custom Agent Creation
```python
class CustomAgent:
    def __init__(self):
        self.personality = """Your custom personality..."""
    
    async def process(self, state):
        # Custom logic
        return updated_state
```

### Entropy Sources Integration
```python
# Add custom entropy source
class WeatherEntropy:
    def get_entropy(self):
        # Use weather API
        return weather_based_randomness
```

### Tool Integration
```python
from langchain_core.tools import tool

@tool
def custom_tool(query: str) -> str:
    """Custom tool description"""
    return result

# Bind to agent
agent_with_tools = llm.bind_tools([custom_tool])
```

## 🔗 References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Multi-Agent Systems Research](https://arxiv.org/abs/2308.08155)

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- LangChain team for the incredible framework
- Ollama for local LLM execution
- The open-source AI community

---

**Built with ❤️ for the those in the field of CS and Machine learning for going out of their way to democratize their most brillant cutting edge discovers and designs for all to learn to use!**
