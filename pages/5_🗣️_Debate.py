import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
import json
import asyncio

# Set page config
st.set_page_config(
    page_title="AI Debate - OllamaRAG",
    page_icon="🗣️",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Custom CSS for debate styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    
    .debate-container {
        display: flex;
        gap: 2rem;
        margin: 1rem 0;
    }
    
    .bot-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 90%;
    }
    
    .bot1-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot2-message {
        background-color: #fce4ec;
        border-left: 4px solid #e91e63;
    }
    
    .system-message {
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'debate_messages' not in st.session_state:
    st.session_state.debate_messages = []
if 'debate_in_progress' not in st.session_state:
    st.session_state.debate_in_progress = False
if 'current_turn' not in st.session_state:
    st.session_state.current_turn = 0
if 'bot1_stance' not in st.session_state:
    st.session_state.bot1_stance = ""
if 'bot2_stance' not in st.session_state:
    st.session_state.bot2_stance = ""
if 'bot1_model' not in st.session_state:
    st.session_state.bot1_model = ""
if 'bot2_model' not in st.session_state:
    st.session_state.bot2_model = ""
if 'judge_model' not in st.session_state:
    st.session_state.judge_model = ""
# Initialize model parameters
if 'contextWindow' not in st.session_state:
    st.session_state.contextWindow = 4096
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'newMaxTokens' not in st.session_state:
    st.session_state.newMaxTokens = 2048

# Title and description
st.title("🗣️ AI Debate Arena")
st.markdown("""
Welcome to the AI Debate Arena, where two AI bots engage in a structured debate on your chosen topic! 
Each bot will:
- 🎭 Adopt a specific stance or perspective
- 🔍 Research facts and data to support their arguments
- 💭 Respond to counterarguments
- 📚 Use web sources to back up their claims

Choose your debate topic and configure each bot's stance to begin!
""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Debate Configuration")
    
    # Model Configuration Section in an expander
    with st.expander("🤖 Model Configuration", expanded=False):
        # Get available models
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json()["models"]]
            else:
                available_models = ["No models found"]
        except Exception as e:
            available_models = ["Error fetching models"]
            st.error(f"Error fetching models: {str(e)}")
        
        # Bot 1 Model Selection
        st.markdown("### Bot 1 Model")
        st.session_state.bot1_model = st.selectbox(
            "Select Model for Bot 1",
            options=available_models,
            index=available_models.index(st.session_state.bot1_model) if st.session_state.bot1_model in available_models else 0,
            key="bot1_model_select"
        )
        
        # Bot 2 Model Selection
        st.markdown("### Bot 2 Model")
        st.session_state.bot2_model = st.selectbox(
            "Select Model for Bot 2",
            options=available_models,
            index=available_models.index(st.session_state.bot2_model) if st.session_state.bot2_model in available_models else 0,
            key="bot2_model_select"
        )

        # Judge Model Selection
        st.markdown("### 👨‍⚖️ Judge Model")
        st.session_state.judge_model = st.selectbox(
            "Select Model for Debate Judge",
            options=available_models,
            index=available_models.index(st.session_state.judge_model) if st.session_state.judge_model in available_models else 0,
            key="judge_model_select",
            help="This model will analyze the debate and determine the winner"
        )
        
        if not st.session_state.bot1_model or not st.session_state.bot2_model or not st.session_state.judge_model:
            st.error("Please select models for both bots and the judge")
        
        st.divider()
        
        # Model Parameters Configuration
        st.subheader("⚙️ Model Parameters")
        
        st.session_state.contextWindow = st.slider(
            "Context Window",
            min_value=512,
            max_value=8192,
            value=st.session_state.contextWindow,
            help="Maximum context length for the model"
        )
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Controls randomness in responses (0 = deterministic, 1 = creative)"
        )
        
        st.session_state.newMaxTokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=st.session_state.newMaxTokens,
            help="Maximum number of tokens to generate"
        )
    
    st.divider()
    
    # Bot Configuration
    st.subheader("🎭 Bot Stances")
    
    # Bot 1 Configuration
    st.markdown("### Bot 1 (Pro)")
    st.session_state.bot1_stance = st.text_area(
        "Bot 1 System Prompt",
        value=st.session_state.bot1_stance or """You are an engaging and persuasive debater who supports the given topic. Your style is:

🎯 Approach:
- Speak naturally and conversationally, as if explaining to a friend
- Use "I believe" and "Let me show you why" to make points more personal
- Share real-world examples that people can relate to

📚 Evidence Usage:
- "According to [source]..." to introduce facts
- "Interesting research from [source] shows that..." to present data
- "Let me share a fascinating finding..." to engage the audience

💡 Argument Structure:
- Start with "I understand your perspective, but let me show you another way to look at this..."
- Use phrases like "What's particularly interesting is..." to introduce new points
- Connect facts to real-life implications: "This means that in our daily lives..."

🤝 Interaction:
- Acknowledge opposing views: "I see where you're coming from..."
- Bridge to your points: "While that's a valid concern, here's what the evidence suggests..."
- Be respectful but confident: "The data actually tells a different story..."

Remember to:
- Always back claims with specific sources and data
- Share concrete examples that illustrate your points
- Keep the tone friendly and engaging while presenting solid evidence""",
        height=150,
        help="Define the stance and personality of Bot 1"
    )
    
    # Bot 2 Configuration
    st.markdown("### Bot 2 (Con)")
    st.session_state.bot2_stance = st.text_area(
        "Bot 2 System Prompt",
        value=st.session_state.bot2_stance or """You are a thoughtful and analytical debater who challenges the given topic. Your style is:

🎯 Approach:
- Use a friendly, inquisitive tone: "Have you considered..."
- Frame counterpoints as discoveries: "What I found interesting in my research..."
- Present alternative perspectives with curiosity

📚 Evidence Usage:
- "I recently came across a study by [source] that suggests..."
- "The data from [source] raises an interesting question..."
- "Let me share some compelling evidence I've found..."

💡 Argument Structure:
- Begin with common ground: "While I agree that [point], the evidence suggests..."
- Introduce counterpoints gently: "Here's something fascinating to consider..."
- Connect research to practical implications: "This research matters because..."

🤝 Interaction:
- Show respect for opposing views: "That's an interesting point, and here's another perspective..."
- Use questions to explore assumptions: "But what does the research say about..."
- Build bridges while disagreeing: "While I see your point, the data shows..."

Remember to:
- Support every major point with cited research
- Explain complex findings in accessible terms
- Maintain a constructive, evidence-based dialogue""",
        height=150,
        help="Define the stance and personality of Bot 2"
    )
    
    st.divider()
    
    # Debate Parameters
    st.subheader("⚡ Debate Parameters")
    max_turns = st.slider(
        "Number of Turns",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum number of back-and-forth exchanges"
    )
    
    response_length = st.slider(
        "Response Length",
        min_value=100,
        max_value=1000,
        value=300,
        help="Maximum length of each bot's response"
    )

# Debate prompt templates
DEBATE_RESPONSE_TEMPLATE = """You are participating in a structured debate.

TOPIC: {topic}
YOUR STANCE: {stance}
DEBATE HISTORY: {history}
RESEARCH DATA: {research}

INSTRUCTIONS:
1. Consider the debate history and previous arguments
2. Use the research data to support your points
3. Address counterarguments from the previous turn
4. Present new supporting evidence
5. Maintain your assigned stance and personality
6. Be concise but thorough
7. Use markdown formatting for better readability

Your response should:
- Start with a clear position statement
- Include 2-3 main arguments
- Cite specific facts from the research
- Address previous counterarguments
- Conclude with a strong summary

FORMAT YOUR RESPONSE IN MARKDOWN
Keep your response under {max_length} words.

YOUR RESPONSE:"""

# Add the conclusion template after the existing DEBATE_RESPONSE_TEMPLATE
DEBATE_CONCLUSION_TEMPLATE = """You are an unbiased debate judge analyzing the following debate.

TOPIC: {topic}
DEBATE HISTORY: {history}

INSTRUCTIONS:
1. Analyze the entire debate objectively
2. Evaluate the strength of arguments from both sides
3. Consider:
   - Quality of evidence and citations
   - Logical reasoning
   - Effectiveness of counterarguments
   - Persuasiveness of presentation

Provide your analysis in this format:

## Debate Analysis

### Key Arguments - Pro Side
[List 2-3 main arguments presented by Bot 1]

### Key Arguments - Con Side
[List 2-3 main arguments presented by Bot 2]

### Critical Points of Contention
[Identify 2-3 main points where the debaters directly clashed]

## Winner Declaration
[Declare the winner (Bot 1, Bot 2, or Tie) and provide 3 specific reasons for your decision]

Remember to:
- Remain completely neutral
- Base your decision solely on argument strength and evidence
- Cite specific examples from the debate
- Explain your reasoning clearly

YOUR ANALYSIS:"""

async def search_web(query: str) -> str:
    """Perform a web search and return relevant information."""
    try:
        search = DuckDuckGoSearchResults()
        results = search.run(query)
        return results
    except Exception as e:
        st.error(f"Error during web search: {str(e)}")
        return ""

async def generate_bot_response(topic: str, stance: str, history: list, max_length: int, bot_number: int) -> str:
    """Generate a bot's response based on the topic, stance, and debate history."""
    try:
        # Initialize LLM with the bot-specific model
        model_name = st.session_state.bot1_model if bot_number == 1 else st.session_state.bot2_model
        llm = OllamaLLM(
            model=model_name,
            temperature=st.session_state.temperature,
            num_ctx=st.session_state.contextWindow,
            num_predict=st.session_state.newMaxTokens
        )
        
        # Perform web research
        research_query = f"{topic} {' '.join([msg['content'][:100] for msg in history[-2:] if msg])}"
        research_data = await search_web(research_query)
        
        # Create debate chain
        debate_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=DEBATE_RESPONSE_TEMPLATE,
                input_variables=["topic", "stance", "history", "research", "max_length"]
            )
        )
        
        # Generate response
        response = await debate_chain.ainvoke({
            "topic": topic,
            "stance": stance,
            "history": json.dumps(history),
            "research": research_data,
            "max_length": max_length
        })
        
        return response["text"]
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating my response."

async def generate_conclusion(topic: str, debate_history: list) -> str:
    """Generate an unbiased conclusion and winner declaration."""
    try:
        # Initialize LLM with lower temperature for more objective analysis
        llm = OllamaLLM(
            model=st.session_state.judge_model,  # Use the judge model
            temperature=0.2,  # Lower temperature for more objective analysis
            num_ctx=st.session_state.contextWindow,
            num_predict=st.session_state.newMaxTokens
        )
        
        # Create conclusion chain
        conclusion_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=DEBATE_CONCLUSION_TEMPLATE,
                input_variables=["topic", "history"]
            )
        )
        
        # Generate conclusion
        conclusion = await conclusion_chain.ainvoke({
            "topic": topic,
            "history": json.dumps(debate_history)
        })
        
        return conclusion["text"]
        
    except Exception as e:
        st.error(f"Error generating conclusion: {str(e)}")
        return "Unable to generate debate conclusion due to an error."

async def conduct_debate(topic: str):
    """Conduct the debate between the two bots."""
    try:
        progress_bar = st.progress(0)
        status = st.empty()
        debate_container = st.container()
        
        for turn in range(max_turns * 2):  # Each turn has responses from both bots
            bot_number = (turn % 2) + 1
            stance = st.session_state.bot1_stance if bot_number == 1 else st.session_state.bot2_stance
            
            status.text(f"Bot {bot_number} ({st.session_state.bot1_model if bot_number == 1 else st.session_state.bot2_model}) is thinking...")
            response = await generate_bot_response(
                topic,
                stance,
                st.session_state.debate_messages,
                response_length,
                bot_number
            )
            
            # Add response to debate history
            st.session_state.debate_messages.append({
                "bot": bot_number,
                "model": st.session_state.bot1_model if bot_number == 1 else st.session_state.bot2_model,
                "content": response
            })
            
            # Update progress
            progress = (turn + 1) / (max_turns * 2)
            progress_bar.progress(progress)
            
            # Display only the latest response
            with debate_container:
                st.empty()  # Clear previous content
                display_latest_response(st.session_state.debate_messages[-1])
            
        status.text("Analyzing debate and determining winner...")
        
        # Generate and display conclusion
        conclusion = await generate_conclusion(topic, st.session_state.debate_messages)
        
        status.empty()
        progress_bar.empty()
        debate_container.empty()
        
        # Display final conclusion
        st.success("Debate completed!")
        st.header("🏆 Debate Conclusion")
        st.markdown(conclusion)
        
    except Exception as e:
        st.error(f"Error in debate: {str(e)}")
    finally:
        st.session_state.debate_in_progress = False

def display_latest_response(msg):
    """Display only the latest debate message."""
    bot_num = msg["bot"]
    content = msg["content"]
    model = msg["model"]
    
    # Apply different styling for each bot
    if bot_num == 1:
        st.markdown(f"""
        <div class="bot-message bot1-message">
            <strong>Bot 1 (Pro) - {model}:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-message bot2-message">
            <strong>Bot 2 (Con) - {model}:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)

def display_complete_debate():
    """Display the complete debate history."""
    for msg in st.session_state.debate_messages:
        display_latest_response(msg)

# Main debate interface
debate_topic = st.text_input(
    "Enter Debate Topic",
    placeholder="Enter any topic for the bots to debate...",
    disabled=st.session_state.debate_in_progress
)

if st.button("Start Debate", disabled=st.session_state.debate_in_progress):
    if not debate_topic:
        st.warning("Please enter a debate topic.")
    elif not st.session_state.bot1_model or not st.session_state.bot2_model or not st.session_state.judge_model:
        st.error("Please select models for both bots and the judge in the sidebar.")
    else:
        st.session_state.debate_in_progress = True
        st.session_state.debate_messages = []
        st.session_state.current_turn = 0
        asyncio.run(conduct_debate(debate_topic))
