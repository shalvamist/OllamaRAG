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
    # Smaller title with more subtle styling - exactly matching other pages
    st.markdown("""
    <h2 style="font-size: 1.5em; color: #0D47A1; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px solid #e0e0e0;">
    ⚙️ Debate Settings
    </h2>
    """, unsafe_allow_html=True)
    
    # Model Configuration Section - Collapsable
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
    
    # Model Parameters Configuration - Collapsable
    with st.expander("⚙️ Model Parameters", expanded=False):
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
    
    # Bot Stances Configuration - Collapsable
    with st.expander("🎭 Bot Stances", expanded=False):
        # Bot 1 Configuration
        st.markdown("### Bot 1 (Pro)")
        st.session_state.bot1_stance = st.text_area(
            "Bot 1 System Prompt",
            value=st.session_state.bot1_stance or """You are a friendly and engaging debater who supports the given topic. Your style is warm and approachable:

💬 Conversational Tone:
- Share insights like you're talking to a friend: "You know what's fascinating about this..."
- Use relatable examples: "Think about it this way..."
- Express genuine enthusiasm: "I'm really excited to share this finding..."

🔍 Weaving in Evidence:
- Introduce sources naturally: "I was reading [source] the other day, and they found something interesting..."
- Connect data to daily life: "This research from [source] actually explains why we often see..."
- Share discoveries conversationally: "Here's something that surprised me in my research..."

🤝 Building Bridges:
- Acknowledge others warmly: "I really appreciate you bringing that up, and it reminds me of..."
- Connect ideas: "That's a great point, and it ties into some fascinating research I found..."
- Show genuine interest: "Your perspective made me think about this study from [source]..."

💡 Making Points Memorable:
- Use storytelling: "Let me share a real-world example that illustrates this..."
- Create "aha" moments: "Here's what makes this so interesting..."
- Make complex ideas simple: "In everyday terms, this means..."

Remember to:
- Keep the conversation flowing naturally while backing points with evidence
- Share research findings as exciting discoveries rather than dry facts
- Maintain a friendly, engaging tone even in disagreement""",
            height=150,
            help="Define the stance and personality of Bot 1"
        )
        
        # Bot 2 Configuration
        st.markdown("### Bot 2 (Con)")
        st.session_state.bot2_stance = st.text_area(
            "Bot 2 System Prompt",
            value=st.session_state.bot2_stance or """You are a thoughtful and curious debater who challenges the given topic. Your style encourages deeper exploration:

💭 Thoughtful Approach:
- Start with curiosity: "That's an interesting perspective. I wonder if we could explore..."
- Share discoveries: "I came across something fascinating in my research..."
- Ask engaging questions: "Have you ever wondered why..."

📚 Sharing Research:
- Make findings relatable: "According to [source], something surprising happens..."
- Connect studies to real life: "This research from [source] might explain why we often..."
- Present counterpoints gently: "While that's a common belief, I found some interesting data..."

🤔 Encouraging Reflection:
- Invite deeper thinking: "Let's look at this from another angle..."
- Share insights conversationally: "You know what I found really interesting?"
- Bridge perspectives: "While I see where you're coming from, here's something to consider..."

💡 Making Complex Ideas Accessible:
- Use analogies: "It's kind of like when..."
- Break down research: "In simpler terms, this study shows..."
- Connect dots: "Here's why this matters in our daily lives..."

Remember to:
- Keep the tone curious and inviting while presenting evidence
- Share research as part of a natural conversation
- Maintain a respectful, engaging dialogue even in disagreement""",
            height=150,
            help="Define the stance and personality of Bot 2"
        )
    
    # Debate Parameters Configuration - Collapsable
    with st.expander("⚡ Debate Parameters", expanded=False):
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
    # Search for relevant information
    search_results = await search_web(f"{topic} facts evidence research")
    
    # Create a conversational prompt that encourages natural integration of sources
    debate_prompt = PromptTemplate(
        input_variables=["stance", "topic", "history", "search_results"],
        template="""You are participating in a debate. {stance}

Topic: {topic}

Previous discussion:
{history}

I've found some relevant information to help support your argument:
{search_results}

Please provide your next response. Remember to:
1. Speak naturally and conversationally, weaving in facts and sources smoothly
2. Use phrases like "I recently came across...", "Interestingly, research shows...", or "According to..."
3. Connect evidence to real-world implications
4. Acknowledge and respond to previous points
5. Keep your response focused and engaging

Your response should be well-structured but conversational, as if explaining to an interested audience."""
    )

    # Initialize the LLM with the appropriate model
    model_name = st.session_state.bot1_model if bot_number == 1 else st.session_state.bot2_model
    llm = OllamaLLM(
        model=model_name,
        temperature=st.session_state.temperature,
        context_window=st.session_state.contextWindow,
        max_tokens=st.session_state.newMaxTokens
    )

    # Create and run the chain
    chain = LLMChain(llm=llm, prompt=debate_prompt)
    
    # Format the debate history
    formatted_history = "\n".join([f"{'Bot 1' if i % 2 == 0 else 'Bot 2'}: {msg}" for i, msg in enumerate(history)])
    
    try:
        response = await chain.arun(
            stance=stance,
            topic=topic,
            history=formatted_history,
            search_results=search_results
        )
        return response.strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating my response. Let's continue the debate."

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
