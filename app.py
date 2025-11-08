import os
import streamlit as st
from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

# Tools (community)
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun

"""
LangChain 1.0+ no longer exposes `create_react_agent` or `AgentExecutor` in
`langchain.agents`. Use `create_agent` which returns a LangGraph compiled graph.
"""
from langchain.agents import create_agent

# Streamlit callback (moved to community)
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

load_dotenv()
st.set_page_config(page_title="🔎 LangChain - Chat with Search", page_icon="🔎")
st.title("🔎 LangChain - Chat with Search")
st.write(
    "This app uses a ReAct agent with Wikipedia, arXiv, and DuckDuckGo tools. "
    "Thoughts/actions stream live below."
)

# --- Sidebar ---
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password")
if not api_key:
    st.sidebar.info("Enter your Groq API key to run queries.")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Build LLM + tools once (lazy init after key present) ---
def build_executor(groq_key: str):
    """Build a LangChain 1.0+ tool-using agent graph via create_agent."""
    llm = ChatGroq(
        groq_api_key=groq_key,
        model="openai/gpt-oss-20b",  # works well; change if you prefer
        temperature=0,
        streaming=True,
    )

    # Tools
    wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)

    wiki = WikipediaQueryRun(api_wrapper=wiki_api)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_api)
    ddg = DuckDuckGoSearchRun()  # no name arg in constructor

    tools = [wiki, arxiv, ddg]

    system_prompt = (
        "You are a helpful research assistant. Use tools when needed. "
        "Cite sources when possible. Keep answers concise."
    )

    # create_agent returns a CompiledStateGraph (LangGraph). We'll invoke it
    # with {"messages": [("human", "...")]} and read the final AI message.
    agent_graph = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    return agent_graph


def _extract_answer(run_output) -> str:
    """Extract final AI text from create_agent's output schema."""
    try:
        messages = run_output.get("messages", []) if isinstance(run_output, dict) else []
        last = messages[-1] if messages else None
        if last is None:
            return ""
        # Support LC message objects and plain dicts
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content")
        if isinstance(content, list):
            # Merge parts if model returned structured content
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    parts.append(str(p.get("text", "")))
            return "".join(parts).strip()
        return str(content or "").strip()
    except Exception:
        return ""

# --- Chat input ---
if prompt := st.chat_input("Ask me something (e.g., 'Latest on diffusion models?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.chat_message("assistant").warning("Please provide your Groq API key in the sidebar.")
    else:
        agent_graph = build_executor(api_key)
        with st.chat_message("assistant"):
            # Note: LangChain 1.0+ agent graphs don't use the legacy callbacks directly.
            # We run a simple invoke and display the final output.
            try:
                with st.spinner("Thinking with tools…"):
                    result = agent_graph.invoke({"messages": [("human", prompt)]})
                answer = _extract_answer(result)
                if not answer:
                    answer = "(No answer returned)"
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
