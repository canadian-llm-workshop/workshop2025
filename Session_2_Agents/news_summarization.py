import getpass
import json
import operator
import os
import time
from typing import Annotated, Dict, List, Tuple, TypedDict

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


def get_api_key(var: str):
    if not os.environ.get(var):
        return getpass.getpass(f"{var}: ")
    else:
        return os.environ.get(var)

API_KEY = get_api_key("API_KEY")
llm = ChatOpenAI(model="Nova-Micro-v1", base_url="https://api.aibrary.dev/v0", api_key=API_KEY)



# Tool for fetching news articles from Hacker News
@tool
def fetch_news(topic: str = "") -> List[Dict]:
    """Fetch news articles from Hacker News."""
    # First, get the top stories from HN
    top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    response = requests.get(top_stories_url)
    story_ids = response.json()[:10]  # Get top 10 stories
    
    articles = []
    for story_id in story_ids:
        # Get story details
        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        story_response = requests.get(story_url)
        story = story_response.json()
        
        if not story:
            continue
            
        # Get article content if URL exists
        content = ""
        if "url" in story:
            try:
                article_response = requests.get(story["url"], timeout=5)
                if article_response.status_code == 200:
                    soup = BeautifulSoup(article_response.text, 'html.parser')
                    # Get text content from p tags
                    paragraphs = soup.find_all('p')
                    content = " ".join([p.get_text() for p in paragraphs[:3]])  # First 3 paragraphs
            except:
                content = story.get("text", "")  # Fallback to story text if available
        
        articles.append({
            "title": story.get("title", ""),
            "url": story.get("url", ""),
            "content": content or story.get("text", ""),
            "score": story.get("score", 0),
            "by": story.get("by", ""),
            "time": story.get("time", "")
        })
        
        # Small delay to be nice to the API
        time.sleep(0.1)
    
    # If a topic is provided, filter the articles
    if topic:
        topic = topic.lower()
        articles = [
            article for article in articles 
            if topic in article["title"].lower() or topic in article["content"].lower()
        ]
    
    return articles

# Tool for summarizing text
@tool
def summarize_article(text: str) -> str:
    """Summarize the given article text."""

    summary = llm.invoke(f"Please summarize the following text concisely: {text}")
    return summary.content

# Define the state that will be passed between nodes
class AgentState(TypedDict):
    messages: List[BaseMessage]
    topic: str
    articles: List[Dict]
    summaries: List[str]

# Node function for fetching articles
def fetch_articles(state: AgentState) -> AgentState:
    topic = state["topic"]
    articles = fetch_news.invoke(topic)
    state["articles"] = articles
    return state

# Node function for summarizing articles
def create_summaries(state: AgentState) -> AgentState:
    summaries = []
    for article in state["articles"]:
        summary = summarize_article(article["content"])
        summaries.append(summary)
    state["summaries"] = summaries
    return state

# Node function for generating final response
def generate_response(state: AgentState) -> AgentState:
    summaries_text = "\n".join(state["summaries"])
    response = llm.invoke(
        f"Based on the following article summaries about {state['topic']}, "
        f"provide a comprehensive overview:\n{summaries_text}"
    )
    state["messages"].append(response)
    return state

# Create the workflow
def create_workflow() -> StateGraph:
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("fetch", fetch_articles)
    workflow.add_node("summarize", create_summaries)
    workflow.add_node("respond", generate_response)

    # Define the edges
    workflow.add_edge("fetch", "summarize")
    workflow.add_edge("summarize", "respond")

    # Set the entry point
    workflow.set_entry_point("fetch")

    # Set the exit point
    workflow.set_finish_point("respond")

    return workflow.compile()

def process_news_query(topic: str = "") -> AgentState:
    """Process a news query for Hacker News articles."""
    workflow = create_workflow()
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content="Tell me about the latest news from Hacker News" + (f" related to {topic}" if topic else ""))],
        "topic": topic,
        "articles": [],
        "summaries": []
    }
    
    # Execute the workflow
    final_state = workflow.invoke(initial_state)
    return final_state

# Example usage
if __name__ == "__main__":
    result = process_news_query()  # Get all top stories
    print("Final Response:", result["messages"][-1].content)