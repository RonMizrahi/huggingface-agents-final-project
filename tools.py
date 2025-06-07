from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from huggingface_hub import list_models
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage
import base64
import requests
import os
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()
search_tool = DuckDuckGoSearchRun()  # Instantiate once

# Get API key from environment variable
openai_api_key = os.getenv("gpt")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


@tool
def web_search(query: str) -> str:
    """Performs a web search to find additional information."""
    results = search_tool.invoke(query)
    if results:
        return results  # DuckDuckGoSearchRun returns a string directly
    else:
        return "No relevant web search results found."

@tool
def web_search_with_images(query: str, max_results: int = 5, include_images: bool = True) -> str:
    """Performs a web search that can return both text results and images from DuckDuckGo.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 3)
        include_images: Whether to include image results (default: True)
    
    Returns:
        A formatted string containing text results and optionally image URLs
    """
    try:
        with DDGS() as ddgs:
            results = {
                "text_results": [],
                "image_results": []
            }
            
            # Get text search results
            text_results = list(ddgs.text(query, max_results=max_results))
            for result in text_results:
                results["text_results"].append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "href": result.get("href", "")
                })
            
            # Get image search results if requested
            if include_images:
                image_results = list(ddgs.images(query, max_results=max_results))
                for img in image_results:
                    results["image_results"].append({
                        "title": img.get("title", ""),
                        "image_url": img.get("image", ""),
                        "thumbnail": img.get("thumbnail", ""),
                        "source": img.get("source", "")
                    })
            
            # Format the results
            formatted_output = []
            
            if results["text_results"]:
                formatted_output.append("=== TEXT SEARCH RESULTS ===")
                for i, result in enumerate(results["text_results"], 1):
                    formatted_output.append(f"{i}. {result['title']}")
                    formatted_output.append(f"   {result['body']}")
                    formatted_output.append(f"   URL: {result['href']}")
                    formatted_output.append("")
            
            if results["image_results"]:
                formatted_output.append("=== IMAGE SEARCH RESULTS ===")
                for i, img in enumerate(results["image_results"], 1):
                    formatted_output.append(f"{i}. {img['title']}")
                    formatted_output.append(f"   Image URL: {img['image_url']}")
                    formatted_output.append(f"   Thumbnail: {img['thumbnail']}")
                    formatted_output.append(f"   Source: {img['source']}")
                    formatted_output.append("")
            
            return "\n".join(formatted_output) if formatted_output else "No search results found."
            
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def image_search(query: str, max_results: int = 10) -> str:
    """Performs an image search using DuckDuckGo to find relevant images.
    
    Args:
        query: The search query for images
        max_results: Maximum number of image results to return (default: 3)
    
    Returns:
        A formatted string containing image URLs and metadata
    """
    try:
        with DDGS() as ddgs:
            image_results = list(ddgs.images(query, max_results=max_results))
            
            if not image_results:
                return "No image results found."
            
            formatted_output = [f"=== IMAGE SEARCH RESULTS FOR '{query}' ==="]
            
            for i, img in enumerate(image_results, 1):
                formatted_output.append(f"{i}. {img.get('title', 'Untitled')}")
                formatted_output.append(f"   Image URL: {img.get('image', 'N/A')}")
                formatted_output.append(f"   Thumbnail: {img.get('thumbnail', 'N/A')}")
                formatted_output.append(f"   Source: {img.get('source', 'N/A')}")
                formatted_output.append(f"   Width: {img.get('width', 'N/A')} x Height: {img.get('height', 'N/A')}")
                formatted_output.append("")
            
            return "\n".join(formatted_output)
            
    except Exception as e:
        return f"Image search failed: {str(e)}"



vision_llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini", api_key=openai_api_key, verbose=True)

def explain_image(img_path_or_url: str) -> str:
    """
    Explain what you understand from the image file or URL using a multimodal model.
    
    Args:
        img_path_or_url: Either a local file path or a URL to an image
    
    Returns:
        Explanation of the image content
    """
    all_text = ""
    try:
        # Determine if it's a URL or local file path
        if img_path_or_url.startswith(('http://', 'https://')):
            # Handle URL
            response = requests.get(img_path_or_url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
        else:
            # Handle local file path
            with open(img_path_or_url, "rb") as image_file:
                image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare the prompt including the base64 image data
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Explain what you see in the image. "
                            "Return only a short answer, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # Call the vision-capable model
        response = vision_llm.invoke(message)

        return response.content
    except Exception as e:
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""

# Initialize Wikipedia API wrapper
wikipedia = WikipediaAPIWrapper()

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information about a topic.
    
    Args:
        query: The search query for Wikipedia
    
    Returns:
        Wikipedia article content or search results
    """
    try:
        # Use the WikipediaQueryRun tool with the wrapper
        wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
        result = wiki_tool.run(query)
        
        if result:
            return result
        else:
            return f"No Wikipedia results found for '{query}'"
            
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"