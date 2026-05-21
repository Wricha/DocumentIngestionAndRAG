from googleapiclient.discovery import build

@mcp.tool
def web_search(query: str, max_results: int = 3) -> str:
    """Perform a real-time web search via Google."""
    try:
        service = build("customsearch", "v1", developerKey=settings.google_api_key)
        res = service.cse().list(
            q=query, cx=settings.google_cse_id, num=max_results
        ).execute()

        lines = []
        for i, item in enumerate(res.get("items", [])):
            lines.append(
                f"[Web Result {i+1}]\n"
                f"Title: {item['title']}\n"
                f"Link: {item['link']}\n"
                f"Snippet: {item.get('snippet', '')}\n"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"Error executing web search: {str(e)}"