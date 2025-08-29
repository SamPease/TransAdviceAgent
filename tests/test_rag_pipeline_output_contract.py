from app.rag_pipeline import output_node
import asyncio


def test_output_node_builds_sources_order_and_dedup():
    # Construct fake state with metadata_map containing duplicate and empty urls
    state = {
        "final_answer": "some summary",
        "doc_ids": ["a", "b", "c", "d"],
        "metadata_map": {
            "a": {"url": "https://example.com/page1"},
            "b": {"url": "https://example.com/page2"},
            "c": {"url": "https://example.com/page1/"},  # duplicate of a (trailing slash)
            "d": {"permalink": "/r/test/comments/xyz"},
        }
    }

    res = asyncio.run(output_node(state))

    assert "final_answer" in res
    assert "sources" in res
    urls = [s.get("url") for s in res["sources"]]

    # a and c should dedupe to a single https://example.com/page1 (first-seen kept)
    assert urls[0] == "https://example.com/page1"
    assert "https://example.com/page2" in urls
    # permalink should be normalized to full reddit url
    assert any("reddit.com" in u for u in urls)
    # ensure duplicates removed
    assert len(urls) == len(set([u.rstrip('/') for u in urls if u]))
