"""
data_synthesizer.py
───────────────────
Intermediate data aggregation layer to process raw Coral GitHub data
into high-signal metrics context for the copywriting LLM.
"""

import re
from typing import Any, Dict, List, Set

def _get_empty_payload() -> Dict[str, Any]:
    """Return a structured empty payload for graceful fallback."""
    return {
        "summary_metrics": {
            "total_count": 0,
            "unique_repos": [],
            "computed_impact_score": "Low (0 data points)",
        },
        "categorized_activity": {
            "features": [],
            "bug_fixes": [],
            "general_modifications": [],
        },
        "extracted_tokens": [],
    }

def synthesize_github_data(raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process a raw list of dictionaries from Coral into an engineered, 
    high-signal metrics context payload.
    """
    if not raw_data:
        return _get_empty_payload()

    # Heuristic detection for dataset types
    is_commits = any("sha" in item and "message" in item for item in raw_data)
    is_issues = any("state" in item and "title" in item for item in raw_data)
    
    unique_repos: Set[str] = set()
    total_count = len(raw_data)
    
    features: List[str] = []
    bug_fixes: List[str] = []
    general: List[str] = []
    readmes: List[str] = []
    extracted_tokens: Set[str] = set()
    
    if is_commits:
        unique_authors: Set[str] = set()
        seen_messages: Set[str] = set()
        
        for item in raw_data:
            repo = item.get("repo") or item.get("repo__name") or item.get("repository")
            if repo:
                unique_repos.add(str(repo))
            
            author = item.get("author_name") or item.get("author")
            if author:
                unique_authors.add(str(author))
                
            msg = str(item.get("message", "")).strip()
            if not msg or msg in seen_messages:
                continue
            seen_messages.add(msg)
            
            # Extract technical tokens (alphanumeric words > 4 chars)
            tokens = re.findall(r'\b[a-zA-Z0-9]{5,}\b', msg.lower())
            extracted_tokens.update(tokens)
            
            # Categorize by active feature keywords
            msg_lower = msg.lower()
            if msg_lower.startswith("feat:") or "feature" in msg_lower:
                features.append(msg)
            elif msg_lower.startswith("fix:") or "bug" in msg_lower or "fix " in msg_lower:
                bug_fixes.append(msg)
            else:
                general.append(msg)
                
    elif is_issues:
        open_count = 0
        closed_count = 0
        
        # Sort by date (descending) to identify priority items based on recency
        def _get_date(x: Dict[str, Any]) -> str:
            return str(x.get("created_at") or x.get("updated_at") or "")
        
        sorted_items = sorted(raw_data, key=_get_date, reverse=True)
        
        for item in sorted_items:
            repo = item.get("repo") or item.get("repo__name") or item.get("repository")
            if repo:
                unique_repos.add(str(repo))
                
            state = str(item.get("state", "")).lower()
            if state == "open":
                open_count += 1
            elif state in ("closed", "merged"):
                closed_count += 1
                
            title = str(item.get("title", "")).strip()
            if title:
                # Extract tokens
                tokens = re.findall(r'\b[a-zA-Z0-9]{5,}\b', title.lower())
                extracted_tokens.update(tokens)
                
                title_lower = title.lower()
                if "feat" in title_lower or "add" in title_lower:
                    features.append(title)
                elif "fix" in title_lower or "bug" in title_lower:
                    bug_fixes.append(title)
                else:
                    general.append(title)

    else:
        # Generic fallback for other Coral tables (e.g. github.events)
        for item in raw_data:
            repo = item.get("repo") or item.get("repo__name") or item.get("name")
            if repo:
                unique_repos.add(str(repo))
            
            readme = item.get("readme_content")
            if readme:
                readmes.append(f"README for {repo}:\n{readme[:15000]}")

            for key, val in item.items():
                if key == "readme_content":
                    continue
                if isinstance(val, str):
                    tokens = re.findall(r'\b[a-zA-Z0-9]{5,}\b', val.lower())
                    extracted_tokens.update(tokens)
            general.append(str({k: v for k, v in item.items() if k != "readme_content"}))

    # Compute a heuristic impact score based on data density
    if total_count > 20 or len(features) > 3:
        impact = "High Impact"
    elif total_count > 5 or len(features) > 0:
        impact = "Medium Impact"
    else:
        impact = "Standard Activity"
        
    impact_score = f"{impact} (Volume: {total_count} records, Features shipped: {len(features)})"

    # Assemble final deeply nested schema
    return {
        "summary_metrics": {
            "total_count": total_count,
            "unique_repos": sorted(list(unique_repos)),
            "computed_impact_score": impact_score,
        },
        "categorized_activity": {
            "features": features,
            "bug_fixes": bug_fixes,
            "readmes": readmes,
            "general_modifications": general,
        },
        "extracted_tokens": sorted(list(extracted_tokens)),
    }
