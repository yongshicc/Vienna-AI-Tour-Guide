def format_prompt(persona: dict, landmark: dict, user_query: str) -> str:
    return (
        f"{persona['pre_prompt']}\n"
        f"Traits: {persona['traits']}\n"
        f"You are at {landmark['name']}: {landmark['description']}\n"
        f"User: {user_query}\n"
        f"Bot:"
    )
