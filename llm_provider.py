"""
CourseMate — LLM provider factory.

Switch between a fully local Ollama model and the Groq cloud API by
setting LLM_PROVIDER=ollama|groq (see config.py / .env.example).
Keeping this in one place means app.py never needs to know which
backend is active.
"""

import config


class ConfigError(RuntimeError):
    pass


def get_llm():
    """Return a LangChain-compatible chat/LLM object for the configured provider."""

    if config.LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )

    if config.LLM_PROVIDER == "groq":
        if not config.GROQ_API_KEY:
            raise ConfigError(
                "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is missing. "
                "Set it in your .env file or environment variables. "
                "Get a free key at https://console.groq.com/keys"
            )
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=config.LLM_TEMPERATURE,
        )

    raise ConfigError(
        f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use 'ollama' or 'groq'."
    )
