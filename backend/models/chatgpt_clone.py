from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from backend.config import settings
from backend.llm_factory import get_llm


class ChatAssistant:
    def __init__(self, llm_model, api_key, temperature=0, history_length=3):
        """
        Initialize the ChatAssistant class.

        Parameters
        ----------
        llm_model : str
            The model name.

        api_key : str
            The API key for accessing the LLM model.

        temperature : float
            The temperature parameter for generating responses.

        history_length : int, optional
            The length of the conversation history to be stored in memory. Default is 3.
        """
        # TODO: Create a string template for the chat assistant. It must indicate the LLM
        # that a chat history is being provided and that a new question is being asked.
        # The template must have two input variables: `history` and `human_input`.
        template = """The following is a friendly conversation between a human and an AI assistant.
The AI is helpful and provides detailed answers based on the context.

Chat History:
{history}
Human: {human_input}

AI Assistant:"""
        

        # Create a prompt template using the string template created above.
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["history", "human_input"]
        )

        # Create an instance of an LLM using the `get_llm` factory function with the appropriate settings.
        self.llm = get_llm(
            model=llm_model,
            api_key=api_key,
            temperature=temperature,
            provider=settings.LLM_PROVIDER
        )

        # Create ConversationBufferWindowMemory instance with k=history_length
        memory = ConversationBufferWindowMemory(k=history_length)
        
        # Create LLMChain instance combining prompt, llm, and memory
        self.model = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=memory,
            verbose=settings.LANGCHAIN_VERBOSE
        )

    def predict(self, human_input: str) -> str:
        """
        Generate a response to a human input.

        Parameters
        ----------
        human_input : str
            The human input to the chat assistant.

        Returns
        -------
        response : str
            The response from the chat assistant.
        """
        response = self.model.invoke({"human_input": human_input})

        return response["text"]


if __name__ == "__main__":
    # Determine which model and API key to use based on provider
    llm_model = settings.OPENAI_LLM_MODEL if settings.LLM_PROVIDER == "openai" else settings.GEMINI_LLM_MODEL
    api_key = settings.OPENAI_API_KEY if settings.LLM_PROVIDER == "openai" else settings.GOOGLE_API_KEY
    
    # Create an instance of ChatAssistant with appropriate settings
    chat_assistant = ChatAssistant(
        llm_model=llm_model,
        api_key=api_key,
        temperature=0,
        history_length=2,
    )

    # Use the instance to generate a response
    output = chat_assistant.predict(
        human_input="what is the answer to life the universe and everything?"
    )

    print(output)