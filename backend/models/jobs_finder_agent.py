from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from backend.config import settings
from backend.models.jobs_finder import JobsFinderAssistant
from backend.llm_factory import get_llm


def build_job_finder(job_finder_assistant):
    def job_finder(human_input: str):
        return job_finder_assistant.predict(human_input)

    return job_finder


def build_cover_letter_writing(llm, resume):
    def cover_letter_writing(job_description: str):
        # Create a string template for this chain
        template = """You are an expert cover letter writer. You will be provided with a resume and a job description. 
Please write a professional and compelling cover letter that highlights how the applicant's skills and experience match the job requirements.

Resume:
{resume}

Job Description:
{job_description}

Please write a well-structured cover letter that emphasizes the candidate's relevant qualifications and enthusiasm for the position."""

        # Create a prompt template
        prompt = PromptTemplate(
            input_variables=["resume", "job_description"],
            template=template,
        )

        # Create an instance of LLMChain
        cover_letter_writing_chain = LLMChain(
            llm=llm,
            prompt=prompt,
        )

        return cover_letter_writing_chain.invoke(
            {"resume": resume, "job_description": job_description}
        )["text"]

    return cover_letter_writing


class JobsFinderAgent:
    def __init__(
        self, resume, llm_model, api_key, temperature=0, history_length=3
    ):
        """
        Initialize the JobsFinderAgent class.

        Parameters
        ----------
        resume : str
            The resume of the user.

        llm_model : str
            The model name.

        api_key : str
            The API key for accessing the LLM model.

        temperature : float
            The temperature parameter for generating responses.
        """

        self.resume = resume

        # Create an instance of an LLM
        self.llm = get_llm(
            model=llm_model,
            api_key=api_key,
            temperature=temperature,
            provider=settings.LLM_PROVIDER,
        )

        # Create the Job finder tool
        self.job_finder = JobsFinderAssistant(
            resume=resume,
            llm_model=llm_model,
            api_key=api_key,
            temperature=temperature,
        )

        self.agent_executor = self.create_agent()
        self.agent_memory = []
        self.history_length = history_length

    def create_agent(self):
        job_finder = build_job_finder(self.job_finder)
        cover_letter_writing = build_cover_letter_writing(
            self.llm, self.resume
        )
        tools = [
            Tool(
                name="jobs_finder",
                func=job_finder,
                description="Look up for jobs based on user preferences.",
                handle_tool_error=True,
            ),
            Tool(
                name="cover_letter_writing",
                func=cover_letter_writing,
                description="Write a cover letter based on a job description, extract as much information you can about the job from the user input and from the chat history.",
                handle_tool_error=True,
            ),
        ]

        # Use different agent types based on provider
        if settings.LLM_PROVIDER == "openai":
            prompt = hub.pull("hwchase17/openai-functions-agent")
            print(f"Prompt pulled from hub: {prompt}")
            from langchain.agents import create_openai_functions_agent
            agent = create_openai_functions_agent(self.llm, tools, prompt)
        else:
            # Use ReAct agent for Gemini and other providers with custom prompt
            from langchain.prompts import PromptTemplate as AgentPromptTemplate
            from langchain.agents import create_react_agent
            
            template = """You are a job search assistant helping a user find jobs and write cover letters. You have already received the user's resume.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important notes:
- The user has already uploaded their resume, so you have access to it through the tools
- Use jobs_finder to search for jobs matching the user's preferences
- Use cover_letter_writing to write cover letters for specific job descriptions
- When writing cover letters, extract the job description from the conversation or ask the user for it

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
            
            prompt = AgentPromptTemplate(
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
                template=template
            )
            print(f"Using custom ReAct prompt for {settings.LLM_PROVIDER}")
            agent = create_react_agent(self.llm, tools, prompt)

        # Create an agent executor by passing in the agent and tools
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            early_stopping_method="force",
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    def predict(self, human_input: str) -> str:
        agent_reseponse = self.agent_executor.invoke(
            {"input": human_input, "chat_memory": self.agent_memory}
        )

        self.agent_memory.extend(
            [
                HumanMessage(content=human_input),
                AIMessage(content=agent_reseponse["output"]),
            ]
        )

        self.agent_memory = self.agent_memory[-self.history_length :]

        return agent_reseponse