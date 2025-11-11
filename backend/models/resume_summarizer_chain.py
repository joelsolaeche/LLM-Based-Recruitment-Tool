from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from backend.config import settings
from backend.llm_factory import get_llm

# Create a string template for this chain
template = """You are an expert resume analyzer. Please summarize the following resume and extract the candidate's key skills, experience, and qualifications.

Resume:
{resume}

Please provide a concise summary focusing on the candidate's technical skills, years of experience, and key qualifications."""


def get_resume_summarizer_chain():
    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["resume"],
        template=template,
    )

    # Create an instance of an LLM
    llm = get_llm(
        temperature=0,
        provider=settings.LLM_PROVIDER,
    )

    # Create an instance of LLMChain
    resume_summarizer_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )

    return resume_summarizer_chain


if __name__ == "__main__":
    resume_summarizer_chain = get_resume_summarizer_chain()
    print(
        resume_summarizer_chain.invoke(
            {"resume": "I am a software engineer with 5 years of experience"}
        )
    )