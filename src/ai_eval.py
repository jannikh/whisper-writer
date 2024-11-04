import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# AUTO_COMPLETE_PROMPT = """
# system: You are an auto-completion expert, please finish the sentence fragments the user provides. If uncertain, provide shorter but accurate completions (except if an accurate list should be completed). Respond with the full text, repeating the user's text. Do NOT write any additional text outside of the completion prediction.
# user: "We finish each others"
# assistant: "We finish each others' sentences."
# user: "Paris is the capi"
# assistant: "Paris is the capital of France."
# user: "ChatGPT is "
# assistant: "ChatGPT is an advanced language model developed by OpenAI."
# user: "The following countries in Latin america do not speak Spanish:"
# assistant: "The following countries in Latin America do not speak Spanish: Belize, Brazil, French Guiana, Guyana, Haiti and Suriname."
# user: \""""

EVAL_PROMPT = """
You are a fast-responding AI assistant, please answer the requests the user provides. If uncertain, provide shorter but accurate responses (except if an accurate list should be completed). Respond just with the exact, concise response, nothing else. Do NOT write any additional text outside of the answer.

Examples of great responses:

	Instruction: Countries in Latin America that don't speak Spanish
	Response: Belize, Brazil, French Guiana, Guyana, Haiti and Suriname

	Instruction: console command to reduce the volume of music.mp3 by half
	Response: ffmpeg -i music.mp3 -af 'volume=0.5' music-quieter.mp3

	Instruction: How many calories are in a banana?
	Response: A medium banana has about 105 calories

	Instruction: Translate Hello to Mandarin Chinese
	Response: 你好

	Instruction: What are some effective study techniques for improving memory?
	Response: Use spaced repetition, active recall, and mnemonics

	Instruction: 8 plus 31 minus 15 times 5
	Response: -6

	Instruction: python 8 + 31 - 15 x 5
	Response: py -c "print(8+31-15*5)"

	Instruction: Find directions to the Eiffel Tower
	Response: https://www.google.com/maps/dir//Eiffel+Tower,+Champ+de+Mars,+5+Avenue+Anatole+France,+75007+Paris,+France/

	Instruction: Wikipedia link for Albert Einstein.
	Response: https://en.wikipedia.org/wiki/Albert_Einstein

	Instruction: Book a hotel room in New York City
	Response: https://www.booking.com/city/us/new-york.en-gb.html

	Instruction: What's the weather forecast for London?
	Response: https://weather.com/weather/today/l/51.5074,-0.1278

---- End of examples ----

Please respond to the following instruction:
Instruction: {instructions}
Response: """

EVAL_CONTEXT_PROMPT = """
You are a diligent, professional and highly reliable AI assistant with decades of experience, please answer the requests the user provides and execute the instructions perfectly, given the context that is provided to you. Respond just with the exact, concise response, nothing else. Do NOT write any explanatory text outside of the answer, but just directly begin with the desired answer and end when the answer is complete.

Your job is the provide a professional, accurate and complete response by executing the instructions in the context of the following text:
```
{context}
```

Your instructions are:
{instructions}

Your response:
"""

# LLM Settings
TEMPERATURE = .75
LLM_MODEL = "gpt-4o-mini"
llm = ChatOpenAI(temperature=TEMPERATURE, model=LLM_MODEL)

TEMPERATURE_ADVANCED = .75
LLM_MODEL_ADVANCED = "gpt-o1-mini"
llm_advanced = ChatOpenAI(temperature=TEMPERATURE_ADVANCED, model=LLM_MODEL_ADVANCED)

# INITIALIZATIONS
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'whisper-writer ai eval'

def evaluate(instructions, context=None, advanced=False):
	"""
	this function takes the instructions given by the user and evaluates them  using the EVAL_PROMPT
	"""

	print(f"Evaluating {instructions} with{' context ' + context if context else ''} using {LLM_MODEL_ADVANCED if advanced else LLM_MODEL}")

	prompt_args = {
		'instructions': instructions,
		'context': context,
	}

	if not context:
		prompt = ChatPromptTemplate.from_template(EVAL_PROMPT)
	else:
		prompt = ChatPromptTemplate.from_template(EVAL_CONTEXT_PROMPT)
	
	if advanced:
		eval_chain = (
			prompt
			| llm_advanced
			| StrOutputParser()
		)
	else:
		eval_chain = (
			prompt
			| llm
			| StrOutputParser()
		)

	langchain_tags = ["evaluate"]
	if context:
		langchain_tags.append("context")

	result = eval_chain.invoke(prompt_args, {'tags': langchain_tags})

	print(f"Result: {result}")
	
	return(result)
