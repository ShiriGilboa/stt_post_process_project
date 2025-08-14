import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from openai import OpenAI

from extractors import BaseResponseExtractor, JsonResponseExtractor

class BaseLLMInvoker(ABC):
    """
    Abstract base class for invoking an LLM.
    Encapsulates client setup and raw prompt-to-response logic.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", tools: Optional[List[Dict[str, Any]]] = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tools = tools or []

    @abstractmethod
    def invoke(self, input_text: str, extra_info: Optional[str] = None) -> str:
        """
        Runs the full invocation workflow and returns the final output string.
        """
        pass

    def _call_llm(self, instructions: str, prompt: str) -> str:
        """
        Low-level call to the OpenAI client.
        """
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": prompt,
        }
        if self.tools:
            payload["tools"] = self.tools
        response = self.client.responses.create(**payload)
        return response.output_text.strip()


class LLMInvoker(BaseLLMInvoker):
    """
    Simple invoker: builds a prompt and returns the LLM's raw output.
    """

    def __init__(
        self,
        api_key: str,
        instructions: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        few_shots: Optional[List[str]] = None,
        model: str = "gpt-4o"
    ):
        super().__init__(api_key=api_key, model=model, tools=tools)
        self.instructions = instructions
        self.few_shots = few_shots or []

    def invoke(self, input_text: str, extra_info: Optional[str] = None) -> str:
        parts: List[str] = []
        if extra_info:
            parts.append(f"Additional Info:\n{extra_info}")
        parts.append(f"Input:\n{input_text}")
        prompt = "\n\n".join(parts)

        # prepend few-shot examples if any
        if self.few_shots:
            shots = "\n\n".join(self.few_shots)
            prompt = shots + "\n\n" + prompt

        # call model
        return self._call_llm(self.instructions, prompt)


# Agent interface
class Agent(ABC):
    """
    Abstract base class for pipeline agents.
    """
    def __init__(self, invoker: BaseLLMInvoker):
        self.invoker = invoker

    @abstractmethod
    async def run(self, transcript: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the transcript (and optional context) and return structured data.
        """
        pass

class TranscriptGenerateExtractAgent(Agent):
    def __init__(
        self,
        invoker: BaseLLMInvoker,
        agent_name: str,
        split_comma_to_array: bool = False,
        extract_entities_from_context: Optional[List[str]] = None,
        verbose: bool = False,
        response_extractor: Optional[BaseResponseExtractor] = None,
        return_response_with_wrapp: bool = True,
        max_retires: int = 2
    ):
        super().__init__(invoker)
        self.agent_name = agent_name
        self.split_comma_to_array = split_comma_to_array
        self.entities_context = extract_entities_from_context or []
        self.verbose = verbose
        self.response_extractor = response_extractor
        self.return_response_with_wrapp = return_response_with_wrapp
        self.max_retires = max_retires

    async def run(self, transcript: str, context: Dict[str, Any]):
        for dx in range(self.max_retires):
            try:
                # Build prompt
                prompt_lines = [f"transcript: {transcript}\n"]
                for entity in self.entities_context:
                    prompt_lines.append(f"{entity}: {context.get(entity, '')}\n")
                prompt = "\n".join(prompt_lines)

                # Invoke LLM
                raw_output = self.invoker.invoke(input_text=prompt)

                # Verbose logging
                if self.verbose:
                    print(f"Prompt sent to LLM:\n{prompt}\n")
                    print(f"Raw LLM response:\n{raw_output}\n")

                # Extract structured data
                if self.response_extractor:
                    result = self.response_extractor.extract(raw_output)
                elif self.split_comma_to_array:
                    result = [item.strip() for item in raw_output.split(",") if item.strip()]
                else:
                    result = raw_output

                return {self.agent_name: result} if self.return_response_with_wrapp else result
            except Exception as ex:
                print(f"Error in Agent: {self.agent_name} try number: {dx}", ex)
    
class DeciderAgent(TranscriptGenerateExtractAgent):
    KEY_ANSWER = "Answer"
    DEFUALT_VALUE = "No"
    YES = "YES"

    def __init__(
            self,
            invoker: BaseLLMInvoker,
            agent_name: str,
            split_comma_to_array: bool = False,
            extract_entities_from_context: Optional[List[str]] = None,
            verbose: bool = False,
            response_extractor: Optional[BaseResponseExtractor] = JsonResponseExtractor(key=KEY_ANSWER, def_val=DEFUALT_VALUE),
            return_response_with_wrapp: bool = True
        ):
            super().__init__(invoker, agent_name, split_comma_to_array, extract_entities_from_context, verbose, response_extractor, return_response_with_wrapp)

    async def run_decision(self, transcript: str, context: Dict[str, Any]):
        result = await self.run(transcript=transcript, context=context)
        return str(result[self.agent_name]()).capitalize() == self.YES

    
class SentenceBuilderAgent(Agent):
    def __init__(self, invoker: BaseLLMInvoker, agent_name: str):
        super().__init__(invoker)
        self.agent_name = agent_name

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        output = self.invoker.invoke(input_text=context)
        return output
