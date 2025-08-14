from abc import ABC
import asyncio
from typing import Any, Dict
from llm_agent_classes import LLMInvoker
from enums.agent_name import AgentNames
from extractors import JsonResponseExtractor
from instructions.agents_instructions import BEST_CANDIDATES_AGENT_INSTRUCTIONS, BUILD_SENTENCE_FROM_PARTS, JARGON_AGENT_INSTRUCTIONS, JARGON_DECIDER_AGENT_INSTRUCTIONS, NER_AGENT_INSTRUCTIONS, NER_DECIDER_AGENT_INSTRUCTIONS, TOPIC_INSTRUCTIONS, FIX_STT_OUTPUT_AGENT_INSTRUCTIONS
from llm_agent_classes import DeciderAgent, SentenceBuilderAgent, TranscriptGenerateExtractAgent

class BasePipeline(ABC):
    """
    Base class for pipelines that orchestrate LLM invokers and agents.
    """
    def __init__(self, api_key: str, pipline_name: str = "BasePipeline", num_iterations_allowed: int = 1, verbose: bool = False):
        self.api_key = api_key
        self.pipline_name = pipline_name
        self.num_iterations_allowed = num_iterations_allowed
        self.verbose = verbose

    def get_num_iterations_allowed(self) -> int:
        """
        Returns the number of iterations allowed for this pipeline.
        """
        return self.num_iterations_allowed

    def get_pipeline_name(self) -> str:
        """
        Returns the name of the pipeline.
        """
        return self.pipline_name

    async def process(self, transcript: str) -> tuple[bool, str]:
        """
        Process the transcript and return a tuple indicating if generation was successful and the result.
        
        Args:
            transcript: The input text to process.

        Returns:
            A tuple (is_generated: bool, result: str).
        """
        raise NotImplementedError("Subclasses should implement this method.")

class GenerateWhisperPromptPipeline(BasePipeline):
    def __init__(
        self,
        api_key: str,
        pipline_name: str = "GenerateWhisperPromptPipeline",
        num_iterations_allowed: int = 1,
        verbose: bool = False):

        super().__init__(api_key, pipline_name, num_iterations_allowed, verbose)

        # Instantiate invokers with tailored instructions
        self.topic_agent = TranscriptGenerateExtractAgent(
            invoker=LLMInvoker(
                api_key,
                instructions=TOPIC_INSTRUCTIONS
            ),
            verbose=verbose,
            agent_name=AgentNames.TOPIC.value
        )
        self.ner_agent = TranscriptGenerateExtractAgent(
            LLMInvoker(
                api_key,
                instructions=NER_AGENT_INSTRUCTIONS,
            ),
            split_comma_to_array=True,
            extract_entities_from_context=[AgentNames.TOPIC.value],
            verbose=verbose,
            agent_name=AgentNames.NER_NAMES.value
        )
        self.jargon_agent = TranscriptGenerateExtractAgent(
            LLMInvoker(
                api_key,
                instructions=JARGON_AGENT_INSTRUCTIONS,
            ),
            split_comma_to_array=True,
            extract_entities_from_context=[AgentNames.TOPIC.value],
            verbose=verbose,
            agent_name=AgentNames.JRAGON_LIST.value
        )
        self.decider_ner = DeciderAgent(
            LLMInvoker(
                api_key,
                instructions=NER_DECIDER_AGENT_INSTRUCTIONS                
            ),
            extract_entities_from_context=[AgentNames.TOPIC.value, AgentNames.NER_NAMES.value],
            verbose=verbose,
            agent_name=AgentNames.NER_DECIDER.value
        )
        self.decider_jargon = DeciderAgent(
            LLMInvoker(
                api_key,
                instructions=JARGON_DECIDER_AGENT_INSTRUCTIONS,
            ),
            extract_entities_from_context=[AgentNames.TOPIC.value, AgentNames.JRAGON_LIST.value],
            verbose=verbose,
            agent_name=AgentNames.JARGON_DEIDER.value
        )
        self.choose_relevant_names = TranscriptGenerateExtractAgent(
            LLMInvoker(
                api_key,
                instructions=BEST_CANDIDATES_AGENT_INSTRUCTIONS,
            ),
            extract_entities_from_context=[AgentNames.TOPIC.value, AgentNames.NER_NAMES.value],
            verbose=verbose,
            agent_name=AgentNames.MOST_RELEVANT_NAMES.value,
            response_extractor=JsonResponseExtractor(key=AgentNames.NER_NAMES.value, def_val=""),
            return_response_with_wrapp=False
        )
        self.sentence_builder = SentenceBuilderAgent(
            LLMInvoker(
                api_key,
                instructions=BUILD_SENTENCE_FROM_PARTS,
            ),
            agent_name=AgentNames.SENTENCE_BUILDER.value
        )

    async def process(self, transcript: str) -> tuple[bool, str, str]:
        ret_val: str = ""
        is_generated_initial_prompt: bool = False
        ctx: Dict[str, Any] = {}

        # 1. Topic
        topic_res = await self.topic_agent.run(transcript, ctx)
        ctx.update(topic_res)
        # 2. NER & Jargon in parallel
        ner_res, jargon_res = await asyncio.gather(
            self.ner_agent.run(transcript, ctx),
            self.jargon_agent.run(transcript, ctx)
        )
        ctx.update(ner_res)
        ctx.update(jargon_res)

        # 3. Decide if to add Names and Jargon terms
        ner_decider_decision, jargon_decider_decision = await asyncio.gather(
            self.decider_ner.run(transcript, ctx),
            self.decider_jargon.run(transcript, ctx)
        )
          
        if ner_decider_decision:
            best_names = await self.choose_relevant_names.run(transcript, ctx)
            print(f"Best Names: {best_names}")
            sent_ctx: Dict[str, Any] = {}
            sent_ctx[AgentNames.TOPIC] = ctx.get(AgentNames.TOPIC, "")
            sent_ctx["names_list"] = best_names

            if jargon_decider_decision:
                sent_ctx[AgentNames.JRAGON_LIST] = ctx.get(AgentNames.JRAGON_LIST, "")

            res = await self.sentence_builder.run(sent_ctx)
            print(f"sentence build: {res}")
            ret_val = res
            is_generated_initial_prompt = True

        return is_generated_initial_prompt, ret_val
    
    
    
class GenerateNamesPipeline(BasePipeline):
    def __init__(
        self,
        api_key: str,
        pipline_name: str = "GenerateNamesPipeline",
        num_iterations_allowed: int = 1,
        verbose: bool = False):
        
        super().__init__(api_key, pipline_name, num_iterations_allowed, verbose)
        
        # Instantiate invokers with tailored instructions
        self.topic_agent = TranscriptGenerateExtractAgent(
            invoker=LLMInvoker(
                api_key,
                instructions=TOPIC_INSTRUCTIONS
            ),
            verbose=verbose,
            agent_name=AgentNames.TOPIC.value
        )
        self.ner_agent = TranscriptGenerateExtractAgent(
            LLMInvoker(
                api_key,
                instructions=NER_AGENT_INSTRUCTIONS,
            ),
            split_comma_to_array=True,
            extract_entities_from_context=[AgentNames.TOPIC.value],
            verbose=verbose,
            agent_name=AgentNames.NER_NAMES.value,
            return_response_with_wrapp=False
        )

    async def process(self, transcript: str) -> tuple[bool, str]:
        ret_val: str = ""
        ctx: Dict[str, Any] = {}
        
        topic_res = await self.topic_agent.run(transcript, ctx)
        ctx.update(topic_res)
        ner_res = await self.ner_agent.run(transcript, ctx)
        ret_val = ', '.join(ner_res)
        if self.verbose:
            print(f"Extracted Names: {ret_val}")
        return True, ret_val
    
class GenerateTopicPipeline(BasePipeline):
    def __init__(
        self,
        api_key: str,
        pipline_name: str = "GenerateTopicPipeline",
        num_iterations_allowed: int = 1,
        verbose: bool = False):
        
        super().__init__(api_key, pipline_name, num_iterations_allowed, verbose)
        
        # Instantiate invokers with tailored instructions
        self.topic_agent = TranscriptGenerateExtractAgent(
            invoker=LLMInvoker(
                api_key,
                instructions=TOPIC_INSTRUCTIONS
            ),
            verbose=verbose,
            agent_name=AgentNames.TOPIC.value,
            return_response_with_wrapp=False
        )

    async def process(self, transcript: str) -> tuple[bool, str]:
        ctx: Dict[str, Any] = {}
        
        ret_val = await self.topic_agent.run(transcript, ctx)
        return True, ret_val
    
class FixTranscriptByLLMPipeline(BasePipeline):
    def __init__(
        self,
        api_key: str,
        pipline_name: str = "FixTranscriptByLLMPipeline",
        num_iterations_allowed: int = 1,
        verbose: bool = False):
        
        super().__init__(api_key, pipline_name, num_iterations_allowed, verbose)
        
        # Instantiate invokers with tailored instructions
        self.stt_output_corrector_agent = TranscriptGenerateExtractAgent(
            invoker=LLMInvoker(
                api_key,
                instructions=FIX_STT_OUTPUT_AGENT_INSTRUCTIONS
            ),
            verbose=verbose,
            agent_name=AgentNames.FIX_TRANSCRIPT_BY_LLM.value,
            return_response_with_wrapp=False
        )

    async def process(self, transcript: str) -> tuple[bool, str]:
        ret_val: str = ""
        ctx: Dict[str, Any] = {}
        
        ret_val = await self.stt_output_corrector_agent.run(transcript, ctx)
        return False, ret_val