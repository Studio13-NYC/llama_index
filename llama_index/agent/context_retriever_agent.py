"""Context retriever agent."""

from typing import List, Optional, Type, Union

from llama_index.agent.openai_agent import (
    DEFAULT_MAX_FUNCTION_CALLS,
    DEFAULT_MODEL_NAME,
    BaseOpenAIAgent,
)
from llama_index.callbacks import CallbackManager
from llama_index.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.core import BaseRetriever
from llama_index.llms.base import LLM, ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_utils import is_function_calling_model
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.prompts import PromptTemplate
from llama_index.schema import NodeWithScore
from llama_index.tools import BaseTool
from llama_index.utils import print_text

# inspired by DEFAULT_QA_PROMPT_TMPL from llama_index/prompts/default_prompts.py
DEFAULT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "either pick the corresponding tool or answer the function: {query_str}\n"
)
DEFAULT_QA_PROMPT = PromptTemplate(DEFAULT_QA_PROMPT_TMPL)


class ContextRetrieverOpenAIAgent(BaseOpenAIAgent):
    """
    Is an OpenAI agent that retrieves context nodes from a retriever and formats
    them into a query string to interact with a large language model (LLM). It
    provides both synchronous (`chat`) and asynchronous (`achat`) methods for chatting.

    Attributes:
        _tools (List[BaseTool]): Initialized in the `__init__` method. It holds a
            list of BaseTool objects, which are used by the agent to perform various
            tasks.
        _qa_prompt (PromptTemplate): Used to format a query string. It takes the
            context string, obtained by joining retrieved texts from the retriever,
            and combines it with the original message to form a formatted query
            string for OpenAI API calls.
        _retriever (BaseRetriever): Used for retrieving nodes from a knowledge
            graph based on a given input message.
        _context_separator (str): Used to separate context strings retrieved from
            the retriever.

    """

    def __init__(
        self,
        tools: List[BaseTool],
        retriever: BaseRetriever,
        qa_prompt: PromptTemplate,
        context_separator: str,
        llm: OpenAI,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """
        Initializes an instance with specified parameters: tools, retriever, prompt
        template, separator, OpenAI language model, memory, prefix messages,
        verbosity, maximum function calls, and callback manager. It sets attributes
        for these parameters and calls its parent's `__init__` method.

        Args:
            tools (List[BaseTool]): Initialized with the value provided to it
                during object creation.
            retriever (BaseRetriever): Assigned to an instance variable `_retriever`.
                This suggests that retriever is responsible for retrieving relevant
                information from a source, possibly a database or another data structure.
            qa_prompt (PromptTemplate): Assigned to an instance variable with the
                same name. This suggests that `qa_prompt` represents a template
                for constructing questions or prompts used in question-answering
                processes.
            context_separator (str): Used to separate multiple contexts in the
                input. It appears to play a crucial role in handling context-switching
                operations within the class's functionality.
            llm (OpenAI): Used to represent an instance of an open AI language
                model, which will be utilized by the class for natural language
                processing tasks.
            memory (BaseMemory): Used to initialize an instance of this class. It
                represents the memory component responsible for storing and managing
                context information during conversations.
            prefix_messages (List[ChatMessage]): Initialized to an empty list by
                default. It represents a collection of messages that serve as
                prefixes for conversation flows.
            verbose (bool): False by default. It controls the verbosity level of
                the object, enabling or disabling detailed logging information
                during its operation.
            max_function_calls (int): 0 by default. It limits the number of recursive
                function calls allowed before raising an exception, ensuring against
                infinite recursion and potential stack overflow.
            callback_manager (Optional[CallbackManager]): Optional by default. It
                allows for an external CallbackManager instance to be passed, which
                can manage callback functions used by other parts of the class.

        """
        super().__init__(
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )
        self._tools = tools
        self._qa_prompt = qa_prompt
        self._retriever = retriever
        self._context_separator = context_separator

    @classmethod
    def from_tools_and_retriever(
        cls,
        tools: List[BaseTool],
        retriever: BaseRetriever,
        qa_prompt: Optional[PromptTemplate] = None,
        context_separator: str = "\n",
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
    ) -> "ContextRetrieverOpenAIAgent":
        """
        Initializes an instance with given tools, retriever, and options for OpenAI
        interaction, including LLMAgent model, memory buffer, and callback manager.
        It sets default values for optional parameters and validates input.

        Args:
            tools (List[BaseTool]): Required. It represents a list of base tools
                used for context retrieval.
            retriever (BaseRetriever): Required for constructing an instance of `ContextRetrieverOpenAIAgent`.
            qa_prompt (Optional[PromptTemplate]): Set to None by default. It is
                used as a prompt for asking questions during conversations, with
                the DEFAULT_QA_PROMPT as its fallback value if not provided.
            context_separator (str): Used to separate context parts when constructing
                the prompt for the retrieval model. It defaults to "\n".
            llm (Optional[LLM]): Used to set the language model, which defaults
                to OpenAI if not provided.
            chat_history (Optional[List[ChatMessage]]): Optional, meaning it
                defaults to an empty list if not provided. It stores previous
                messages exchanged between the user and the chatbot.
            memory (Optional[BaseMemory]): Optional by default. If provided, it
                will be used as the memory instance; otherwise, a default memory
                instance is created from chat history and LLM using the
                `memory_cls.from_defaults` method.
            memory_cls (Type[BaseMemory]): Set to `ChatMemoryBuffer` by default.
                It allows users to specify an alternate memory class if needed,
                overriding the default behavior.
            verbose (bool): False by default. It controls the level of verbosity
                when the class is instantiated, which could potentially affect
                logging or other output produced by the class.
            max_function_calls (int): Used to set the maximum number of recursive
                calls allowed for the LLM model.
            callback_manager (Optional[CallbackManager]): Used to configure callbacks
                for the LLM. It allows the developer to manage callback functions
                that are triggered during the execution of certain events in the
                OpenAI model.
            system_prompt (Optional[str]): Used to specify a system prompt for the
                context retriever. If provided, this prompt will be added as the
                first message in the chat history.
            prefix_messages (Optional[List[ChatMessage]]): Used to specify prefix
                messages for the context retriever. It can be used to provide
                additional context or system information.

        Returns:
            "ContextRetrieverOpenAIAgent": An instance of a class. This object
            represents an OpenAI agent that uses a retriever and tools to interact
            with users, based on the input parameters provided.

        """
        qa_prompt = qa_prompt or DEFAULT_QA_PROMPT
        chat_history = chat_history or []
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")
        if callback_manager is not None:
            llm.callback_manager = callback_manager

        memory = memory or memory_cls.from_defaults(chat_history=chat_history, llm=llm)

        if not is_function_calling_model(llm.model):
            raise ValueError(
                f"Model name {llm.model} does not support function calling API."
            )
        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(
            tools=tools,
            retriever=retriever,
            qa_prompt=qa_prompt,
            context_separator=context_separator,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    def _get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._tools

    def _build_formatted_message(self, message: str) -> str:
        # augment user message
        """
        Retrieves relevant nodes and their scores from the input message, extracts
        node contents, concatenates them into a context string, and then formats
        an OpenAI query prompt using this context string and the original message.

        """
        retrieved_nodes_w_scores: List[NodeWithScore] = self._retriever.retrieve(
            message
        )
        retrieved_nodes = [node.node for node in retrieved_nodes_w_scores]
        retrieved_texts = [node.get_content() for node in retrieved_nodes]

        # format message
        context_str = self._context_separator.join(retrieved_texts)
        return self._qa_prompt.format(context_str=context_str, query_str=message)

    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        """
        Processes user input, formats it, and sends it to the superclass for further
        processing. It also handles chat history and tool choice configuration.

        Args:
            message (str): Required. It represents the message to be sent through
                the chat interface and will be used as an input for generating the
                response from the agent.
            chat_history (Optional[List[ChatMessage]]): Optional by default with
                a value of None, allowing for the chat history to be passed as an
                argument or not at all.
            tool_choice (Union[str, dict]): Default set to "auto". It accepts
                either a string or a dictionary as its value and controls the tool
                used for the chat session.

        Returns:
            AgentChatResponse: Likely a response generated by an AI agent or a
            conversational model after processing the input message and any relevant
            context from the chat history.

        """
        formatted_message = self._build_formatted_message(message)
        if self._verbose:
            print_text(formatted_message + "\n", color="yellow")

        return super().chat(
            formatted_message, chat_history=chat_history, tool_choice=tool_choice
        )

    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        """
        Formats and prints a message if verbose mode is on, then calls its
        superclass's `achat` method with formatted message, chat history, and tool
        choice.

        Args:
            message (str): Required by default. It represents a message that is
                passed to the chatbot for processing.
            chat_history (Optional[List[ChatMessage]]): Optional. It allows for
                passing chat history, if available, to assist in processing the
                current message.
            tool_choice (Union[str, dict]): Optional with a default value of "auto".
                This means it can be either a string or a dictionary, and if not
                provided, it defaults to "auto".

        Returns:
            AgentChatResponse: Awaited from a superclass's method with the same
            name. The returned response likely contains information about the chat
            interaction, possibly including the generated response to the input message.

        """
        formatted_message = self._build_formatted_message(message)
        if self._verbose:
            print_text(formatted_message + "\n", color="yellow")

        return await super().achat(
            formatted_message, chat_history=chat_history, tool_choice=tool_choice
        )

    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._get_tools(message)
