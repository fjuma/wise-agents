import json
import logging
import uuid
from typing import Callable, List, Optional

from openai.types.chat import ChatCompletionMessageParam
from wiseagents import WiseAgent, WiseAgentCollaborationType, WiseAgentMessage, WiseAgentRegistry, WiseAgentTransport, \
    WiseAgentTool
from wiseagents.llm import WiseAgentLLM


class PassThroughClientAgent(WiseAgent):
    """
    This utility agent simply passes a request that it receives to another agent and sends the
    response back to the client.
    """
    yaml_tag = u'!wiseagents.agents.PassThroughClientAgent'
    
    _response_delivery = None
    _destination_agent_name = None
    
    def __new__(cls, *args, **kwargs):
        """Create a new instance of the class, setting default values for the optional instance variables."""
        obj = super().__new__(cls)
        return obj

    def __init__(self, name: str, description: str , transport: WiseAgentTransport,
                 destination_agent_name: Optional[str] = "WiseIntelligentAgent"):
        """
        Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            transport (WiseAgentTransport): the transport to use for communication
            destination_agent_name (str): the name of the agent to send requests to
        """
        self._name = name
        self._destination_agent_name = destination_agent_name
        super().__init__(name=name, description=description, transport=transport, llm=None)

    def __repr__(self):
        """Return a string representation of the agent."""
        return f"{self.__class__.__name__}(name={self.name}, \
            description={self.description}, transport={self.transport}, \
            destination_agent_name={self.destination_agent_name},\
            response_delivery={self.response_delivery}"
     
    def handle_request(self, request):
        """Process a request message by just passing it to another agent."""
        self.send_request(WiseAgentMessage(request, self.name), self.destination_agent_name)
        return True

    def process_response(self, response):
        """Process a response message just sending it back to the client."""
        if self.response_delivery is not None:
            self.response_delivery(response)
        else:
            logging.debug(f"############################### Not sending response {response}")
        return True

    def process_event(self, event):
        """Do nothing"""
        return True

    def process_error(self, error):
        """Do nothing"""
        return True

    def get_recipient_agent_name(self, message):
        """Return the name of the agent to send the message to."""
        return self.name

    def stop(self):
        """Do nothing"""
        pass

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

    @property
    def destination_agent_name(self) -> str:
        """Get the name of the agent to send requests to."""
        return self._destination_agent_name

    @property
    def response_delivery(self) -> Optional[Callable[[], WiseAgentMessage]]:
        """Get the function to deliver the response to the client.
        return (Callable[[], WiseAgentMessage]): the function to deliver the response to the client"""
        return self._response_delivery

    def set_response_delivery(self, response_delivery: Callable[[], WiseAgentMessage]):
        """
        Set the function to deliver the response to the client.

        Args:
            response_delivery (Callable[[], WiseAgentMessage]): the function to deliver the response to the client
        """
        self._response_delivery = response_delivery

class LLMOnlyWiseAgent(WiseAgent):
    """
    This utility agent simply passes a request that it receives to an LLM for processing and returns the
    response received from the LLM.
    """
    yaml_tag = u'!wiseagents.agents.LLMOnlyWiseAgent'
    
    def __new__(cls, *args, **kwargs):
        """Create a new instance of the class, setting default values for the instance variables."""
        obj = super().__new__(cls)
        obj._system_message = None
        return obj

    def __init__(self, name: str, description: str, llm : WiseAgentLLM, transport: WiseAgentTransport,
                 system_message: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            llm (WiseAgentLLM): the LLM agent to use for processing requests
            transport (WiseAgentTransport): the transport to use for communication
            system_message (Optional[str]): the optional system message to be used by the collaborator when processing
            chat completions using its LLM
        """
        self._name = name
        self._description = description
        self._transport = transport
        llm_agent = llm
        super().__init__(name=name, description=description, transport=self.transport, llm=llm_agent,
                         system_message=system_message)

    def __repr__(self):
        """Return a string representation of the agent."""
        return (f"{self.__class__.__name__}(name={self.name}, description={self.description}, llm={self.llm}, transport={self.transport},"
                f"system_message={self.system_message})")
        
    def process_event(self, event):
        """Do nothing"""
        return True

    def process_error(self, error):
        """Log the error and return True."""
        logging.error(error)
        return True

    def process_request(self, request: WiseAgentMessage, conversation_history: List[ChatCompletionMessageParam]) -> str:
        """
        Process a request message by passing it to the LLM.

        Args:
            request (WiseAgentMessage): the request message to process
            conversation_history (List[ChatCompletionMessageParam]): The conversation history that
            can be used while processing the request. If this agent isn't involved in a type of
            collaboration that makes use of the conversation history, this will be an empty list.
        """
        conversation_history.append({"role": "system", "content": self.system_message or self.llm.system_message})
        conversation_history.append({"role": "user", "content": request.message})
        llm_response = self.llm.process_chat_completion(conversation_history, [])
        return llm_response.choices[0].message.content

    def process_response(self, response : WiseAgentMessage):
        """Do nothing"""
        return True

    def get_recipient_agent_name(self, message):
        """
        Return the name of the agent to send the message to.

        Args:
            message (WiseAgentMessage): the message to process
        """
        return self.name

    def stop(self):
        pass

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name


class LLMWiseAgentWithTools(WiseAgent):
    """
    This utility agent makes use of an LLM along with tools to process a request and determine the response
    to send back to the client.
    """
    yaml_tag = u'!wiseagents.agents.LLMWiseAgentWithTools'
    
    def __new__(cls, *args, **kwargs):
        """Create a new instance of the class, setting default values for the instance variables."""
        obj = super().__new__(cls)
        obj._system_message = None
        return obj
    
    def __init__(self, name: str, description: str, llm : WiseAgentLLM, transport: WiseAgentTransport, tools: List[str],
                 system_message: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            llm (WiseAgentLLM): the LLM agent to use for processing requests
            transport (WiseAgentTransport): the transport to use for communication
            system_message (Optional[str]): the optional system message to be used by the collaborator when processing
            chat completions using its LLM
        """
        self._name = name
        self._description = description
        self._transport = transport
        llm_agent = llm
        self._tools = tools
        super().__init__(name=name, description=description, transport=self.transport, llm=llm_agent, system_message=system_message)

    def __repr__(self):
        """Return a string representation of the agent."""
        return (f"{self.__class__.__name__}(name={self.name}, description={self.description}, llm={self.llm}, transport={self.transport},"
                f"system_message={self.system_message})")

    def process_event(self, event):
        """Do nothing"""
        return True

    def process_error(self, error):
        """Log the error and return True."""
        logging.error(error)
        return True

    def process_request(self, request: WiseAgentMessage, conversation_history: List[ChatCompletionMessageParam]) -> str:
        """
        Process a request message by passing it to the LLM agent.
        It also invokes tool(s) if required. Tool(s) could be a callback function or another agent.

        Args:
            request (WiseAgentMessage): the request message to process
            conversation_history (List[ChatCompletionMessageParam]): The conversation history that
            can be used while processing the request. If this agent isn't involved in a type of
            collaboration that makes use of the conversation history, this will be an empty list.
        """
        logging.debug(f"IA Request received: {request}")
        chat_id= str(uuid.uuid4())
        ctx = WiseAgentRegistry.get_or_create_context(request.context_name)
        ctx.append_chat_completion(chat_uuid=chat_id, messages= {"role": "system", "content": self.llm.system_message})
        ctx.append_chat_completion(chat_uuid=chat_id, messages= {"role": "user", "content": request.message})
        
        for tool in self._tools:
            ctx.append_available_tool_in_chat(chat_uuid=chat_id, tools=WiseAgentRegistry.get_tool(tool).get_tool_OpenAI_format())
            
        logging.debug(f"messages: {ctx.llm_chat_completion[chat_id]}, Tools: {ctx.get_available_tools_in_chat(chat_uuid=chat_id)}")    
        llm_response = self.llm.process_chat_completion(ctx.llm_chat_completion[chat_id], ctx.get_available_tools_in_chat(chat_uuid=chat_id))
        
        ##calling tool
        response_message = llm_response.choices[0].message
        tool_calls = response_message.tool_calls
        logging.debug(f"Tool calls: {tool_calls}")
        logging.debug(f"Response message: {response_message}")
        # Step 2: check if the model wanted to call a function
        if tool_calls is not None:
            # Step 3: call the function
            # TODO: the JSON response may not always be valid; be sure to handle errors
            ctx.append_chat_completion(chat_uuid=chat_id, messages= response_message)  # extend conversation with assistant's reply
            
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                #record the required tool call in the context/chatid
                ctx.append_required_tool_call(chat_uuid=chat_id, tool_name=tool_call.function.name)
                
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                wise_agent_tool : WiseAgentTool = WiseAgentRegistry.get_tool(function_name)
                if wise_agent_tool.is_agent_tool:
                    #call the agent with correlation ID and complete the chat on response
                    self.send_request(WiseAgentMessage(message=tool_call.function.arguments, sender=self.name, 
                                                       chat_id=chat_id, tool_id=tool_call.id, context_name=request.context_name,
                                                       route_response_to=request.sender), 
                                      dest_agent_name=function_name)
                else:
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = wise_agent_tool.exec(**function_args)
                    logging.debug(f"Function response: {function_response}")
                    ctx.append_chat_completion(chat_uuid=chat_id, messages= 
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )  # extend conversation with function response
                    ctx.remove_required_tool_call(chat_uuid=chat_id, tool_name=tool_call.function.name)
            
        
        #SEND THE RESPONSE IF NOT ASYNC, OTHERWISE WE WILL DO LATER IN PROCESS_RESPONSE
        if ctx.get_required_tool_calls(chat_uuid=chat_id) == []: # if all tool calls have been completed (no asynch needed)
            llm_response = self.llm.process_chat_completion(ctx.llm_chat_completion[chat_id], 
                                                            ctx.get_available_tools_in_chat(chat_uuid=chat_id))
            response_message = llm_response.choices[0].message
            logging.debug(f"sending response {response_message.content} to: {request.sender}")
            ctx.llm_chat_completion.pop(chat_id)
            return response_message.content

    def process_response(self, response : WiseAgentMessage):
        """
        Process a response message and sending the response back to the client.
        It invoke also the tool if required. Tool could be a callback function or another agent.

        Args:
            response (WiseAgentMessage): the response message to process
        """
        print(f"Response received: {response}")
        chat_id = response.chat_id
        ctx = WiseAgentRegistry.get_or_create_context(response.context_name)
        ctx.append_chat_completion(chat_uuid=chat_id, messages= 
            {
                "tool_call_id": response.tool_id,
                "role": "tool",
                "name": response.sender,
                "content": response.message,
            }
        )  # extend conversation with function response
        ctx.remove_required_tool_call(chat_uuid=chat_id, tool_name=response.sender)
            
        if ctx.get_required_tool_calls(chat_uuid=chat_id) == []: # if all tool calls have been completed (no asynch needed)
            llm_response = self.llm.process_chat_completion(ctx.llm_chat_completion[chat_id], 
                                                            ctx.get_available_tools_in_chat(chat_uuid=chat_id))
            response_message = llm_response.choices[0].message
            logging.debug(f"sending response {response_message.content} to: {response.route_response_to}")
            self.send_response(WiseAgentMessage(response_message.content, self.name), response.route_response_to )
            ctx.llm_chat_completion.pop(chat_id)
            return True

    def get_recipient_agent_name(self, message):
        """
        Return the name of the agent to send the message to.

        Args:
            message (WiseAgentMessage): the message to process
        """
        return self.name

    def stop(self):
        """Do nothing"""
        pass

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name


