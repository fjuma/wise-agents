import json
import logging
from typing import Callable, List, Optional
import uuid

from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel

from wiseagents.graphdb import WiseAgentGraphDB

from wiseagents.vectordb import Document, WiseAgentVectorDB

from wiseagents import WiseAgent, WiseAgentContext, WiseAgentMessage, WiseAgentRegistry, WiseAgentTransport, \
    WiseAgentTool
from wiseagents.llm import WiseAgentLLM


class PassThroughClientAgent(WiseAgent):
    '''This agent is used mainly for test purposes. It just passes the request to another agent and sends back the response to the client.'''
    '''Use Stomp protocol'''
    yaml_tag = u'!wiseagents.PassThroughClientAgent'
    def __init__(self, name, description , transport):
        '''Initialize the agent.
        
        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            transport (WiseAgentTransport): the transport to use for communication'''
        
        self._name = name
        super().__init__(name=name, description=description, transport=transport, llm=None)
    def __repr__(self):
        '''Return a string representation of the agent.'''
        return f"{self.__class__.__name__}(name={self.name}, description={self.description})"
    def process_request(self, request):
        '''Process a request message just passing it to the another agent.'''
        self.send_request(WiseAgentMessage(request, self.name), 'WiseIntelligentAgent' )
        return True
    def process_response(self, response):
        '''Process a response message just sending it back to the client.'''
        self.response_delivery(response)
        return True
    def process_event(self, event):
        '''Do nothing'''
        return True
    def process_error(self, error):
        '''Do nothing'''
        return True
    def get_recipient_agent_name(self, message):
        '''Return the name of the agent to send the message to.'''
        return self.name
    def stop(self):
        '''Do nothing'''
        pass
    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name
    @property
    def response_delivery(self) -> Optional[Callable[[], WiseAgentMessage]]:
        '''Get the function to deliver the response to the client.
        return (Callable[[], WiseAgentMessage]): the function to deliver the response to the client'''
        return self._response_delivery
    def set_response_delivery(self, response_delivery: Callable[[], WiseAgentMessage]):
        '''Set the function to deliver the response to the client.
        
        Args:
            response_delivery (Callable[[], WiseAgentMessage]): the function to deliver the response to the client'''
        self._response_delivery = response_delivery
class LLMOnlyWiseAgent(WiseAgent):
    '''This agent implementation is used to test the LLM only agent.'''
    '''Use Stomp protocol''' 
    yaml_tag = u'!wiseagents.LLMOnlyWiseAgent'
    def __init__(self, name: str, description: str, llm : WiseAgentLLM, trasport: WiseAgentTransport):
        '''Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            llm (WiseAgentLLM): the LLM agent to use for processing requests
            transport (WiseAgentTransport): the transport to use for communication'''
        self._name = name
        self._description = description
        self._transport = trasport
        llm_agent = llm
        super().__init__(name=name, description=description, transport=self.transport, llm=llm_agent)

    def __repr__(self):
        '''Return a string representation of the agent.'''
        return f"{self.__class__.__name__}(name={self.name}, description={self.description}, llm={self.llm}, transport={self.transport})"
        
    def process_event(self, event):
        '''Do nothing'''
        return True
    def process_error(self, error):
        '''Log the error and return True.'''
        logging.error(error)
        return True
    def process_request(self, request: WiseAgentMessage):
        '''Process a request message by passing it to the LLM agent and sending the response back to the client.

        Args:
            request (WiseAgentMessage): the request message to process'''
        llm_response = self.llm.process_single_prompt(request.message)
        self.send_response(WiseAgentMessage(message=llm_response.content, sender=self.name, context_name=request.context_name), request.sender )
        return True
    def process_response(self, response : WiseAgentMessage):
        '''Do nothing'''
        return True
    def get_recipient_agent_name(self, message):
        '''Return the name of the agent to send the message to.

        Args:
            message (WiseAgentMessage): the message to process'''
        return self.name
    def stop(self):
        pass    
    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name


    
    

class LLMWiseAgentWithTools(WiseAgent):
    '''This agent implementation is used to test the LLM agent providing a simple tool.'''
    '''Use Stomp protocol''' 
    yaml_tag = u'!wiseagents.LLMWiseAgentWithTools'
    def __init__(self, name: str, description: str, llm : WiseAgentLLM, transport: WiseAgentTransport, tools: List[str]):
        '''Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            llm (WiseAgentLLM): the LLM agent to use for processing requests
            transport (WiseAgentTransport): the transport to use for communication'''
        self._name = name
        self._description = description
        self._transport = transport
        llm_agent = llm
        self._tools = tools
        super().__init__(name=name, description=description, transport=self.transport, llm=llm_agent)

    def __repr__(self):
        '''Return a string representation of the agent.'''
        return f"{self.__class__.__name__}(name={self.name}, description={self.description}, llm={self.llm}, transport={self.transport})"
        
    def process_event(self, event):
        '''Do nothing'''
        return True
    def process_error(self, error):
        '''Log the error and return True.'''
        logging.error(error)
        return True
    def process_request(self, request: WiseAgentMessage):
        '''Process a request message by passing it to the LLM agent and sending the response back to the client.
        It invoke also the tool if required. Tool could be a callback function or another agent.

        Args:
            request (WiseAgentMessage): the request message to process'''
        logging.debug(f"IA Request received: {request}")
        chat_id= str(uuid.uuid4())
        ctx = WiseAgentRegistry.get_or_create_context(request.context_name)
        _plan_and_execute_tools_calls(ctx, chat_id, request.message, self._tools, self, request)
        #ctx.append_chat_completion(chat_uuid=chat_id, messages= {"role": "system", "content": self.llm.system_message})
        #ctx.append_chat_completion(chat_uuid=chat_id, messages= {"role": "user", "content": request.message})
        
        #for tool in self._tools:
        #    ctx.append_available_tool_in_chat(chat_uuid=chat_id, tools=WiseAgentRegistry.get_tool(tool).get_tool_OpenAI_format())
            
        #logging.debug(f"messages: {ctx.llm_chat_completion[chat_id]}, Tools: {ctx.get_available_tools_in_chat(chat_uuid=chat_id)}")    
        #llm_response = self.llm.process_chat_completion(ctx.llm_chat_completion[chat_id], ctx.get_available_tools_in_chat(chat_uuid=chat_id))
        
        ##calling tool
        #response_message = llm_response.choices[0].message
        #tool_calls = response_message.tool_calls
        #logging.debug(f"Tool calls: {tool_calls}")
        #logging.debug(f"Response message: {response_message}")
        # Step 2: check if the model wanted to call a function
        #if tool_calls is not None:
            # Step 3: call the function
            # TODO: the JSON response may not always be valid; be sure to handle errors
        #    ctx.append_chat_completion(chat_uuid=chat_id, messages= response_message)  # extend conversation with assistant's reply
            
            # Step 4: send the info for each function call and function response to the model
        #    for tool_call in tool_calls:
                #record the required tool call in the context/chatid
        #        ctx.append_required_tool_call(chat_uuid=chat_id, tool_name=tool_call.function.name)
                
        #    for tool_call in tool_calls:
        #        function_name = tool_call.function.name
        #        wise_agent_tool : WiseAgentTool = WiseAgentRegistry.get_tool(function_name)
        #        if wise_agent_tool.is_agent_tool:
                    #call the agent with correlation ID and complete the chat on response
        #            self.send_request(WiseAgentMessage(message=tool_call.function.arguments, sender=self.name, 
        #                                               chat_id=chat_id, tool_id=tool_call.id, context_name=request.context_name,
        #                                               route_response_to=request.sender), 
        #                              dest_agent_name=function_name)
        #        else:
        #            function_args = json.loads(tool_call.function.arguments)
        #            function_response = wise_agent_tool.exec(**function_args)
        #            logging.debug(f"Function response: {function_response}")
        #            ctx.append_chat_completion(chat_uuid=chat_id, messages= 
        #                {
        #                    "tool_call_id": tool_call.id,
        #                    "role": "tool",
        #                    "name": function_name,
        #                    "content": function_response,
        #                }
        #            )  # extend conversation with function response
        #            ctx.remove_required_tool_call(chat_uuid=chat_id, tool_name=tool_call.function.name)
            
        
        #SEND THE RESPONSE IF NOT ASYNC, OTHERWISE WE WILL DO LATER IN PROCESS_RESPONSE
        #if ctx.get_required_tool_calls(chat_uuid=chat_id) == []: # if all tool calls have been completed (no asynch needed)
        #    llm_response = self.llm.process_chat_completion(ctx.llm_chat_completion[chat_id], 
        #                                                    ctx.get_available_tools_in_chat(chat_uuid=chat_id))
        #    response_message = llm_response.choices[0].message
        #    logging.debug(f"sending response {response_message.content} to: {request.sender}")
        #    self.send_response(WiseAgentMessage(response_message.content, self.name), request.sender )
        #    ctx.llm_chat_completion.pop(chat_id)
        #    return True
    def process_response(self, response : WiseAgentMessage):
        '''Process a response message and sending the response back to the client.
        It invoke also the tool if required. Tool could be a callback function or another agent.

        Args:
            response (WiseAgentMessage): the response message to process
            '''
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
        '''Return the name of the agent to send the message to.

        Args:
            message (WiseAgentMessage): the message to process'''
        return self.name
    def stop(self):
        '''Do nothing'''
        pass    
    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name



class RAGWiseAgent(WiseAgent):
    """
    This agent implementation is used to test the RAG agent.
    Use Stomp protocol
    """
    yaml_tag = u'!wiseagents.RAGWiseAgent'

    def __init__(self, name: str, description: str, llm: WiseAgentLLM, vector_db: WiseAgentVectorDB,
                 transport: WiseAgentTransport, collection_name: Optional[str] = "wise-agents-collection"):
        '''Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            llm (WiseAgentLLM): the LLM agent to use for processing requests
            vector_db (WiseAgentVectorDB): the vector database to use for retrieving documents
            transport (WiseAgentTransport): the transport to use for communication
            collection_name Optional(str): the name of the collection to use for retrieving documents'''
        self._name = name
        self._description = description
        self._transport = transport
        self._vector_db = vector_db
        self._collection_name = collection_name
        llm_agent = llm
        super().__init__(name=name, description=description, transport=self.transport, llm=llm_agent,
                         vector_db=vector_db, collection_name=collection_name)

    def __repr__(self):
        """Return a string representation of the agent."""
        return (f"{self.__class__.__name__}(name={self.name}, description={self.description}, llm={self.llm},"
                f"vector_db={self.vector_db}, collection_name={self.collection_name}, transport={self.transport})")

    def process_event(self, event):
        '''Do nothing'''
        return True

    def process_error(self, error):
        '''Log the error and return True.'''
        logging.error(error)
        return True

    def process_request(self, request: WiseAgentMessage):
        '''Process a request message by passing it to the RAG agent and sending the response back to the client.

        Args:
            request (WiseAgentMessage): the request message to process'''
        retrieved_documents = self.vector_db.query([request.message], self.collection_name, 4)
        llm_response_with_sources = _create_and_process_rag_prompt(retrieved_documents[0], request.message, self.llm)
        self.send_response(WiseAgentMessage(llm_response_with_sources, self.name), request.sender)
        return True

    def process_response(self, response: WiseAgentMessage):
        '''Do nothing'''
        return True

    def get_recipient_agent_name(self, message):
        '''Return the name of the agent to send the message to.'''
        return self.name

    def stop(self):
        '''Do nothing'''
        pass

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

class GraphRAGWiseAgent(WiseAgent):
    """
    This agent implementation is used to test a knowledge graph based RAG agent.
    Use Stomp protocol
    """
    yaml_tag = u'!wiseagents.GraphRAGWiseAgent'

    def __init__(self, name: str, description: str, llm: WiseAgentLLM, graph_db: WiseAgentGraphDB,
                 transport: WiseAgentTransport):
        '''Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            llm (WiseAgentLLM): the LLM agent to use for processing requests
            graph_db (WiseAgentGraphDB): the graph database to use for retrieving documents
            transport (WiseAgentTransport): the transport to use for communication'''
        self._name = name
        self._description = description
        self._transport = transport
        self._graph_db = graph_db
        llm_agent = llm
        super().__init__(name=name, description=description, transport=self.transport, llm=llm_agent,
                         graph_db=graph_db)

    def __repr__(self):
        """Return a string representation of the agent."""
        return (f"{self.__class__.__name__}(name={self.name}, description={self.description}, llm={self.llm},"
                f"graph_db={self.graph_db}, transport={self.transport})")

    def process_event(self, event):
        '''Do nothing'''
        return True

    def process_error(self, error):
        '''Log the error and return True.'''
        logging.error(error)
        return True

    def process_request(self, request: WiseAgentMessage):
        '''Process a request message by passing it to the RAG agent and sending the response back to the client.

        Args:
            request (WiseAgentMessage): the request message to process'''
        retrieved_documents = self.graph_db.query_with_embeddings(query=request.message, k=1, retrieval_query=self._get_retrieval_query())
        llm_response_with_sources = _create_and_process_rag_prompt(retrieved_documents, request.message, self.llm)
        self.send_response(WiseAgentMessage(llm_response_with_sources, self.name), request.sender)
        return True

    def process_response(self, response: WiseAgentMessage):
        '''Do nothing'''
        return True

    def get_recipient_agent_name(self, message):
        '''Return the name of the agent to send the message to.'''
        return self.name

    def stop(self):
        '''Do nothing'''
        pass

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

    def _get_retrieval_query(self) -> str:
        '''Return the retrieval query to use for retrieving documents from the graph database.'''
        # this is specific to the test
        return """
          WITH node AS landmark, score AS similarity
          CALL  {
            WITH landmark
            MATCH (landmark)--(city)--(province)--(country)
            RETURN country.name AS Country
          }
          RETURN landmark.name + ' is located in ' + Country AS text, similarity as score, {} AS metadata
        """


class CoVeChallengerRAGWiseAgent(WiseAgent):
    """
    This agent implementation is used to challenge the response from a RAG agent using the
    Chain-of-Verification (CoVe) method (https://arxiv.org/pdf/2309.11495) to try to prevent
    hallucinations.
    Uses the Stomp protocol.
    """
    yaml_tag = u'!wiseagents.CoVeChallengerRAGWiseAgent'

    def __init__(self, name: str, description: str, llm: WiseAgentLLM, vector_db: WiseAgentVectorDB,
                 transport: WiseAgentTransport, collection_name: Optional[str] = "wise-agents-collection",
                 k: Optional[int] = 4, num_verification_questions: Optional[int] = 4):
        """
        Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            llm (WiseAgentLLM): the LLM agent to use for processing requests
            vector_db (WiseAgentVectorDB): the vector database to use for retrieving documents
            transport (WiseAgentTransport): the transport to use for communication
            collection_name Optional(str): the name of the collection to use in the vector database
            k Optional(int): the number of documents to retrieve from the vector database
            num_verification_questions Optional(int): the number of verification questions to generate
        """
        self._name = name
        self._description = description
        self._transport = transport
        self._vector_db = vector_db
        self._collection_name = collection_name
        self._k = k
        self._num_verification_questions = num_verification_questions
        llm_agent = llm
        super().__init__(name=name, description=description, transport=self.transport, llm=llm_agent,
                         vector_db=vector_db, collection_name=collection_name)

    def __repr__(self):
        """Return a string representation of the agent."""
        return (f"{self.__class__.__name__}(name={self.name}, description={self.description}, llm={self.llm},"
                f"vector_db={self.vector_db}, collection_name={self.collection_name}, k={self.k},"
                f"num_verification_questions={self._num_verification_questions}, transport={self.transport})")

    def process_event(self, event):
        """Do nothing"""
        return True

    def process_error(self, error):
        """Log the error and return True."""
        logging.error(error)
        return True

    def process_request(self, request: WiseAgentMessage):
        """
        Process a message containing a question and a baseline response to the question
        by challenging the baseline response to generate a revised response to the original question.

        Args:
            request (WiseAgentMessage): the request message to process
        """
        llm_response = self._create_and_process_chain_of_verification_prompts(request.message)
        self.send_response(WiseAgentMessage(llm_response, self.name), request.sender)
        return True

    def process_response(self, response: WiseAgentMessage):
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
    def k(self) -> int:
        """Get the number of documents to retrieve from the vector database."""
        return self._k

    @property
    def num_verification_questions(self) -> int:
        """Get the number of verification questions to generate."""
        return self._num_verification_questions

    def _create_and_process_chain_of_verification_prompts(self, message: str) -> str:
        """
        Create prompts to challenge the baseline response to a question to try to generate a revised response
        to the original question.

        Args:
            message (str): the message containing the question and baseline response
        """

        """Plan verifications"""
        prompt = (f"Given the following question and baseline response, generate a list of {self.num_verification_questions} "
                  f" verification questions that could help determine if there are any mistakes in the baseline response:\n{message}\n"
                  f"Your response should contain only the list of questions, one per line.\n")
        llm_response = self.llm.process_single_prompt(prompt)

        """Execute verifications"""
        verification_questions = llm_response.content.splitlines()
        verification_responses = ""
        for question in verification_questions:
            retrieved_documents = self.vector_db.query([question], self.collection_name, self.k)
            context = "\n".join([document.content for document in retrieved_documents[0]])
            prompt = (f"Answer the question based only on the following context:\n{context}\n"
                      f"Question: {question}\n")
            llm_response = self.llm.process_single_prompt(prompt)
            verification_responses = (verification_responses + "Verification Question: " + question + "\n"
                                      + "Verification Result: " + llm_response.content + "\n")

        """Generate the final revised response"""
        complete_info = message + "\n" + verification_responses
        prompt = (f"Given the following question, baseline response, and a list of verification questions and results,"
                  f" generate a revised response incorporating the verification results:\n{complete_info}\n"
                  f"Your response must contain only the revised response to the question in the JSON format shown below:\n"
                  f"{{'revised': 'Your revised response to the question.'}}\n")

        llm_response = self.llm.process_single_prompt(prompt)
        return llm_response.content


class SequentialCoordinatorWiseAgent(WiseAgent):
    """
    This agent will coordinate the execution of a sequence of agents.
    Use Stomp protocol.
    """
    yaml_tag = u'!wiseagents.SequentialCoordinatorWiseAgent'

    def __init__(self, name: str, description: str, transport: WiseAgentTransport, agents: List[str]):
        '''Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            transport (WiseAgentTransport): the transport to use for communication
            agents (List[str]): the list of agents to coordinate'''
        self._name = name
        self._agents = agents
        self._route_response_to = ""
        super().__init__(name=name, description=description, transport=transport, llm=None)

    def __repr__(self):
        '''Return a string representation of the agent.'''
        return f"{self.__class__.__name__}(name={self.name}, description={self.description}, agents={self.agents})"

    def process_request(self, request):
        '''Process a request message by passing it to the first agent in the sequence.

        Args:
            request (WiseAgentMessage): the request message to process'''
        logging.debug(f"Sequential coordinator received request: {request}")
        self._route_response_to = request.sender
        ctx = WiseAgentRegistry.get_or_create_context(request.context_name)
        ctx.set_agents_sequence(self._agents)
        self.send_request(WiseAgentMessage(message=request.message, sender=self.name, context_name=request.context_name), self._agents[0])

    def process_response(self, response):
        '''Process a response message by passing it to the next agent in the sequence.

        Args:
            response (WiseAgentMessage): the response message to process'''
        ctx = WiseAgentRegistry.get_or_create_context(response.context_name)
        next_agent = ctx.get_next_agent_in_sequence(response.sender)
        if next_agent is None:
            logging.debug(f"Sequential coordinator sending response from " + response.sender + " to " + self._route_response_to)
            self.send_response(WiseAgentMessage(message=response.message, sender=self.name, context_name=response.context_name), self._route_response_to)
        else:
            logging.debug(f"Sequential coordinator sending response from " + response.sender + " to " + next_agent)
            self.send_request(WiseAgentMessage(message=response.message, sender=self.name, context_name=response.context_name), next_agent)
        return True

    def process_event(self, event):
        '''Do nothing'''
        return True

    def process_error(self, error):
        '''Log the error and return True.'''
        logging.error(error)
        return True

    def get_recipient_agent_name(self, message):
        '''Return the name of the agent to send the message to.

        Args:
            message (WiseAgentMessage): the message to process'''
        return self.name

    def stop(self):
        '''Do nothing'''
        pass

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

    @property
    def agents(self) -> List[str]:
        """Get the list of agents."""
        return self._agents

    @property
    def response_delivery(self) -> Optional[Callable[[], WiseAgentMessage]]:
        '''Get the function to deliver the response to the client.
        return (Callable[[], WiseAgentMessage]): the function to deliver the response to the client'''
        return self._response_delivery

    def set_response_delivery(self, response_delivery: Callable[[], WiseAgentMessage]):
        '''Set the function to deliver the response to the client.'''
        self._response_delivery = response_delivery


class CoordinatorWiseAgent(WiseAgent):
    """
    This agent will coordinate the execution of a group of agents, make use of
    a reasoning agent to coordinate the collaboration and a final answer agent
    to format the final answer and send it back to the client.
    """
    yaml_tag = u'!wiseagents.CoordinatorWiseAgent'

    def __init__(self, name: str, description: str, transport: WiseAgentTransport, llm: WiseAgentLLM,
                 reasoning_agent_name: str, final_answer_agent_name: str):
        """
        Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            transport (WiseAgentTransport): the transport to use for communication
            llm (WiseAgentLLM): the LLM to use for coordinating the collaboration
            reasoning_agent_name (str): the name of the agent to use for reasoning
            final_answer_agent_name (str): the name of the agent to use for formatting the final answer
        """
        self._name = name
        self._reasoning_agent_name = reasoning_agent_name
        self._final_answer_agent_name = final_answer_agent_name
        self._route_response_to = ""
        super().__init__(name=name, description=description, transport=transport, llm=llm, is_orchestration_agent=True)

    def __repr__(self):
        """Return a string representation of the agent."""
        return (f"{self.__class__.__name__}(name={self.name}, description={self.description}, transport={self.transport},"
                f"llm={self.llm}, reasoning_agent_name={self._reasoning_agent_name},"
                f"final_answer_agent_name={self._final_answer_agent_name})")

    def process_request(self, request):
        """
        Process a request message by kicking off the collaboration.

        Args:
            request (WiseAgentMessage): the request message to process
        """
        logging.debug(f"Coordinator received request: {request}")
        self._route_response_to = request.sender
        chat_id = str(uuid.uuid4())
        ctx = WiseAgentRegistry.get_or_create_context(request.context_name)

        # Decompose the original request into sub-tasks
        task_decomposition_prompt = ("Given the following query and a description of the agents that are available to"
                                     " help answer the query, decompose the query into a list of sub-tasks."
                                     " Assume that one or more agents can work together to complete a sub-task."
                                     " Format the response with one sub-task per line and don't include anything else in"
                                     " the response:\n" + "Query: " + request.message + "\n" + "Available agents:\n" +
                                     "\n".join(WiseAgentRegistry.get_agent_names_and_descriptions()) + "\n")
        ctx.append_chat_completion(chat_uuid=chat_id, messages={"role": "system", "content": self.llm.system_message})
        ctx.append_chat_completion(chat_uuid=chat_id, messages={"role": "user", "content": task_decomposition_prompt})

        logging.debug(f"messages: {ctx.llm_chat_completion[chat_id]}")
        llm_response = self.llm.process_chat_completion(ctx.llm_chat_completion[chat_id], tools=[])
        sub_tasks = llm_response.choices[0].message.content.splitlines()
        ctx.set_task(chat_uuid=chat_id, task=request.message)
        for sub_task in sub_tasks:
            ctx.append_required_sub_task(chat_uuid=chat_id, sub_task=sub_task)

        # Determine the agents that can achieve its sub-task and execute them


    def process_response(self, response):
        """Process a response message by passing it to the next agent in the sequence.

        Args:
            response (WiseAgentMessage): the response message to process"""
        ctx = WiseAgentRegistry.get_or_create_context(response.context_name)
        self.send_response(WiseAgentMessage(message=response.message, sender=self.name, context_name=response.context_name), self._route_response_to)
        return True

    def process_event(self, event):
        """Do nothing"""
        return True

    def process_error(self, error):
        """Log the error and return True."""
        logging.error(error)
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

    @property
    def reasoning_agent_name(self) -> str:
        """Get the name of the reasoning agent."""
        return self._reasoning_agent_name

    @property
    def final_answer_agent_name(self) -> str:
        """Get the name of the final answer agent."""
        return self._final_answer_agent_name

    @property
    def response_delivery(self) -> Optional[Callable[[], WiseAgentMessage]]:
        """
        Get the function to deliver the response to the client.

        Returns:
            (Callable[[], WiseAgentMessage]): the function to deliver the response to the client
        """
        return self._response_delivery

    def set_response_delivery(self, response_delivery: Callable[[], WiseAgentMessage]):
        """
        Set the function to deliver the response to the client.

        Args:
            response_delivery (Callable[[], WiseAgentMessage]): the function to deliver the response to the
        """
        self._response_delivery = response_delivery

class ReasoningWiseAgent(WiseAgent):
    """
    This agent implementation is used to determine if a sub-task or a task
    has been completed.
    """
    yaml_tag = u'!wiseagents.ReasoningWiseAgent'

    def __init__(self, name: str, description: str, transport: WiseAgentTransport, llm: WiseAgentLLM):
        """
        Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            transport (WiseAgentTransport): the transport to use for communication
            llm (WiseAgentLLM): the LLM to use for reasoning
        """
        super().__init__(name=name, description=description, transport=transport, llm=llm, is_orchestration_agent=True)

    def __repr__(self):
        """Return a string representation of the agent."""
        return (f"{self.__class__.__name__}(name={self.name}, description={self.description}, transport={self.transport},"
                f"llm={self.llm})")

    def process_request(self, request):
        """
        TODO
        """

    def process_response(self, response):
        """
        TODO
        """
        return True

    def process_event(self, event):
        """Do nothing"""
        return True

    def process_error(self, error):
        """Log the error and return True."""
        logging.error(error)
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
    @property
    def response_delivery(self) -> Optional[Callable[[], WiseAgentMessage]]:
        """
        Get the function to deliver the response to the client.

        Returns:
            (Callable[[], WiseAgentMessage]): the function to deliver the response to the client
        """
        return self._response_delivery

    def set_response_delivery(self, response_delivery: Callable[[], WiseAgentMessage]):
        """
        Set the function to deliver the response to the client.

        Args:
            response_delivery (Callable[[], WiseAgentMessage]): the function to deliver the response to the
        """
        self._response_delivery = response_delivery

class FinalAnswerWiseAgent(WiseAgent):
    """
    This agent implementation is used to format the final answer in the desired manner.
    """
    yaml_tag = u'!wiseagents.FinalAnswerWiseAgent'

    def __init__(self, name: str, description: str, transport: WiseAgentTransport, llm: Optional[WiseAgentLLM] = None):
        """
        Initialize the agent.

        Args:
            name (str): the name of the agent
            description (str): a description of the agent
            transport (WiseAgentTransport): the transport to use for communication
            llm (Optional[WiseAgentLLM]): the LLM to use for reasoning
        """
        super().__init__(name=name, description=description, transport=transport, llm=llm, is_orchestration_agent=True)

    def __repr__(self):
        """Return a string representation of the agent."""
        return (f"{self.__class__.__name__}(name={self.name}, description={self.description}, transport={self.transport},"
                f"llm={self.llm})")

    def process_request(self, request):
        """
        TODO
        """

    def process_response(self, response):
        """
        TODO
        """
        return True

    def process_event(self, event):
        """Do nothing"""
        return True

    def process_error(self, error):
        """Log the error and return True."""
        logging.error(error)
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
    @property
    def response_delivery(self) -> Optional[Callable[[], WiseAgentMessage]]:
        """
        Get the function to deliver the response to the client.

        Returns:
            (Callable[[], WiseAgentMessage]): the function to deliver the response to the client
        """
        return self._response_delivery

    def set_response_delivery(self, response_delivery: Callable[[], WiseAgentMessage]):
        """
        Set the function to deliver the response to the client.

        Args:
            response_delivery (Callable[[], WiseAgentMessage]): the function to deliver the response to the
        """
        self._response_delivery = response_delivery


def _create_and_process_rag_prompt(retrieved_documents: List[Document], question: str, llm: WiseAgentLLM) -> str:
    '''Create a RAG prompt and process it with the LLM agent.

    Args:
        retrieved_documents (List[Document]): the list of retrieved documents
        question (str): the question to ask
        llm (WiseAgentLLM): the LLM agent to use for processing the prompt'''
    context = "\n".join([document.content for document in retrieved_documents])
    prompt = (f"Answer the question based only on the following context:\n{context}\n"
              f"Question: {question}\n")
    llm_response = llm.process_single_prompt(prompt)

    source_documents = ""
    for document in retrieved_documents:
        source_documents += f"Source Document:\n    Content: {document.content}\n    Metadata: {json.dumps(document.metadata)}\n\n"
    llm_response_with_sources = f"{llm_response.content}\n\nSource Documents:\n{source_documents}"
    return llm_response_with_sources

def _plan_and_execute_tools_calls(ctx: WiseAgentContext, chat_id: str, user_message: str,
                                  tools: List[str], sender_agent: WiseAgent, request: WiseAgentMessage):
    ctx.append_chat_completion(chat_uuid=chat_id, messages={"role": "system", "content": sender_agent.llm.system_message})
    ctx.append_chat_completion(chat_uuid=chat_id, messages={"role": "user", "content": user_message})

    for tool in tools:
        ctx.append_available_tool_in_chat(chat_uuid=chat_id,
                                          tools=WiseAgentRegistry.get_tool(tool).get_tool_OpenAI_format())

    logging.debug(f"messages: {ctx.llm_chat_completion[chat_id]}, Tools: {ctx.get_available_tools_in_chat(chat_uuid=chat_id)}")
    llm_response = sender_agent.llm.process_chat_completion(ctx.llm_chat_completion[chat_id],
                                               ctx.get_available_tools_in_chat(chat_uuid=chat_id))

    ##calling tool
    response_message = llm_response.choices[0].message
    tool_calls = response_message.tool_calls
    logging.debug(f"Tool calls: {tool_calls}")
    logging.debug(f"Response message: {response_message}")
    # Step 2: check if the model wanted to call a function
    if tool_calls is not None:
        # Step 3: call the function
        # TODO: the JSON response may not always be valid; be sure to handle errors
        ctx.append_chat_completion(chat_uuid=chat_id,
                                   messages=response_message)  # extend conversation with assistant's reply

        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            # record the required tool call in the context/chatid
            ctx.append_required_tool_call(chat_uuid=chat_id, tool_name=tool_call.function.name)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            wise_agent_tool: WiseAgentTool = WiseAgentRegistry.get_tool(function_name)
            if wise_agent_tool.is_agent_tool:
                # call the agent with correlation ID and complete the chat on response
                sender_agent.send_request(WiseAgentMessage(message=tool_call.function.arguments, sender=sender_agent.name,
                                                   chat_id=chat_id, tool_id=tool_call.id,
                                                   context_name=request.context_name,
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

    # SEND THE RESPONSE IF NOT ASYNC, OTHERWISE WE WILL DO LATER IN PROCESS_RESPONSE
    if ctx.get_required_tool_calls(chat_uuid=chat_id) == []:  # if all tool calls have been completed (no asynch needed)
        llm_response = sender_agent.llm.process_chat_completion(ctx.llm_chat_completion[chat_id],
                                                        ctx.get_available_tools_in_chat(chat_uuid=chat_id))
        response_message = llm_response.choices[0].message
        logging.debug(f"sending response {response_message.content} to: {request.sender}")
        sender_agent.send_response(WiseAgentMessage(response_message.content, sender_agent.name), request.sender)
        ctx.llm_chat_completion.pop(chat_id)
        return True

