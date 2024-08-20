import logging
import os
import threading
import pytest

from wiseagents import WiseAgent, WiseAgentMessage, WiseAgentRegistry
from wiseagents.llm import OpenaiAPIWiseAgentLLM
from wiseagents.transports import StompWiseAgentTransport
from wiseagents.wise_agent_impl import (CoordinatorWiseAgent, FinalAnswerWiseAgent, LLMOnlyWiseAgent, PassThroughClientAgent,
                                        ReasoningWiseAgent, SequentialCoordinatorWiseAgent)

cond = threading.Condition()


def response_delivered(message: WiseAgentMessage):
    with cond:
        response = message.message
        assert "Farah" in response
        assert "Agent1" in response
        assert "Agent2" in response
        print(f"C Response delivered: {response}")
        cond.notify()

@pytest.mark.needsllama
def test_sequential_coordinator():
    """
    Requires STOMP_USER and STOMP_PASSWORD.
    """
    llm1 = OpenaiAPIWiseAgentLLM(system_message="Your name is Agent1. Answer my greeting saying Hello and my name and tell me your name.",
                                 model_name="llama3.1", remote_address="http://localhost:11434/v1")
    agent1 = LLMOnlyWiseAgent(name="Agent1", description="This is a test agent", llm=llm1,
                              trasport=StompWiseAgentTransport(host='localhost', port=61616, agent_name="Agent1"))

    llm2 = OpenaiAPIWiseAgentLLM(system_message="Your name is Agent2. Answer my greeting saying Hello and include all names from the given message and tell me your name.",
                                 model_name="llama3.1", remote_address="http://localhost:11434/v1")
    agent2 = LLMOnlyWiseAgent(name="Agent2", description="This is a test agent", llm=llm2,
                              trasport=StompWiseAgentTransport(host='localhost', port=61616, agent_name="Agent2"))

    coordinator = SequentialCoordinatorWiseAgent(name="SequentialCoordinator", description="This is a coordinator agent",
                                                 transport=StompWiseAgentTransport(host='localhost', port=61616, agent_name="SequentialCoordinator"),
                                                 agents=["Agent1", "Agent2"])

    with cond:
        client_agent1 = PassThroughClientAgent(name="PassThroughClientAgent1", description="This is a test agent",
                                               transport=StompWiseAgentTransport(host='localhost', port=61616, agent_name="PassThroughClientAgent1")
                                               )
        client_agent1.set_response_delivery(response_delivered)
        client_agent1.send_request(WiseAgentMessage("My name is Farah", "PassThroughClientAgent1"),
                                   "SequentialCoordinator")
        cond.wait()

        for agent in WiseAgentRegistry.get_agents():
            print(f"Agent: {agent}")
        for message in WiseAgentRegistry.get_or_create_context('default').message_trace:
            print(f'{message.sender} : {message.message} ')

@pytest.mark.needsllama
def test_coordinator():
    groq_api_key = os.getenv("API_KEY")
    llm1 = OpenaiAPIWiseAgentLLM(system_message="Answer my greeting saying Hello and my name",
                                model_name="llama-3.1-70b-versatile",
                                remote_address="https://api.groq.com/openai/v1",
                                api_key=groq_api_key)
    agent1 = CoordinatorWiseAgent(name="Coordinator", description="This is a test agent", llm=llm1,
                                  transport=StompWiseAgentTransport(host='localhost', port=61616, agent_name="Coordinator"),
                                  reasoning_agent_name="ReasoningAgent", final_answer_agent_name="FinalAnswerAgent")

    llm2 = OpenaiAPIWiseAgentLLM(system_message="Answer my greeting saying Hello and my name",
                                 model_name="llama-3.1-70b-versatile",
                                 remote_address="https://api.groq.com/openai/v1",
                                 api_key=groq_api_key)
    agent2 = ReasoningWiseAgent(name="ReasoningAgent", description="This is a test agent", llm=llm2,
                              transport=StompWiseAgentTransport(host='localhost', port=61616, agent_name="ReasoningAgent"))

    llm3 = OpenaiAPIWiseAgentLLM(system_message="Answer my greeting saying Hello and my name",
                                 model_name="llama-3.1-70b-versatile",
                                 remote_address="https://api.groq.com/openai/v1",
                                 api_key=groq_api_key)
    agent3 = FinalAnswerWiseAgent(name="FinalAnswerAgent", description="This is a test agent", llm=llm3,
                                  transport=StompWiseAgentTransport(host='localhost', port=61616,
                                                                    agent_name="FinalAnswerAgent"))

    llm4 = OpenaiAPIWiseAgentLLM(system_message="Provide information about the given error message",
                                 model_name="llama-3.1-70b-versatile",
                                 remote_address="https://api.groq.com/openai/v1",
                                 api_key=groq_api_key)
    agent4 = LLMOnlyWiseAgent(name="Agent4", description="This agent provides information about error messages using Source1", llm=llm4,
                              trasport=StompWiseAgentTransport(host='localhost', port=61616,
                                                               agent_name="Agent4"))

    llm5 = OpenaiAPIWiseAgentLLM(system_message="Provide information about the given error message",
                                 model_name="llama-3.1-70b-versatile",
                                 remote_address="https://api.groq.com/openai/v1",
                                 api_key=groq_api_key)
    agent5 = LLMOnlyWiseAgent(name="Agent5", description="This agent provides information about error messages using Source2",
                              llm=llm5,
                              trasport=StompWiseAgentTransport(host='localhost', port=61616,
                                                               agent_name="Agent5"))

    llm6 = OpenaiAPIWiseAgentLLM(system_message="Determine the underlying cause of a problem given information about the problem",
                                 model_name="llama-3.1-70b-versatile",
                                 remote_address="https://api.groq.com/openai/v1",
                                 api_key=groq_api_key)
    agent6 = LLMOnlyWiseAgent(name="Agent6",
                              description="This agent describes the underlying cause of a problem given information about the problem. This agent"
                                          "should be called after getting information about the problem using Agent4 and Agent5.",
                              llm=llm6,
                              trasport=StompWiseAgentTransport(host='localhost', port=61616,
                                                               agent_name="Agent6"))

    llm7 = OpenaiAPIWiseAgentLLM(
        system_message="Determine the underlying cause of a problem given information about the problem",
        model_name="llama-3.1-70b-versatile",
        remote_address="https://api.groq.com/openai/v1",
        api_key=groq_api_key)
    agent7 = LLMOnlyWiseAgent(name="Agent7",
                              description="This agent can be used to verify the information provided by other agents",
                              llm=llm7,
                              trasport=StompWiseAgentTransport(host='localhost', port=61616,
                                                               agent_name="Agent7"))

    with cond:
        client_agent1 = PassThroughClientAgent(name="PassThroughClientAgent1", description="This is a test agent",
                                               transport=StompWiseAgentTransport(host='localhost', port=61616, agent_name="PassThroughClientAgent1")
                                               )
        client_agent1.set_response_delivery(response_delivered)
        client_agent1.send_request(WiseAgentMessage("How do I prevent the following exception from occuring:"
                                                    "Exception Details: java.lang.NullPointerException at com.example.ExampleApp.processData(ExampleApp.java:47)", 
                                                    "PassThroughClientAgent1"),
                                   "Coordinator")
        cond.wait()

        for agent in WiseAgentRegistry.get_agents():
            print(f"Agent: {agent}")
        for message in WiseAgentRegistry.get_or_create_context('default').message_trace:
            print(f'{message.sender} : {message.message} ')
