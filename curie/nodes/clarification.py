# References used:
# https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop
from nodes.base_node import BaseNode, NodeConfig
from langgraph.graph import StateGraph, START, END
from scheduler import SchedNode
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage 

import os
import json

import settings
import model
import utils

class Clarification(BaseNode):
    """Node responsible for clarifying ambiguous user input at the start of a DeepResearch task."""

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)
        self.create_transition_objs()
        self.max_clarification_rounds = 2
        self.questions_per_round = 5

    def create_transition_objs(self):
        self.node_config.transition_objs["done"] = lambda clarified_context: {
            "messages": clarified_context,
            "next_agent": "clarification_router",
            "clarification_complete": True
        }

    def transition_handle_func(self, state):
        """Forward the clarified context to the router."""
        self.curie_logger.info("------------ Handle Clarification ------------")
        
        # Get clarification data from metadata store
        clarification_data = self.metadata_store.get(self.sched_namespace, "clarification_data").dict()["value"]
        
        # Build clarified context
        clarified_context = self._build_clarified_context(clarification_data)
        
        return self.node_config.transition_objs["done"](clarified_context)

    def _build_clarified_context(self, clarification_data):
        """Build a comprehensive context from user responses."""
        original_question = clarification_data.get("original_question", "")
        rounds = clarification_data.get("rounds", [])
        
        context = f"Original question: {original_question}\n\n"
        
        for i, round_data in enumerate(rounds):
            context += f"Clarification round {i+1}:\n"
            for q, a in round_data.items():
                if a and a.strip():  # Only include answered questions
                    context += f"Q: {q}\n"
                    context += f"A: {a}\n"
            context += "\n"
        
        return context

    def create_subgraph(self):
        """Creates a Node subgraph specific to Clarification."""
        subgraph_builder = StateGraph(self.State)
        clarification_node = self._create_clarification_node()

        subgraph_builder.add_node(self.node_config.name, clarification_node)
        subgraph_builder.add_edge(START, self.node_config.name)

        subgraph = subgraph_builder.compile(checkpointer=self.memory)
        
        def call_subgraph(state: self.State) -> self.State:
            # Get original question
            original_question = state["messages"][0].content if state["messages"] else ""
            
            # Initialize clarification data
            clarification_data = {
                "original_question": original_question,
                "rounds": [],
                "current_round": 0
            }
            
            # Store initial data
            self.metadata_store.put(self.sched_namespace, "clarification_data", clarification_data)
            
            # Run clarification rounds
            for round_num in range(self.max_clarification_rounds):
                # Generate questions for this round
                questions = self._generate_clarification_questions(
                    original_question, 
                    clarification_data["rounds"]
                )
                
                if not questions:
                    break
                
                # Ask questions and get responses
                responses = self._ask_clarification_questions(questions)
                
                # Store responses
                clarification_data["rounds"].append(responses)
                clarification_data["current_round"] = round_num + 1
                self.metadata_store.put(self.sched_namespace, "clarification_data", clarification_data)
                
                # Check if we need another round
                if self._is_clarification_sufficient(clarification_data):
                    break
            
            # Build final message with clarified context
            clarified_context = self._build_clarified_context(clarification_data)
            
            return {
                "messages": [
                    HumanMessage(content=clarified_context, name=f"{self.node_config.name}_graph")
                ],
                "prev_agent": self.node_config.name,
                "remaining_steps_display": state["remaining_steps"],
            }
        
        return call_subgraph

    def _create_clarification_node(self):
        """Create the clarification node logic."""
        def Node(state: self.State):
            # This is a placeholder that gets invoked by the subgraph
            return {"messages": [], "prev_agent": self.node_config.name}
        
        return Node

    def _generate_clarification_questions(self, original_question, previous_rounds):
        """Generate clarification questions based on the original question and previous responses."""
        # Build context from previous rounds
        context = f"Original question: {original_question}\n"
        
        if previous_rounds:
            context += "\nPrevious clarifications:\n"
            for i, round_data in enumerate(previous_rounds):
                context += f"Round {i+1}:\n"
                for q, a in round_data.items():
                    if a and a.strip():
                        context += f"Q: {q}\nA: {a}\n"
        
        # Use LLM to generate questions
        system_prompt = """You are a research assistant helping to clarify ambiguous research questions. 
        Generate 4-5 concise clarification questions to better understand the user's research goals.
        
        Focus on:
        1. Baseline: Any existing solutions or baselines the user has
        2. Instructions: Specific constraints or requirements for the code/method
        3. Objectives: What specific metrics or outcomes are expected
        4. Results: Where results should be logged and how to interpret them
        5. Assumptions: Any implicit context or constraints
        
        Format your response as a JSON array of questions.
        Example: ["What baseline methods do you want to compare against?", "What is your target accuracy?"]
        
        Keep questions concise and specific. If the context already provides clear information for a category, skip it.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]
        
        try:
            response = model.query_model_safe(messages)
            questions = json.loads(response.content)
            # Limit to 5 questions
            return questions[:self.questions_per_round]
        except:
            # Fallback questions if LLM fails
            return [
                "Do you have any baseline solutions or existing code to compare against?",
                "What specific metrics or outcomes are you looking to optimize?",
                "Are there any specific constraints or requirements for the implementation?",
                "Where should the results be logged, and what format do you prefer?",
                "Are there any assumptions or context we should be aware of?"
            ]

    def _ask_clarification_questions(self, questions):
        """Present questions to user and collect responses."""
        # ANSI escape codes for colors
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        
        print(f"\n{BOLD}{CYAN}🔍 Clarification Questions{RESET}")
        print(f"{YELLOW}Please answer the following questions to help us better understand your research goals.")
        print(f"Press Enter to skip a question if not applicable.{RESET}\n")
        
        responses = {}
        for i, question in enumerate(questions, 1):
            print(f"{GREEN}{i}. {question}{RESET}")
            response = input(f"{CYAN}Your answer: {RESET}")
            responses[question] = response
            print()  # Add spacing between questions
        
        return responses

    def _is_clarification_sufficient(self, clarification_data):
        """Determine if we have enough clarification to proceed."""
        # For now, we always proceed after getting responses
        # Could be enhanced to check if critical information is still missing
        return True


class ClarificationRouter(BaseNode):
    """Router node that processes clarification responses and determines next steps."""

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)
        self.create_transition_objs()

    def create_transition_objs(self):
        intro_message = "Processing clarification responses and updating research context.\n"
        self.node_config.transition_objs["route_to_next"] = lambda next_agent, enriched_question: {
            "messages": intro_message + enriched_question,
            "next_agent": next_agent
        }

    def create_subgraph(self):
        """Creates a Node subgraph specific to ClarificationRouter. Override BaseNode implementation."""
        subgraph_builder = StateGraph(self.State)
        router_node = self._create_router_node()

        subgraph_builder.add_node(self.node_config.name, router_node)
        subgraph_builder.add_edge(START, self.node_config.name)

        subgraph = subgraph_builder.compile(checkpointer=self.memory)
        
        def call_subgraph(state: self.State) -> self.State:
            response = subgraph.invoke({
                    "messages": state["messages"][-1]
                },
                {
                    "configurable": {
                        "thread_id": f"{self.node_config.name}_graph_id"
                    }
                }
            )
            
            return {
                "messages": [
                    HumanMessage(content=response["messages"][-1].content, name=f"{self.node_config.name}_graph")
                ],
                "prev_agent": response["prev_agent"],
                "remaining_steps_display": state["remaining_steps"],
            }
        
        return call_subgraph

    def _create_router_node(self):
        """Create the router node logic without LLM."""
        def Node(state: self.State):
            if state["remaining_steps"] <= settings.CONCLUDER_BUFFER_STEPS:
                return {
                    "messages": [], 
                    "prev_agent": self.node_config.name,
                }

            # Get clarification data
            clarification_data = self.metadata_store.get(self.sched_namespace, "clarification_data").dict()["value"]
            
            # Build enriched question with clarifications
            enriched_question = self._build_enriched_question(clarification_data)
            
            # Store enriched question for later use
            self.metadata_store.put(self.sched_namespace, "enriched_question", enriched_question)
            
            self.curie_logger.info(f"<><><><><> {self.node_config.node_icon} {self.node_config.name.upper()} {self.node_config.node_icon} <><><><><>")
            self.curie_logger.info(f"Enriched question created with clarifications")

            return {"messages": [HumanMessage(content="Clarification processing complete")], "prev_agent": self.node_config.name}
        
        return Node

    def transition_handle_func(self, state):
        """Route to the appropriate next node based on configuration."""
        self.curie_logger.info("------------ Handle Clarification Router ------------")
        
        # Get clarification data
        clarification_data = self.metadata_store.get(self.sched_namespace, "clarification_data").dict()["value"]
        
        # Build enriched question with clarifications
        enriched_question = self._build_enriched_question(clarification_data)
        
        # Store enriched question for later use
        self.metadata_store.put(self.sched_namespace, "enriched_question", enriched_question)
        
        # Determine next agent based on config
        with open(self.node_config.config_filename, 'r') as file:
            config = json.load(file)
        
        # Route to data analyzer if dataset is provided, otherwise to architect
        if config.get('dataset_dir', ''):
            next_agent = "data_analyzer"
        else:
            next_agent = "supervisor"
        
        return self.node_config.transition_objs["route_to_next"](next_agent, enriched_question)

    def _build_enriched_question(self, clarification_data):
        """Build an enriched question that includes all clarification details."""
        original = clarification_data.get("original_question", "")
        rounds = clarification_data.get("rounds", [])
        
        enriched = f"Research Question: {original}\n\n"
        
        # Add clarification details
        clarifications = []
        for round_data in rounds:
            for q, a in round_data.items():
                if a and a.strip():
                    clarifications.append(f"- {q}\n  Answer: {a}")
        
        if clarifications:
            enriched += "Additional Context from Clarification:\n"
            enriched += "\n".join(clarifications)
            enriched += "\n"
        
        return enriched 