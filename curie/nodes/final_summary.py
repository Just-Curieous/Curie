# Final Summary and User Check Node
from nodes.base_node import BaseNode, NodeConfig
from langgraph.graph import StateGraph, START, END
from scheduler import SchedNode
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage 

import os
import json
import glob

import settings
import model
import utils

class FinalSummary(BaseNode):
    """Node responsible for providing a final summary and checking if user needs additional assistance."""

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)
        self.create_transition_objs()

    def create_transition_objs(self):
        self.node_config.transition_objs["terminate"] = lambda: {
            "next_agent": END
        }
        
        self.node_config.transition_objs["additional_help"] = lambda help_request: {
            "next_agent": "final_assistant"
        }

    def transition_handle_func(self, state):
        """Handle the transition based on user's final needs."""
        self.curie_logger.info("------------ Handle Final Summary ------------")
        
        # Check if user requested additional help
        try:
            final_check_data = self.metadata_store.get(self.sched_namespace, "final_check_data").dict()["value"]
            if final_check_data.get("needs_help", False):
                help_request = final_check_data.get("help_request", "")
                return self.node_config.transition_objs["additional_help"](help_request)
        except:
            pass
        
        # Default to termination
        return self.node_config.transition_objs["terminate"]()

    def create_subgraph(self):
        """Creates a Node subgraph specific to FinalSummary."""
        subgraph_builder = StateGraph(self.State)
        summary_node = self._create_summary_node()

        subgraph_builder.add_node(self.node_config.name, summary_node)
        subgraph_builder.add_edge(START, self.node_config.name)

        subgraph = subgraph_builder.compile(checkpointer=self.memory)
        
        def call_subgraph(state: self.State) -> self.State:
            # Generate experiment summary
            summary = self._generate_experiment_summary()
            
            # Present summary and ask for final needs
            final_check_data = self._conduct_final_check(summary)
            
            # Store final check data
            self.metadata_store.put(self.sched_namespace, "final_check_data", final_check_data)
            
            # Mark final summary as completed in this session
            self.metadata_store.put(self.sched_namespace, "final_summary_completed", True)
            
            return {
                "messages": [
                    HumanMessage(content="Final summary completed", name=f"{self.node_config.name}_graph")
                ],
                "prev_agent": self.node_config.name,
                "remaining_steps_display": state["remaining_steps"],
            }
        
        return call_subgraph

    def _create_summary_node(self):
        """Create the summary node logic."""
        def Node(state: self.State):
            return {"messages": [], "prev_agent": self.node_config.name}
        
        return Node

    def _generate_experiment_summary(self):
        """Generate a comprehensive summary of the experiment."""
        summary_parts = []
        
        # Get all experiment plans
        items = self.store.search(self.plan_namespace)
        plans = [item.dict()["value"] for item in items]
        
        summary_parts.append("🔬 EXPERIMENT SUMMARY")
        summary_parts.append("=" * 50)
        
        # Original question
        original_question = self.sched_node.get_question()
        summary_parts.append(f"\n📝 Research Question:")
        summary_parts.append(f"{original_question}")
        
        # Plans executed
        summary_parts.append(f"\n📋 Experimental Plans Executed: {len(plans)}")
        
        # Results locations
        result_locations = []
        workspace_dirs = []
        
        for plan in plans:
            plan_id = plan.get("plan_id", "unknown")
            workspace_dir = plan.get("workspace_dir", "")
            if workspace_dir:
                workspace_dirs.append(workspace_dir)
                
            # Check for result files
            for group_type in ["control_group", "experimental_group"]:
                if group_type in plan:
                    for partition_name, partition_data in plan[group_type].items():
                        if partition_data.get("done", False):
                            result_file = partition_data.get("control_experiment_results_filename")
                            if result_file and os.path.exists(result_file):
                                result_locations.append(result_file)
        
        # Result locations summary
        summary_parts.append(f"\n📊 Results Generated:")
        if result_locations:
            for location in result_locations[:5]:  # Show first 5
                summary_parts.append(f"  • {location}")
            if len(result_locations) > 5:
                summary_parts.append(f"  • ... and {len(result_locations) - 5} more files")
        else:
            summary_parts.append("  • No result files found")
        
        # Workspace directories
        summary_parts.append(f"\n📁 Workspace Directories:")
        for workspace_dir in set(workspace_dirs):
            summary_parts.append(f"  • {workspace_dir}")
        
        # Check for final report
        with open(self.node_config.config_filename, 'r') as file:
            config = json.load(file)
        
        log_dir = os.path.dirname(config.get('log_filename', ''))
        if log_dir:
            # Look for generated reports
            report_files = glob.glob(f"{log_dir}/research_*.md")
            if report_files:
                summary_parts.append(f"\n📄 Generated Reports:")
                for report_file in report_files:
                    summary_parts.append(f"  • {report_file}")
        
        return "\n".join(summary_parts)

    def _conduct_final_check(self, summary):
        """Present summary and ask user for final needs."""
        # ANSI escape codes for colors
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        
        print(f"\n{BOLD}{BLUE}🎉 Experiment Completed!{RESET}")
        print(f"{CYAN}{summary}{RESET}")
        
        print(f"\n{BOLD}{YELLOW}Final Check - Do you need additional assistance?{RESET}")
        print(f"{GREEN}Please let us know if you need help with:{RESET}")
        print(f"  1. Understanding or interpreting the results")
        print(f"  2. Post-processing or analyzing the data")
        print(f"  3. Planning follow-up experiments")
        print(f"  4. Generating additional visualizations or reports")
        print(f"  5. Any other questions about the experiment")
        
        print(f"\n{CYAN}Press Enter to finish, or describe what additional help you need:{RESET}")
        user_response = input(f"{CYAN}Your request: {RESET}")
        
        if user_response.strip():
            return {
                "needs_help": True,
                "help_request": user_response.strip(),
                "summary": summary
            }
        else:
            return {
                "needs_help": False,
                "help_request": "",
                "summary": summary
            }


class FinalAssistant(BaseNode):
    """Node that provides additional assistance based on user's final requests."""

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)
        self.create_transition_objs()

    def create_transition_objs(self):
        self.node_config.transition_objs["help_provided"] = lambda: {
            "next_agent": END
        }

    def transition_handle_func(self, state):
        """Provide assistance and then terminate."""
        self.curie_logger.info("------------ Handle Final Assistant ------------")
        return self.node_config.transition_objs["help_provided"]()

    def create_subgraph(self):
        """Creates a Node subgraph specific to FinalAssistant. Override BaseNode implementation."""
        subgraph_builder = StateGraph(self.State)
        
        with open(self.node_config.config_filename, 'r') as file:
            config = json.load(file)
        
        # Use a specialized prompt for final assistance
        system_prompt_file = config.get("final_assistant_system_prompt_filename", "prompts/final-assistant.txt")
        assistant_node = self._create_assistant_node(system_prompt_file)

        subgraph_builder.add_node(self.node_config.name, assistant_node)
        subgraph_builder.add_edge(START, self.node_config.name)

        subgraph = subgraph_builder.compile(checkpointer=self.memory)
        
        def call_subgraph(state: self.State) -> self.State:
            # Get final check data
            try:
                final_check_data = self.metadata_store.get(self.sched_namespace, "final_check_data").dict()["value"]
                help_request = final_check_data.get("help_request", "")
                summary = final_check_data.get("summary", "")
                
                # Create message with context
                context_message = f"Experiment Summary:\n{summary}\n\nUser's Request for Help:\n{help_request}"
                
                response = subgraph.invoke({
                        "messages": [HumanMessage(content=context_message)]
                    },
                    {
                        "configurable": {
                            "thread_id": f"{self.node_config.name}_graph_id"
                        }
                    }
                )
                
                # Display the assistant's response
                print(f"\n🤖 {self.node_config.node_icon} Additional Assistance:")
                print(f"{response['messages'][-1].content}")
                
            except Exception as e:
                self.curie_logger.error(f"Error providing final assistance: {e}")
                print(f"\n❌ Sorry, there was an error providing additional assistance.")
            
            return {
                "messages": [
                    HumanMessage(content="Final assistance provided", name=f"{self.node_config.name}_graph")
                ],
                "prev_agent": self.node_config.name,
                "remaining_steps_display": state["remaining_steps"],
            }
        
        return call_subgraph

    def _create_assistant_node(self, system_prompt_file):
        """Create the assistant node with specialized prompt."""
        def Node(state: self.State):
            if state["remaining_steps"] <= settings.CONCLUDER_BUFFER_STEPS:
                return {
                    "messages": [], 
                    "prev_agent": self.node_config.name,
                }
            
            # Read from prompt file
            try:
                with open(system_prompt_file, "r") as file:
                    system_prompt = file.read()
            except FileNotFoundError:
                # Fallback prompt if file doesn't exist
                system_prompt = """You are a helpful research assistant providing final support after an experiment has completed.

Your role is to help users understand their results, suggest post-processing steps, and answer any questions about the experiment.

Be helpful, specific, and actionable in your responses. If the user asks about results interpretation, provide concrete suggestions. If they need follow-up experiments, give specific recommendations."""

            system_message = SystemMessage(content=system_prompt)
            
            messages = state["messages"]
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages.insert(0, system_message)
            
            response = model.query_model_safe(messages, tools=self.tools)
            
            self.curie_logger.info(f"<><><><><> {self.node_config.node_icon} {self.node_config.name.upper()} {self.node_config.node_icon} <><><><><>")
            self.curie_logger.info(f"Provided final assistance to user")
            
            return {"messages": [response], "prev_agent": self.node_config.name}
        
        return Node 