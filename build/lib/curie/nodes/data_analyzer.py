from nodes.base_node import BaseNode, NodeConfig
from langgraph.graph import END
from scheduler import SchedNode
from typing import Annotated
from langgraph.prebuilt import InjectedState
import json
import os

class DataAnalyzer(BaseNode):
    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)  # Call parent class's __init__
        self.create_transition_objs()
        
        with open(self.node_config.config_filename, 'r') as file:
            self.config = json.load(file)
        

    def create_transition_objs(self):
        intro_message = "Analyze the dataset thoroughly to inform the experimental design. Use the dataagent_openhands tool to perform comprehensive data analysis.\n"
        self.node_config.transition_objs["analyze_data"] = lambda: {
            "messages": intro_message + self.sched_node.get_question(), 
            "next_agent": "data_analyzer"
        }

        self.node_config.transition_objs["done"] = lambda completion_messages: {
            "messages": str(completion_messages),
            "prev_agent": "data_analyzer", 
            "next_agent": "supervisor"
        }

    def transition_handle_func(self):
        """
        DataAnalyzer will analyze the dataset and store the analysis in memory.
        After analysis is complete, it will transition to the supervisor.
        """
        self.curie_logger.info("------------ Handle Data Analyzer 🔍 ------------")
        
        # Get the dataset directory from config
        dataset_dir = self.config["dataset_dir"] if "dataset_dir" in self.config else None
        if not dataset_dir:
            self.curie_logger.info("No dataset directory specified. Skipping data analysis.")
            return self.node_config.transition_objs["done"](None)
        
        # can also get the analysis from tool response 
        log_dir = os.path.dirname(self.config["log_filename"])
        data_log_filename = f'/{log_dir}/data_analysis_results.txt'
        if os.path.exists(data_log_filename):
            with open(data_log_filename, 'r') as file:
                data_analysis = file.read()

            data_analysis = self.summarize_data_analysis(data_analysis)
            
            self.metadata_store.put(self.sched_node.sched_namespace, "data_analysis", str(data_analysis))
            return self.node_config.transition_objs["done"](str(data_analysis))

        # Call the data analysis tool
        response = self.tools[0]._run()
        self.metadata_store.put(self.sched_node.sched_namespace, "data_analysis", str(response))
        
        return self.node_config.transition_objs["done"](response)

    def summarize_data_analysis(self, data_analysis):
        """
        Summarize the data analysis.
        """
        # TODO
        return data_analysis