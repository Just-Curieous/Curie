from nodes.base_node import BaseNode, NodeConfig
from langgraph.graph import END
from scheduler import SchedNode

class Analyzer(BaseNode):

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)  # Call parent class's __init__
        self.create_transition_objs()

    def create_transition_objs(self):
        intro_message = "The following partitions have completed execution and have also been executed twice with the same independent variable inputs to check for reproducibility.\n"
        self.node_config.transition_objs["progress_not_recorded"] = lambda assignments: {
            "messages": intro_message + str(assignments), 
            "next_agent": "analyzer"
        }

        self.node_config.transition_objs["correct_and_conclude"] = lambda: {
            "messages": [], 
            "prev_agent": "analyzer", 
            "next_agent": "concluder"
        }

        intro_message2 = "The following experimental plan partitions (with plan IDs, groups, and partitions) have completed execution, each run twice with the same inputs for reproducibility. Their results were analyzed, and next-step suggestions appended. Review each suggestion to assess result validity. If incorrect, mark for redo using 'redo_exp_partition'; otherwise, leave the plan unchanged. Modify or create new plans as needed.\n"
        self.node_config.transition_objs["otherwise"] = lambda completion_messages: {
            "messages": intro_message2 + str(completion_messages), 
            "prev_agent": "analyzer", 
            "next_agent": "supervisor"
        }

    def transition_handle_func(self):
        """
            Analyzer has completed a run. 
            We will now:
            - remove the analyzer from the analyzer assignment dict. 
            - assign to concluder (conditionally).
        """
        self.curie_logger.info("------------ Handle Analyzer 📊 ------------")
        # Get plan id and partition names assigned to verifier name:
        assignments = self.sched_node.get_verifier_assignment(self.node_config.name) # format: [(plan_id1, partition_name1), (plan_id2, partition_name2), ...]

        completion_messages = [] # format: [{"plan_id": plan_id1, "partition_name": partition_name1, "is_correct": True, "verifier_log_message": "no error"}, ...]

        # Assert that all assigned partition names are now done
        has_false = False # if there exist one workflow that is considered incorrect by the verifier.
        for assignment in assignments:
            plan_id, group, partition_name = assignment["plan_id"], assignment["group"], assignment["partition_name"]
            # Check if the verifier has written to the verifier_wrote_list:
            item = self.sched_node.get_verifier_wrote_list_item(self.node_config.name, plan_id, group, partition_name)
            if item is None:
                self.curie_logger.info("Warning: Analyzer has not written plan_id {}, group {}, partition_name {} to analyzer_wrote_list yet. We will rerun analyzer.".format(plan_id, group, partition_name))
                return self.node_config.transition_objs["progress_not_recorded"](assignments)
            completion_messages.append(item)
            if not item["no_change"]:
                has_false = True

        # Remove verifier from verifier assignment dict:
        self.sched_node.unassign_verifier_all(self.node_config.name)

        # Remove from verifier_wrote_list:
        self.sched_node.remove_verifier_wrote_list_all(self.node_config.name)

        # utils.print_workspace_contents()

        # self.curie_logger.info("------------Exiting handle analyzer!!!------------")
        # NOTE: currently because I don't think divergent parallel execution is possible, we will just return to supervisor if even one workflow is considered incorrect (even though there may be others that are correct which we can in principle forward to the exec_verifier)
        # Inform supervisor that verifier has completed a run:
        is_terminate = self.sched_node.check_exp_termination_condition()
        if not has_false and is_terminate: # go to concluder -> supervisor 
            return self.node_config.transition_objs["correct_and_conclude"]()
        else:
            return self.node_config.transition_objs["otherwise"](completion_messages)