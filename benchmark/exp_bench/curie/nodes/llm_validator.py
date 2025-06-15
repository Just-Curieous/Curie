from nodes.base_node import BaseNode, NodeConfig
from nodes.exec_validator import exec_validator
from langgraph.graph import END
from scheduler import SchedNode

class LLMValidator(BaseNode):

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)  # Call parent class's __init__
        self.create_transition_objs()

    def create_transition_objs(self):
        self.node_config.transition_objs["progress_not_recorded"] = lambda assignments: {
            "messages": assignments, 
            "next_agent": "llm_verifier"
        }

        self.node_config.transition_objs["has_false"] = lambda completion_messages: {
            "messages": completion_messages, 
            "prev_agent": "llm_verifier", 
            "next_agent": "patch_verifier"
        }

        intro_message = "The following partitions have completed execution.\n"
        # intro_message = "The following partitions need to be executed to generate real results.\n"
        self.node_config.transition_objs["after_exec_verifier"] = lambda completion_messages: {
            "messages": intro_message + str(completion_messages), 
            "prev_agent": "exec_verifier", 
            "next_agent": "analyzer"
        }
    
    def remove_verifier_from_assignment(self):
        """
            Remove verifier from verifier assignment dict.
            This is called when the verifier has completed a run.
        """ 
        self.sched_node.unassign_verifier_all(self.node_config.name) 
        self.sched_node.remove_verifier_wrote_list_all(self.node_config.name)


    def transition_handle_func(self):
        """
            LLM verifier has completed a run. 
            We will now:
            - remove the verifier from the verifier assignment dict. 
            - return information back to supervisor.
        """
        self.curie_logger.info(f"------------ Handle LLM Verifier ------------")
        # Get plan id and partition names assigned to verifier name:
        assignments = self.sched_node.get_verifier_assignment(self.node_config.name) # format: [(plan_id1, partition_name1), (plan_id2, partition_name2), ...]

        completion_messages = [] # format: [{"plan_id": plan_id1, "partition_name": partition_name1, "is_correct": True, "verifier_log_message": "no error"}, ...]

        # Assert that all assigned partition names are now done
        has_false = False # if there exist one workflow that is considered incorrect by the verifier.
        has_not_recorded = False # if there exist one workflow that is not recorded by the verifier.
        for assignment in assignments:
            plan_id, group, partition_name = assignment["plan_id"], assignment["group"], assignment["partition_name"]
            # Check if the verifier has written to the verifier_wrote_list:
            item = self.sched_node.get_verifier_wrote_list_item(self.node_config.name, plan_id, group, partition_name)
            if item is None:
                has_not_recorded = True
                self.curie_logger.info("Warning: LLM verifier has not written plan_id {}, group {}, partition_name {} to verifier_wrote_list yet. We will rerun LLM verifier.".format(plan_id, group, partition_name))
                # return self.node_config.transition_objs["progress_not_recorded"](assignments)
            else:
                self.curie_logger.info("👮‍♀️ LLM verifier has written plan_id {}, group {}, partition_name {} to verifier_wrote_list: {}.".format(plan_id, group, partition_name, item))
                completion_messages.append(item)
                if not item["is_correct"]:
                    has_false = True

        # utils.print_workspace_contents()
        # NOTE: currently because I don't think divergent parallel execution is possible, we will just return to supervisor if even one workflow is considered incorrect (even though there may be others that are correct which we can in principle forward to the exec_verifier)
        # Inform supervisor that verifier has completed a run:
        if has_false: # go to patch verifier
            # Pass all assignments to patch_verifier:
            self.remove_verifier_from_assignment()
            for task_details in completion_messages:
                self.sched_node.assign_verifier("patch_verifier", task_details)
            return self.node_config.transition_objs["has_false"](completion_messages)
        # elif has_not_recorded: 
        #     self.curie_logger.info("Warning: LLM verifier has not written plan_id / group / partition_name / to verifier_wrote_list yet. We will rerun LLM verifier.")
        #     return self.node_config.transition_objs["progress_not_recorded"](assignments)
        else: # go to exec verifier -> supervisor 
            self.remove_verifier_from_assignment()
            for item in completion_messages:
                item["control_experiment_results_filename"] = self.sched_node.get_control_experiment_results_filename(item["plan_id"], item["group"], item["partition_name"])
            completion_messages = exec_validator(completion_messages)
            self.sched_node.write_all_control_experiment_results_filenames(completion_messages)
            for task_details in completion_messages:
                self.sched_node.assign_verifier("analyzer", task_details)
            return self.node_config.transition_objs["after_exec_verifier"](completion_messages)