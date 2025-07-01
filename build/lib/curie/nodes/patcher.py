from nodes.base_node import BaseNode, NodeConfig
from nodes.exec_validator import exec_validator
from langgraph.graph import END
from scheduler import SchedNode

class Patcher(BaseNode):

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)  # Call parent class's __init__
        self.create_transition_objs()

    def create_transition_objs(self):
        self.node_config.transition_objs["progress_not_recorded"] = lambda assignments: {
            "messages": assignments, 
            "next_agent": "patch_verifier"
        }

        intro_message = "The following experimental plan workflows (containing plan IDs, group, partitions) have been attempted to be patched by a code debugging agent but failed. Please review. You may re-execute partitions where the workflow was not correct using the 'redo_exp_partition' tool. Otherwise, you can leave the plans unchanged, write new plans, or modify existing ones as needed.\n"
        self.node_config.transition_objs["has_false"] = lambda completion_messages: {
            "messages": intro_message + str(completion_messages), 
            "prev_agent": "patch_verifier", 
            "next_agent": "supervisor"
        }

        intro_message2 = "The following partitions have completed execution and have also been executed twice with the same independent variable inputs to check for reproducibility.\n"
        self.node_config.transition_objs["after_patch_verifier"] = lambda completion_messages: {
            "messages": intro_message2 + str(completion_messages),
            "prev_agent": "exec_verifier", 
            "next_agent": "analyzer"
        }

    def transition_handle_func(self):
        """
            Patch verifier has completed a run. 
            We will now:
            - remove the verifier from the verifier assignment dict. 
            - return information back to supervisor.
        """
        self.curie_logger.info(f"------------ Handle patcher {self.node_config.name} ------------")
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
                self.curie_logger.info("Warning: Patch verifier has not written plan_id {}, group {}, partition_name {} to verifier_wrote_list yet. We will rerun patch verifier.".format(plan_id, group, partition_name))
                return self.node_config.transition_objs["progress_not_recorded"](assignments)
            completion_messages.append(item)
            if not item["is_correct"]:
                has_false = True

        # Remove verifier from verifier assignment dict:
        self.sched_node.unassign_verifier_all(self.node_config.name)

        # Remove from verifier_wrote_list:
        self.sched_node.remove_verifier_wrote_list_all(self.node_config.name)

        # utils.print_workspace_contents()

        # self.curie_logger.info("------------Exiting handle patch verifier!!!------------")
        # NOTE: currently because I don't think divergent parallel execution is possible, we will just return to supervisor if even one workflow is considered incorrect (even though there may be others that are correct which we can in principle forward to the exec_verifier)
        # Inform supervisor that verifier has completed a run:
        if has_false: # go to supervisor
            return self.node_config.transition_objs["has_false"](completion_messages)
        else: # go to exec verifier -> supervisor 
            for item in completion_messages:
                item["control_experiment_results_filename"] = self.sched_node.get_control_experiment_results_filename(item["plan_id"], item["group"], item["partition_name"])
            completion_messages = exec_validator(completion_messages)
            self.sched_node.write_all_control_experiment_results_filenames(completion_messages)
            for task_details in completion_messages:
                self.sched_node.assign_verifier("analyzer", task_details)
            return self.node_config.transition_objs["after_patch_verifier"](completion_messages)