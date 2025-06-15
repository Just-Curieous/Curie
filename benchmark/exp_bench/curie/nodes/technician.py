from nodes.base_node import BaseNode, NodeConfig
from langgraph.graph import END
from scheduler import SchedNode

class Technician(BaseNode):

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)  # Call parent class's __init__
        self.create_transition_objs()

    def create_transition_objs(self):
        self.node_config.transition_objs["exp_not_done"] = lambda not_done_groups: {
            "control_work": {"messages": {}, "next_agent": "control_worker"},
            "experimental_work": {"messages": {self.node_config.name: not_done_groups}, "next_agent": "worker"}
        }

        self.node_config.transition_objs["control_not_done"] = lambda not_done_groups: {
            "control_work": {"messages": {self.node_config.name: not_done_groups}, "next_agent": "control_worker"},
            "experimental_work": {"messages": {}, "next_agent": "worker"}
        }

        self.node_config.transition_objs["llm_verifier"] = lambda completion_messages: {
            "messages": completion_messages,    
            "next_agent": "llm_verifier"
        }

    def transition_handle_func(self):
        """
            A worker has completed a run. 
            We will now:
            - remove the worker from the worker assignment dict. 
            - set the executed group to done. NOTE: update, this will be handled by the worker itself instead.
            - return information back to supervisor.
        """
        self.curie_logger.info("------------Handle worker------------")
        # Get plan id and partition names assigned to worker name:
        assignments = self.sched_node.get_worker_assignment(self.node_config.name) # format: [(plan_id1, partition_name1), (plan_id2, partition_name2), ...]

        group_type = self.sched_node.get_worker_group_type(self.node_config.name)

        completion_messages = []

        not_done_groups = []

        # Assert that all assigned partition names are now done
        for assignment in assignments:
            plan_id, group, partition_name = assignment["plan_id"], assignment["group"], assignment["partition_name"]
            plan = self.store.get(self.sched_node.plan_namespace, plan_id).dict()["value"]
            self.curie_logger.info(f'Plan ID: {plan_id}')
            
            self.curie_logger.info(f"Partition Name: {partition_name}")
            self.curie_logger.info(f"Plan details: {plan}")
            # if plan_id not in completion_messages:
            #     completion_messages[plan_id] = plan
            if plan[group][partition_name]["done"] != True:
                not_done_groups.append((plan_id, group, partition_name))

        if not_done_groups: # we will reexecute the groups that are not done, by the remaining groups to the same worker
            # Determine the worker type:
            if group_type == "experimental":
                return self.node_config.transition_objs["exp_not_done"](not_done_groups)
            elif group_type == "control":
                return self.node_config.transition_objs["control_not_done"](not_done_groups)
            assert False # should not reach here
            
        # Remove worker from worker assignment dict:
        self.sched_node.unassign_worker_all(self.node_config.name) # NOTE: a worker will only return to the supervisor once all its groups are done. 

        # Pass all assignments to llm_verifier:
        for assignment in assignments:
            plan_id, group, partition_name = assignment["plan_id"], assignment["group"], assignment["partition_name"]
            plan = self.store.get(self.sched_node.plan_namespace, plan_id).dict()["value"]
            task_details = {
                "plan_id": plan_id,
                "group": group,
                "partition_name": partition_name,
                "workspace_dir": self.sched_node.get_workspace_dirname(plan_id),
                "control_experiment_filename": self.sched_node.get_control_experiment_filename(plan_id, group, partition_name),
                "control_experiment_results_filename": self.sched_node.get_control_experiment_results_filename(plan_id, group, partition_name),
            }
            completion_messages.append(task_details)
            self.sched_node.assign_verifier("llm_verifier", task_details)

        # utils.print_workspace_contents()
        # Inform supervisor that worker has completed a run:
        return self.node_config.transition_objs["llm_verifier"](completion_messages)

    # TODO: remove context if one plan is finished    
    # def _create_model_response(self, system_prompt_file):    
