from nodes.base_node import BaseNode
from langgraph.graph import END

class Architect(BaseNode):

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)  # Call parent class's __init__
        self.create_transition_objs()

    def create_transition_objs(self):
        self.node_config.transition_objs["no_plan"] = {"messages": self.sched_node.get_question(), "next_agent": "supervisor", "prev_agent": "supervisor"}

        self.node_config.transition_objs["is_terminate"] = {"next_agent": END}

        self.node_config.transition_objs["control_has_work"] = lambda assignment_messages: {
            "control_work": {"messages": assignment_messages, "next_agent": "control_worker"},
            "experimental_work": {"messages": [], "next_agent": "control_worker"}
        }

        self.node_config.transition_objs["experimental_has_work"] = lambda assignment_messages: return {
            "control_work": {"messages": assignment_messages, "next_agent": "control_worker"},
            "experimental_work": {"messages": self.sched_node.assign_worker("experimental"), "next_agent": "worker"}
        }

    def transition_handle_func(self):
        """
        All of the plans that were edited/written by the supervisor will be scheduled for execution now or added to the queue. 
        If the plan existed before, we check if the experimental groups have been modified: 
            if yes, we will preempt the existing group (TODO: not implemented yet. this requires that we for instance also record the old plan to know if it has changed..) from the current worker, 
            or modify within the queue. 
        If the plan is new, 
            we simply add to worker if some are idle, 
            or add to queue.
        """
        self.curie_logger.info("------------ Handle Supervisor ------------")

        # Zero, if no plan exists at all, we need to re-prompt the architect to force it to create a plan:
        if self.sched_node.is_no_plan_exists():
            return self.node_config.transition_objs["no_plan"]

        # First, check for exp termination condition:
        is_terminate = self.sched_node.check_exp_termination_condition()
        if is_terminate:
            return self.node_config.transition_objs["is_terminate"]

        # Second, for control groups that are done, move their experimental groups (if any exist) to the worker queue:
        self.curie_logger.info("Checking control group done..")
        memory_id = str("standby_exp_plan_list")
        standby_exp_plan_list = self.metadata_store.get(self.sched_node.sched_namespace, memory_id).dict()["value"]
        for plan_id in standby_exp_plan_list[:]: # iterate over a copy of standby_exp_plan_list
            self.sched_node.update_queues(plan_id, assert_new_control=True)

        # Third, for plans that were written/modified by the supervisor, add them to the correct queue:
        self.curie_logger.info("Checking supervisor wrote list..")
        memory_id = str("supervisor_wrote_list")
        supervisor_wrote_list = self.metadata_store.get(self.sched_node.sched_namespace, memory_id).dict()["value"]
        # Create a new /workspace dir for each new plan_id, and other related inits for a new plan:
        self.sched_node.init_new_plan(supervisor_wrote_list)

        # Update queues:
        for _ in range(len(supervisor_wrote_list)):
            # NOTE: currently, we ignore plan groups that are already executing in the worker.. 
            plan_id = supervisor_wrote_list.pop(0)
            self.sched_node.update_queues(plan_id) 
        # Reset supervisor_wrote_list:
        assert len(supervisor_wrote_list) == 0
        self.metadata_store.put(self.sched_node.sched_namespace, memory_id, supervisor_wrote_list)

        # Fourth, for partition's that need to be redone, add them to the correct queue: the difference here is that an explicit error feedback is provided by the supervisor.
        self.curie_logger.info("Checking supervisor redo partition list..")
        memory_id = str("supervisor_redo_partition_list")
        supervisor_redo_partition_list = self.metadata_store.get(self.sched_node.sched_namespace, memory_id).dict()["value"]
        for _ in range(len(supervisor_redo_partition_list)):
            redo_details = supervisor_redo_partition_list.pop(0)
            self.sched_node.update_queues(redo_details["plan_id"], redo_details=redo_details)
        # Reset supervisor_redo_partition_list:
        assert len(supervisor_redo_partition_list) == 0
        self.metadata_store.put(self.sched_node.sched_namespace, memory_id, supervisor_redo_partition_list)

        # Fifth, Assign to control and experimental workers if there are idle workers: according to priority
        # TODO: currently since we have NOT implemented ASYNC, we will only run 1 worker at a time (i.e., either control or normal worker). Note that we also only have 1 control and 1 normal worker only. Since there is no async I did not implement parallelism yet. 
        assignment_messages = self.sched_node.assign_worker("control")
        if not assignment_messages: # if there is no control group to run, we will run the experimental group of some partition
            return self.node_config.transition_objs["experimental_has_work"](assignment_messages)
        else: # if there exist a control group to run, we do not run the experimental group of any partition
            return self.node_config.transition_objs["control_has_work"](assignment_messages)

        # Original version:  
        # return {
        #     "control_work": {"messages": self.assign_worker("control"), "next_agent": "control_worker"},
        #     "experimental_work": {"messages": self.assign_worker("experimental"), "next_agent": "worker"}
        # }