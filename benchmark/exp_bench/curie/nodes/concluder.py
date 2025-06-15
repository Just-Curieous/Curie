from nodes.base_node import BaseNode, NodeConfig
from langgraph.graph import END
from scheduler import SchedNode
from typing import Annotated, List
from langgraph.prebuilt import InjectedState
import settings

class Concluder(BaseNode):

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)  # Call parent class's __init__
        self.create_transition_objs()

    def create_transition_objs(self):
        self.node_config.transition_objs["progress_not_recorded"] = lambda: {
            "messages": [], 
            "next_agent": "concluder"
        }

        self.node_config.transition_objs["need_terminate_but_not_concluding"] = lambda: {
            "messages": [self.sched_node.get_concluder_terminate_message()], 
            "next_agent": "concluder"
        }

        self.node_config.transition_objs["terminate"] = lambda: {"next_agent": END}

        intro_message = '''
All partitions for all experimental plans have completed, with results produced and analyzed. A next-step suggestion is appended. Conclude the experiment if you believe it provides a rigorous and comprehensive answer. Report all neccessary experiment results/numbers for the conclusion. Otherwise, if results are insufficient or further questions remain, create a new experimental plan.\n
'''
        self.node_config.transition_objs["after_concluder"] = lambda item: {
            "messages": intro_message + str(item), 
            "prev_agent": "concluder", 
            "next_agent": "supervisor"
        }

    def transition_handle_func(self, state: Annotated[dict, InjectedState]):
        """
            Concluder has completed a run. 
            We will now:
            - remove the concluder from the concluder assignment dict. 
            - assign to supervisor.
        """
        self.curie_logger.info("------------ Handle Concluder 🔚 ------------")
        # Get plan id and partition names assigned to verifier name:
        # assignments = self.get_verifier_assignment(self.node_config.name) # format: [(plan_id1, partition_name1), (plan_id2, partition_name2), ...]

        # completion_messages = [] # format: [{"plan_id": plan_id1, "partition_name": partition_name1, "is_correct": True, "verifier_log_message": "no error"}, ...]

        # Assert that all assigned partition names are now done
        item = self.sched_node.get_concluder_wrote_list_item()
        if len(item) == 0:
            self.curie_logger.info("Warning: Concluder has not written to concluder_wrote_list yet. We will rerun concluder.")
            return self.node_config.transition_objs["progress_not_recorded"]()
        if len(item) > 1:
            self.curie_logger.info("Warning: Concluder has written more than one item to concluder_wrote_list. We will only use the last item written.")

        if state["remaining_steps"] <= settings.CONCLUDER_BUFFER_STEPS:
            if item[-1]["is_conclude"] != True:
                self.curie_logger.info("Warning: Concluder has not concluded the experiment yet, but we do not have enough iterations (i.e., must conclude). We will rerun concluder.")
                self.sched_node.remove_verifier_wrote_list_all(self.node_config.name)
                return self.node_config.transition_objs["need_terminate_but_not_concluding"]() # reset remaining steps to allow concluder to eventually exit. 
            else:
                return self.node_config.transition_objs["terminate"]()

        # Remove verifier from verifier assignment dict:
        self.sched_node.unassign_verifier_all(self.node_config.name)

        # Remove from verifier_wrote_list:
        self.sched_node.remove_verifier_wrote_list_all(self.node_config.name)

        # utils.print_workspace_contents()

        # self.curie_logger.info("------------Exiting handle concluder!!!------------")
        # NOTE: currently because I don't think divergent parallel execution is possible, we will just return to supervisor if even one workflow is considered incorrect (even though there may be others that are correct which we can in principle forward to the exec_verifier)
        # Inform supervisor that verifier has completed a run:
        return self.node_config.transition_objs["after_concluder"](item)