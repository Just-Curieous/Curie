from pydantic import BaseModel, Field, Extra, model_validator
from typing import List, Dict, Any, Union, Optional
from typing_extensions import Self
import re

class NewExperimentalPlanResponseFormatter(BaseModel):
    # TODO: Currently we ignore (default behaviour, see 2nd link) extra fields (i.e., our partitions, and done status fields, etc.), but ideally we want to validate them too, e.g., using a separate function for pattern matching. 
    # UPDATE2: we have fixed this. UPDATE: currently, we allow additional fields, my previous understanding of ignore is incorrect, ignore means those fields will be silently removed from the input argument. I'm thinking of defining a separate formatter for subsequent writes to the plan, in the future, allows for better control since I think I've seen models hallucinate. 
    # References for the above: https://stackoverflow.com/questions/71837398/pydantic-validations-for-extra-fields-that-not-defined-in-schema https://stackoverflow.com/questions/69617489/can-i-get-incoming-extra-fields-from-pydantic
    # TODO: adapt to other task.
    hypothesis: str = Field(
        ...,
        description="A hypothesis to be tested. Example: AWS EC2 VMs in us-east-1 are slower than those in us-east-2."
    )
    constant_vars: List[str] = Field(
        ...,
        description="A list of variables that remain constant during the experiment. Example: ['var1', 'var2']"
    )
    independent_vars: List[str] = Field(
        ...,
        description="A list of variables that are intentionally changed to observe their effect. Example: ['AWS region']"
    )
    dependent_vars: List[str] = Field(
        ...,
        description="A list of variables being measured. Example: ['execution_time']"
    )
    controlled_experiment_setup_description: str = Field(
        ...,
        description="A high-level description of how the experiment will be conducted. Example: 'Create EC2 VM, run task.'"
    )
    control_group: List[Dict[str, Union[str, bool, int]]] = Field(
        ...,
        description="A dictionary representing the control group. Example: [{'region': 'us-east-1', 'instance_type': 't2.micro'}]"
    )
    experimental_group: Optional[List[Dict[str, Union[str, bool, int]]]] = Field(
        [],
        description="A dictionary representing the experimental group. Example: [{'region': 'us-east-2', 'instance_type': 't2.micro'}, {'region': 'us-west-2', 'instance_type': 't2.micro'}]"
    )
    priority: int = Field(
        ..., gt=0,
        description="An integer representing the priority of the experiment. Lower values indicate higher priority."
    )

    @model_validator(mode="after")
    def control_group_has_vals(self) -> Self:
        # print("Entering custom model validator: control_group_has_vals")
        if len(self.control_group) == 0:
            raise ValueError("control_group must have values.")
        return self
    
    # # TODO: we only support one partition in control group for now, but will extend it later. 
    # @model_validator(mode="after")
    # def control_group_has_one_vals(self) -> Self:
    #     # print("Entering custom model validator: control_group_has_one_vals")
    #     if len(self.control_group) != 1:
    #         raise ValueError("control_group must only have one value, put everything else in experimental_group.")
    #     return self

    # @model_validator(mode="after")
    # def exp_group_has_vals(self) -> Self:
    #     # print("Entering custom model validator: experimental_group_has_vals")
    #     if len(self.experimental_group) == 0:
    #         raise ValueError("experimental_group must have values.")
    #     return self

class ExistingExperimentalPlanResponseFormatter(BaseModel):
    # TODO: Currently we ignore (default behaviour, see 2nd link) extra fields (i.e., our partitions, and done status fields, etc.), but ideally we want to validate them too, e.g., using a separate function for pattern matching. 
    # UPDATE2: we have fixed this. UPDATE: currently, we allow additional fields, my previous understanding of ignore is incorrect, ignore means those fields will be silently removed from the input argument. I'm thinking of defining a separate formatter for subsequent writes to the plan, in the future, allows for better control since I think I've seen models hallucinate. 
    # References for the above: https://stackoverflow.com/questions/71837398/pydantic-validations-for-extra-fields-that-not-defined-in-schema https://stackoverflow.com/questions/69617489/can-i-get-incoming-extra-fields-from-pydantic

    plan_id : str = Field(
        ...,
        description="The ID of the experimental plan, used for retrieval from long-term store."
    )

    question: str = Field(
        ...,
        description="The question that the experiment is trying to answer. Example: 'Are AWS EC2 VMs in us-east-1 slower than those in us-east-2?'"
    )

    hypothesis: str = Field(
        ...,
        description="A hypothesis to be tested. Example: AWS EC2 VMs in us-east-1 are slower than those in us-east-2."
    )
    constant_vars: List[str] = Field(
        ...,
        description="A list of variables that remain constant during the experiment. Example: ['var1', 'var2']"
    )
    independent_vars: List[str] = Field(
        ...,
        description="A list of variables that are intentionally changed to observe their effect. Example: ['AWS region']"
    )
    dependent_vars: List[str] = Field(
        ...,
        description="A list of variables being measured. Example: ['execution_time']"
    )
    controlled_experiment_setup_description: str = Field(
        ...,
        description="A high-level description of how the experiment will be conducted, provided by the supervisor. Example: 'Create EC2 VM, run task.'"
    )
    control_group: Dict[str, Dict[str, Any]] = Field(
        ...,
        description='''
A dictionary representing the control group. Example: 
{
    "partition_1": {
        "independent_vars": [{
            "region": "us-east-1",
            "instance_type": "t2.micro"
        }],
        "control_experiment_filename": "/workspace/control_experiment_<plan_id>.sh",
        "control_experiment_results_filename": "/workspace/results_<plan_id>_control_group.txt",
        "done": False, 
        # "error_feedback": "", 
    }
}
'''
    )
    experimental_group: Dict[str, Dict[str, Any]] = Field(
        ...,
        description='''
A dictionary representing the experimental group. Example: 
{
    "partition_1": {
        "independent_vars": [{
            "region": "us-east-1",
            "instance_type": "t2.micro"
        }],
        "control_experiment_filename": "/workspace/control_experiment_<plan_id>_experimental_group_partition_<partition_number>.sh",
        "control_experiment_results_filename": "/workspace/results_<plan_id>_experimental_group_partition_<partition_number>.txt",
        "done": False, 
        # "error_feedback": "", 
    },
    "partition_2": {
        "independent_vars": [{
            "region": "us-west-2",
            "instance_type": "t2.micro"
        }],
        "control_experiment_filename": "/workspace/control_experiment_<plan_id>_experimental_group_partition_<partition_number>.sh",
        "control_experiment_results_filename": "/workspace/results_<plan_id>_experimental_group_partition_<partition_number>.txt",
        "done": False, 
        # "error_feedback": "", 
    }
}
'''
    )
    priority: int = Field(
        ..., gt=0,
        description="An integer representing the priority of the experiment. Lower values indicate higher priority."
    )

    # https://docs.pydantic.dev/latest/examples/custom_validators/#validating-nested-model-fields
    # https://medium.com/@marcnealer/a-practical-guide-to-using-pydantic-8aafa7feebf6
    @model_validator(mode="after")
    def groups_first_level_keys_are_partitions(self) -> Self:
        print("Entering custom model validator: groups_first_level_keys_are_partitions")
        for key, value in self.control_group:
            # Check that key follows the pattern "partition_<number>" using regex:
            if not re.match(r'partition_\d+$', key):
                raise ValueError(f"Key {key} in control_group does not follow the pattern 'partition_<number>'.")
        for key, value in self.experimental_group:
            # Check that key follows the pattern "partition_<number>" using regex:
            if not re.match(r'partition_\d+$', key):
                raise ValueError(f"Key {key} in experimental_group does not follow the pattern 'partition_<number>'.")
        return self

    @model_validator(mode="after")
    def independent_vars_is_list(self) -> Self:
        print("Entering custom model validator: independent_vars_is_list")
        for partition, value in self.control_group:
            if not isinstance(value["independent_vars"], list):
                raise ValueError("independent_vars must be a list.")
        for partition, value in self.experimental_group:
            if not isinstance(value["independent_vars"], list):
                raise ValueError("independent_vars must be a list.")
        return self

    @model_validator(mode="after")
    def required_partition_keys_exist(self) -> Self:
        print("Entering custom model validator: required_partition_keys_exist")
        for partition, value in self.control_group:
            if "independent_vars" not in value:
                raise ValueError("independent_vars key is required.")
            if "control_experiment_filename" not in value:
                raise ValueError("control_experiment_filename key is required.")
            if "control_experiment_results_filename" not in value:
                raise ValueError("control_experiment_results_filename key is required.")
            if "all_control_experiment_results_filename" not in value:
                raise ValueError("all_control_experiment_results_filename key is required.")
            if "done" not in value:
                raise ValueError("done key is required.")
            # if "error_feedback" not in value:
            #     raise ValueError("error_feedback key is required.")
        for partition, value in self.experimental_group:
            if "independent_vars" not in value:
                raise ValueError("independent_vars key is required.")
            if "control_experiment_filename" not in value:
                raise ValueError("control_experiment_filename key is required.")
            if "control_experiment_results_filename" not in value:
                raise ValueError("control_experiment_results_filename key is required.")
            if "all_control_experiment_results_filename" not in value:
                raise ValueError("all_control_experiment_results_filename key is required.")
            if "done" not in value:
                raise ValueError("done key is required.")
            # if "error_feedback" not in value:
            #     raise ValueError("error_feedback key is required.")
        return self