
def get_log_filename_eval_gen(config, unique_id, iteration):
    log_filename = config["output_log_folder"] + f"/{str(config["paper_id"])}_{config["mode"]}_{unique_id}_iter_{iteration}_duration_{config["max_duration_per_task_in_hours"]}_log.txt"
    return log_filename

def get_log_filename_judge_gen(config, unique_id, iteration):
    log_filename = config["output_log_folder"] + f"/{str(config["paper_id"])}_{config["mode"]}_{unique_id}_iter_{iteration}_judge_log.txt"
    return log_filename

def get_relative_output_path_eval(config: dict, task_counter: int, iteration: int, mode: str):
    """
        Get the final relative output path (local, i.e., exp-bench docker or actual local) for gen (design and conclusion) or judge
    """
    if mode == "generate":
        return config["output_folder"] + f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{iteration}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen.json"
    elif mode == "judge":
        return config["output_folder"] + f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{iteration}_duration_{config["max_duration_per_task_in_hours"]}_eval_judge.json"