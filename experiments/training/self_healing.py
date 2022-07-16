import os
import sys

from ray.tune.logger import pretty_print

parent_directory = os.path.join(os.environ["BLACK_BOX"])
sys.path.append(parent_directory)

from experiments.utils import training_utils


if __name__ == "__main__":

    # organize parameters
    env, general_config, healing_config = training_utils.initialize_config(
        env_config_path="/env_config.yaml",
        model_config_path="/ppo_config.yaml",
        train_config_path="/self_healing.yaml"
    )

    # define PPO agent trainer
    trainer = training_utils.ppo_model_initialize(
        general_config=general_config
    )

    # load model from defined checkpoint
    if healing_config["is_restore"] is True:
        checkpoint_num = healing_config["checkpoint_number"]

        agent_path = os.path.join(training_utils.repo_path, "results/trained_models/" + healing_config["load_agent_name"])
        print("\n[INFO]-> Agent Path:\t", agent_path)

        checkpoint_path = agent_path + "/checkpoint_%06i"%(checkpoint_num) + "/checkpoint-" + str(checkpoint_num)
        trainer.restore(checkpoint_path)
        print("\n[INFO]-> Restore Checkpoint:\t", checkpoint_path)

    # default training loop
    for _ in range(healing_config["stop"]["training_iteration"]):
        # perform one iteration of training the policy
        result = trainer.train()
        print("\n[INFO]-> Training Results:\t", pretty_print(result))

        checkpoint = trainer.save()
        print("\n[INFO]-> Checkpoint Saved:\t", checkpoint)

        print("\n\n-------------------------------------------------------------------------------------------")
