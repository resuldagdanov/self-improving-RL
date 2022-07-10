import os
import sys

from ray.tune.logger import pretty_print

sys.path.append(os.path.join(os.environ["BLACK_BOX"]))

from experiments.utils import training_utils


if __name__ == "__main__":

    # organize parameters
    env, general_config, train_config = training_utils.initialize_config(
        env_config_path="/env_config.yaml",
        model_config_path="/ppo_config.yaml",
        train_config_path="/train_config.yaml"
    )

    # define PPO agent trainer
    trainer = training_utils.ppo_model_initialize(
        general_config=general_config
    )

    # load defined checkpoint for evalution
    if train_config["train_or_eval"] is False:
        checkpoint_num = train_config["checkpoint_number"]

        agent_path = os.path.join(training_utils.repo_path, "results/trained_models/" + train_config["load_agent_name"])
        print("\n[INFO]-> Agent Path:\t", agent_path)

        checkpoint_path = agent_path + "/checkpoint_%06i"%(checkpoint_num) + "/checkpoint-" + str(checkpoint_num)
        trainer.restore(checkpoint_path)
        print("\n[INFO]-> Restore Checkpoint:\t", checkpoint_path)

        # evaluate for one iteration
        eval_results = trainer.evaluate()
        print("\n[INFO]-> Evaluation Results:\t", pretty_print(eval_results))
    
    # default training loop
    else:
        for _ in range(train_config["stop"]["training_iteration"]):
            # perform one iteration of training the policy
            result = trainer.train()
            print("\n[INFO]-> Training Results:\t", pretty_print(result))

            checkpoint = trainer.save()
            print("\n[INFO]-> Checkpoint Saved:\t", checkpoint)

            print("\n\n-------------------------------------------------------------------------------------------")
