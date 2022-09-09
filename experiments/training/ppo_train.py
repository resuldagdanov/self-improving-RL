import os
import sys

from ray.tune.logger import pretty_print

parent_directory = os.path.join(os.environ["BLACK_BOX"])
sys.path.append(parent_directory)

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

    # load model from defined checkpoint
    if train_config["is_restore"] is True:
        checkpoint_num = train_config["checkpoint_number"]

        agent_path = os.path.join(training_utils.repo_path, "results/trained_models/" + train_config["load_agent_name"])
        print("\n[INFO]-> Agent Path:\t", agent_path)

        checkpoint_path = agent_path + "/checkpoint_%06i"%(checkpoint_num) + "/checkpoint-" + str(checkpoint_num)
        trainer.restore(checkpoint_path)
        print("\n[INFO]-> Restore Checkpoint:\t", checkpoint_path)
    
    best_reward = 0.0

    # default training loop
    for idx in range(train_config["stop"]["training_iteration"]):
        # perform one iteration of training the policy
        result = trainer.train()
        print("\n[INFO]-> Training Results:\t", pretty_print(result))

        checkpoint = trainer.save()
        print("\n[INFO]-> Checkpoint is Saved:\t", checkpoint)

        # save best performing model to defined checkpoint
        avg_eps_reward = result["episode_reward_mean"]
        if avg_eps_reward > best_reward:
            best_reward = avg_eps_reward
            print("\n[INFO]-> Best Average Reward:\t", best_reward)

            model_path = "/" + "/".join(checkpoint.split("/")[1:-1])
            checkpoint_folder = model_path + "/best_checkpoints"
            if os.path.exists(checkpoint_folder) is False:
                os.makedirs(checkpoint_folder)
            
            best_checkpoint = trainer.save_checkpoint(checkpoint_folder)
            print("\n[INFO]-> Best Checkpoint is Saved:\t", best_checkpoint)
        
        print("\n\n-------------------------------------------------------------------------------------------")
