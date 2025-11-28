import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, Any, List
import gc

import hydra
import numpy as np
import torch
import torch.utils.data.dataloader
import tqdm
from omegaconf import OmegaConf
import aloha_insertion_env
import wandb
from video import VideoRecorder
import pickle

config_name = "train_aloha_insertion"


# Assuming seed_everything is defined elsewhere
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


MAX_ACTION_DELTA = 0.05


def vqbet_collate_fn(batch: List[Dict[str, Any]]):
    """
    Custom collate function to correctly stack the list of dictionary items
    into a single dictionary of batched tensors, with a special fix for
    oversized action targets that fail to flatten in __getitem__.
    """
    if not isinstance(batch[0], dict):
        return torch.utils.data.dataloader.default_collate(batch)

    output_dict = {}

    # 1. Stack all items normally
    for key in batch[0].keys():
        tensors = [item[key] for item in batch]

        if isinstance(tensors[0], torch.Tensor) and tensors[0].dim() >= 1:
            output_dict[key] = torch.stack(tensors)
        else:
            output_dict[key] = torch.utils.data.dataloader.default_collate(tensors)

    # 2. 🚨 CRITICAL FAIL-SAFE FIX FOR ACTION TARGET SIZE (256 vs 16)
    if "target_actions" in output_dict and "obs" in output_dict:
        act = output_dict["target_actions"]
        B_true = len(batch)  # The correct batch size (e.g., 16)

        if act.dim() == 3 and act.shape[0] != B_true:
            print(f"[COLLATE_FIX] Correcting action tensor size: Forcing B={B_true} (Target was {act.shape[0]})")
            output_dict["target_actions"] = act[:B_true, ...].clone().detach()

    return output_dict


@hydra.main(config_path="configs", config_name=config_name, version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    train_data, test_data = hydra.utils.instantiate(cfg.data)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=False,
        collate_fn=vqbet_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=False,
        collate_fn=vqbet_collate_fn
    )

    if "visual_input" in cfg and cfg.visual_input:
        print("use visual environment")
        cfg.model.gpt_model.config.input_dim = 1024

    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))

    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )

    env = hydra.utils.instantiate(cfg.env.gym)
    goal_fn = hydra.utils.instantiate(cfg.goal_fn)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = run.name or "Offline"
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)
    video = VideoRecorder(dir_name=save_path)

    @torch.no_grad()
    def eval_on_env(
            cfg,
            num_evals=cfg.num_env_evals,
            num_eval_per_goal=1,
            videorecorder=None,
            epoch=None,
    ):
        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []
        ii=0
        for goal_idx in range(num_evals):
            if videorecorder is not None:
                # Video recording is typically only enabled for the first goal (goal_idx == 0)
                videorecorder.init(enabled=True)

            for _ in range(num_eval_per_goal):
                obs_stack = deque(maxlen=cfg.eval_window_size)

                obs_data, info = env.reset()
                obs_stack.append(obs_data)

                done, step, total_reward = False, 0, 0
                goal, _ = goal_fn(env, obs_stack[-1], goal_idx, step)

                while (not done) and step<500:
                    # Stack the observations (T, C, W, H)
                    obs = torch.from_numpy(np.stack(obs_stack)).float().to(cfg.device)
                    obs = obs.squeeze(1)

                    goal = torch.as_tensor(goal, dtype=torch.float32, device=cfg.device)
                    # Model call here uses single item (unsqueeze(0))
                    action, _, _ = cbet_model(obs.unsqueeze(0), goal.unsqueeze(0), None)

                    if cfg.action_window_size > 1:
                        action_list.append(action[-1].cpu().detach().numpy())
                        if len(action_list) > cfg.action_window_size:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)
                        curr_action = (
                                np.sum(curr_action, axis=0)[0] / curr_action.shape[0]
                        )
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[1:], np.zeros((1, a_chunk.shape[1])))
                                )
                            )
                        action_list = new_action_list
                    else:
                        curr_action = action[-1, 0, :].cpu().detach().numpy()

                    curr_action = np.clip(curr_action, -MAX_ACTION_DELTA, MAX_ACTION_DELTA)

                    obs_data, reward, done, info = env.step(curr_action)

                    if videorecorder.enabled:
                        videorecorder.record(info["image"])

                    step += 1
                    total_reward += reward
                    obs_stack.append(obs_data)

                    if "pusht" not in config_name:
                        goal, _ = goal_fn(env, obs_stack[-1], goal_idx, step)
                    
                avg_reward += total_reward

                if "pusht" in config_name:
                    env.env._seed += 1
                    avg_max_coverage.append(info["max_coverage"])
                    avg_final_coverage.append(info["final_coverage"])

                if "all_completions_ids" in info:
                    completion_id_list.append(info["all_completions_ids"])

            # Save the video locally
            print(f"{epoch} rollout complete")
            video_filename = "eval_{}_{}.mp4".format(epoch, ii)
      
            videorecorder.save(video_filename)

            # 🎥 WandB Video Logging (Only logs the first goal if enabled)
            if videorecorder.enabled and wandb.run:
                video_path = Path(videorecorder.dir_name) / video_filename
                wandb.log({
                    f"videos/eval_goal_{goal_idx}": wandb.Video(
                        str(video_path),
                        caption=f"Epoch {epoch}, Goal {ii}",
                        fps=10,
                        format="mp4"
                    )
                }, step=epoch)
            ii+=1
            # ✅ MEMORY CLEANUP after each eval run
            torch.cuda.empty_cache()
            gc.collect()

        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    for epoch in tqdm.trange(cfg.epochs):
        cbet_model.eval()

        if (epoch % cfg.eval_on_env_freq == 0):
            avg_reward, completion_ids, avg_max_coverage, avg_final_coverage = eval_on_env(
                cfg, videorecorder=video, epoch=epoch
            )

            # 🚀 WandB Logging for Environment Evaluation Metrics
            log_dict = {
                "eval/avg_reward": avg_reward,
                "eval/completion_rate": len(completion_ids) / cfg.num_env_evals,
            }

            if "pusht" in config_name:
                if avg_max_coverage:
                    log_dict["eval/avg_max_coverage"] = np.mean(avg_max_coverage)
                if avg_final_coverage:
                    log_dict["eval/avg_final_coverage"] = np.mean(avg_final_coverage)

            wandb.log(log_dict, step=epoch)

            # ✅ MEMORY CLEANUP after all evaluation and before training
            del avg_reward, completion_ids, avg_max_coverage, avg_final_coverage
            torch.cuda.empty_cache()
            gc.collect()

        if epoch % cfg.eval_freq == 0 and epoch !=0 :
            total_loss_dict = {}
            num_batches = 0
            with torch.no_grad():
                for data in tqdm.tqdm(test_loader):
                    obs = data["obs"].to(cfg.device)
                    act = data["target_actions"].to(cfg.device)

                    # ⬅️ GOAL SHAPE FIX (Offline Evaluation Loop)
                    if "goals" in data:
                        goal = data["goals"].to(cfg.device)
                        if goal.dim() == 3:
                            goal = goal[:, 0, :]
                    else:
                        goal = torch.zeros(obs.shape[0], cfg.goal_dim, device=cfg.device)

                    # --- 🔥 ADDED: Validation Loss Calculation and Aggregation ---
                    predicted_act, loss, loss_dict = cbet_model(obs, goal, act)

                    for k, v in loss_dict.items():
                        if k not in total_loss_dict:
                            total_loss_dict[k] = 0.0

                        if isinstance(v, torch.Tensor):
                            total_loss_dict[k] += v.item()
                        else:
                            # If it's already a float (the cause of your error), just add it
                            total_loss_dict[k] += v
                    num_batches += 1
                    

                # Calculate and Log Average Validation Metrics
                if num_batches > 0:
                    avg_loss_dict = {
                        f"eval_offline/{k}": v / num_batches for k, v in total_loss_dict.items()
                    }
                    wandb.log(avg_loss_dict, step=epoch)
                print(f"Epoch {epoch} Eval loss {avg_loss_dict}")
            # Optional: Clean up memory after test evaluation
            torch.cuda.empty_cache()
            gc.collect()

        # --- Training Loop ---
        cbet_model.train()
        train_total_loss_dict = {}
        train_num_batches = 0

        for data in tqdm.tqdm(train_loader):
            if epoch < (cfg.epochs * 0.5):
                optimizer["optimizer1"].zero_grad()
                optimizer["optimizer2"].zero_grad()
            else:
                optimizer["optimizer2"].zero_grad()

            obs = data["obs"].to(cfg.device)
            act = data["target_actions"].to(cfg.device)

            # ⬅️ GOAL SHAPE FIX (Training Loop)
            if "goals" in data:
                goal = data["goals"].to(cfg.device)
                if goal.dim() == 3:
                    goal = goal[:, 0, :]
            else:
                goal = torch.zeros(obs.shape[0], cfg.goal_dim, device=cfg.device)

            predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
        
            for k, v in loss_dict.items():
                if k not in train_total_loss_dict:
                    train_total_loss_dict[k] = 0.0
                
                if isinstance(v, torch.Tensor):
                    train_total_loss_dict[k] += v.item()
                else:
                    train_total_loss_dict[k] += v
            
            train_num_batches += 1
            loss.backward()

            if epoch < (cfg.epochs * 0.5):
                optimizer["optimizer1"].step()
                optimizer["optimizer2"].step()
            else:
                optimizer["optimizer2"].step()
            

        if train_num_batches > 0:
            avg_train_loss_dict = {
                f"train/{k}_avg_epoch": v / train_num_batches for k, v in train_total_loss_dict.items()
            }
            # Log the epoch averages
            wandb.log(avg_train_loss_dict, step=epoch)
        print(f"Epoch {epoch} Training loss {avg_train_loss_dict}")
        if epoch % cfg.save_every == 0:
            cbet_model.save_model(save_path)

    # ... (Final Evaluation)


if __name__ == "__main__":
    main()
