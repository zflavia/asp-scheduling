import torch
import torch.nn as nn
from torch_geometric.nn import  GATv2Conv, Linear, to_hetero
from torch_geometric.data import HeteroData
from torch.distributions import Categorical
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import random
from src.utils.logger import Logger
import numpy as np
import pickle

import copy

device = torch.device('cpu')

if(torch.cuda.is_available()) and random.random()<1:
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    #print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    pass
    #print("Device set to : cpu")


class RolloutBuffer:
    """
    Handles episode data collection and batch generation

    :param buffer_size: Buffer size
    :param batch_size: Size for batches to be generated

    """
    def __init__(self, buffer_size: int, batch_size: int):

        self.states = []
        self.logprobs= []
        self.actions = []
        self.rewards = []
        self.dones = []

        if buffer_size % batch_size != 0:
            raise TypeError("rollout_steps has to be a multiple of batch_size")
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.reset()

    def store_memory(self, state, action: int, prob: float,
                     reward, done: bool) -> None:
        """
        Appends all data from the recent step

        :param state: at beginning of the step
        :param action: Index of the selected action
        :param prob: Probability of the selected action
        :param reward: Reward the env returned in this step
        :param done: True if the episode ended in this step

        :return: None

        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def reset(self) -> None:
        """
        Resets all buffer lists

        :return: None

        """
        self.states = []
        self.logprobs= []
        self.actions = []
        self.rewards = []
        self.dones = []


class GAT(torch.nn.Module):
    def __init__(self, hidden_layers = 8, out_channels = 8, num_layers = 2, heads = 2):
        super().__init__()
        self.lin1 = Linear(-1, 8)
        #self.lin2 = Linear(-1, out_channels)
        self.s = torch.nn.Softmax(dim=0)
        self.tanh = nn.Tanh()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GATv2Conv(-1, hidden_layers, add_self_loops=False, edge_dim=3, heads = heads) # TODO: instead of 5 should be the number of features per our edge features (3-4?)
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr_dict):
        # TODO: print x before and after call lin1
        x = self.lin1(x)
        x = self.tanh(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr_dict)
        x = self.tanh(x)
        #x = self.lin2(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, hidden_layers, out_channels, metadata, actor, num_layers, heads):
        super().__init__()
        self.actor = actor
        self.gnn = GAT(hidden_layers, out_channels, num_layers=num_layers, heads=heads)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')
        self.lin3 = Linear(-1, 1)
        self.lin4 = Linear(-1, hidden_layers)
        self.lin5 = Linear(-1, hidden_layers)
        self.lin6 = Linear(-1, 1)
        self.tanh = nn.Tanh()

    def forward(self, data: HeteroData):
        res = self.gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        if self.actor:
            x_src, x_dst = res['machine'][data.edge_index_dict[('machine','exec','operation')][0]], res['operation'][data.edge_index_dict['machine','exec','operation'][1]]
            edge_feat = torch.cat([x_src,  data.edge_attr_dict[('machine','exec','operation')], x_dst], dim=-1)
            res = self.lin3(edge_feat)
        else:
            res = self.lin6(res["operation"])
            res = self.tanh(res)

        return res


class ActorCritic(nn.Module):
    def __init__(self, metadata, hidden_layers=128, num_layers=2, heads = 3):
        super(ActorCritic, self).__init__()
        self.actor = Model(hidden_layers, 32, metadata, True, num_layers, heads)
        self.critic = Model(hidden_layers, 32, metadata, False, num_layers, heads)
        self.metadata = metadata
        self.soft = torch.nn.Softmax(dim=0)

    def forward(self, state, deterministic: bool = True):
        action_probs = self.actor(state).T[0]
        # print('self.action(state', self.actor(state) )
        # print('action_probs_shape in forward before', action_probs.shape)
        # print('action_probs in forward before', action_probs)
        action_probs[state[('machine','exec','operation')].mask] = float("-inf")
        action_probs = self.soft(action_probs)
        # print('action_probs in forward after', action_probs)


        dist = Categorical(action_probs)

        if not deterministic:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs)

        log_prob = dist.log_prob(action)

        # print('action in forward before detach', action)

        return action.detach(), log_prob.detach()


    def evaluate(self, state, action):
        action_probs = torch.Tensor([])
        res = self.actor(state)
        action_logprobs = []
        dist_entropy = []

        row, col = state[('machine', 'exec', 'operation')].edge_index
        batch_index = state["machine"].batch[row]

        for i in range(state["operation"].batch[-1]+1):
            action_probs = res[batch_index==i].T[0]
            action_probs[state[('machine','exec','operation')].mask[batch_index==i]] = float("-inf")
            action_probs = self.soft(action_probs)
            dist = Categorical(action_probs)
            action_logprobs.append(dist.log_prob(action[i]))
            dist_entropy.append(dist.entropy())

        action_logprobs = torch.stack(action_logprobs)
        dist_entropy =  torch.stack(dist_entropy)

        state_values = self.critic(state)
        state_values = global_mean_pool(state_values, state["operation"].batch)
        return action_logprobs, state_values, dist_entropy


class PPOGNN:
    """PPO Agent class"""
    def __init__(self, env, config: dict, logger: Logger = None, metadata = None):
        """
       | gamma: Discount factor for the advantage calculation
       | learning_rate: Learning rate for both, policy_net and value_net
       | gae_lambda: Smoothing parameter for the advantage calculation
       | clip_range: Limitation for the ratio between old and new policy
       | batch_size: Size of batches which were sampled from the buffer and fed into the nets during training
       | n_epochs: Number of repetitions for each training iteration
       | rollout_steps: Step interval within the update is performed. Has to be a multiple of batch_size
       """

        self.env = env
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.n_epochs = config.get('n_epochs', 0.5)
        self.rollout_steps = config.get('rollout_steps', 2048)
        self.ent_coef = config.get('ent_coef', 0.0)
        self.num_timesteps = 0
        self.n_updates = 0
        self.learning_rate = config.get('learning_rate', 0.002)
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate_actor = config.get('learning_rate_actor', self.learning_rate)
        self.learning_rate_critic = config.get('learning_rate_critic', self.learning_rate)

        self.hidden_layers = config.get('hidden_layers', 128)
        self.num_layers = config.get('num_layers', 2)
        self.heads = config.get('heads', 3)

        # self.metadata = metadata

        self.logger = logger if logger else Logger(config=config)
        self.seed = config.get('seed', None)

        # torch seed setting
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.env.seed(self.seed)

        self.rollout_buffer = RolloutBuffer(self.rollout_steps, self.batch_size)

        if metadata is None:
            self.metadata = self.env.get_metadata()
        else:
            self.metadata = metadata

        self.policy = ActorCritic(self.metadata, self.hidden_layers, self.num_layers, self.heads).to(device)

        #self.policy.actor = self.policy.actor.to(device)
        #self.policy.critic = self.policy.critic.to(device)

        self.policy_old = ActorCritic(self.metadata, self.hidden_layers, self.num_layers, self.heads).to(device)
        self.policy_old = ActorCritic(self.metadata, self.hidden_layers, self.num_layers, self.heads).to('cpu')


        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.learning_rate_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.learning_rate_critic}
        ])

        aux_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(aux_old.state_dict())
        self.policy_old.actor = self.policy_old.actor.to(device)
        self.policy_old.critic = self.policy_old.critic.to(device)

        self.MseLoss = nn.MSELoss()

    def predict(self, obs = None, state = None, deterministic: bool = True):
        with torch.no_grad():
            state = self.env.normalize_state(state)
            state = state.to(device)
            action, log_prob = self.policy_old.forward(state, deterministic)

        if deterministic:
            self.rollout_buffer.states.append(copy.deepcopy(state))
            self.rollout_buffer.actions.append(action)
            self.rollout_buffer.logprobs.append(log_prob)

        return action, state

    def train(self):
        all_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rollout_buffer.rewards), reversed(self.rollout_buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            all_rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32).to(device)
        if len(all_rewards)>1:
            all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-7)
            #all_rewards = (2*(all_rewards + 2200)/(2200 + 1e-7 )-1).float()

        # convert list to tensor
        batch_size = self.batch_size
        batches = DataLoader(self.rollout_buffer.states, batch_size=batch_size)
        all_actions = torch.squeeze(torch.stack(self.rollout_buffer.actions, dim=0)).detach().to(device)
        all_logprobs = torch.squeeze(torch.stack(self.rollout_buffer.logprobs, dim=0)).detach().to(device)

        all_losses = 0
        all_losses_cri = 0

        #print(all_rewards)
        # Optimize policy for N epochs
        i=0
        for _ in range(self.n_epochs):
            # Evaluating old actions and values
            for batch in batches:
                batch = batch.to(device)
                old_actions = all_actions[i*batch_size: (i+1)*batch_size]
                old_logprobs = all_logprobs[i*batch_size: (i+1)*batch_size]
                rewards = all_rewards[i*batch_size: (i+1)*batch_size]
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch, old_actions)
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())
                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()

                all_losses_cri += 0.5*self.MseLoss(state_values, rewards).mean()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip_range, 1+self.clip_range) * advantages
                # final loss of clipped objective PPO

                #print("-torch.min(surr1, surr2)", -torch.min(surr1, surr2))
                #print("- 0.01*dist_entropy", - 0.01*dist_entropy)
                #print("0.5*self.MseLoss(state_values, rewards)", 0.5*self.MseLoss(state_values, rewards))
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                i+=1
                all_losses+=loss.mean()


        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.n_updates += self.n_epochs

        # clear buffer
        self.rollout_buffer.reset()

        self.logger.record(
            {
                'agent_training/n_updates': self.n_updates,
                'agent_training/loss': float(all_losses),
                'agent_training/actor_loss': float(all_losses)/(i*self.n_epochs),
                'agent_training/critic_loss': float(all_losses_cri)/(i*self.n_epochs)
            }
        )
        self.logger.dump()

    def save(self, file: str) -> None:

        """
        Save model as pickle file

        :param file: Path under which the file will be saved

        :return: None

        """
        # torch.save(self.policy_old.state_dict(), checkpoint_path)

        params_dict = self.__dict__.copy()
        del params_dict['logger']
        data = {
            "params": params_dict,
            "policy_old": self.policy_old.state_dict(),
        }

        with open(f"{file}.pkl", "wb") as handle:
            pickle.dump(data, handle)

    @classmethod
    def load(cls, file: str, config: dict, logger: Logger = None):
        """
        Creates a PPO object according to the parameters saved in file.pkl

        :param file: Path and filename (without .pkl) of your saved model pickle file
        :param config: Dictionary with parameters to specify PPO attributes
        :param logger: Logger

        :return: MaskedPPO object

        """
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        # self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

        with open(f"{file}.pkl", "rb") as handle:
            data = pickle.load(handle)

        #
        env = data["params"]["env"]
        metadata = env.get_metadata()

        # create PPO object, commit necessary parameters. Update remaining parameters
        model = cls(env=env, config=config, logger=logger, metadata=metadata)
        model.__dict__.update(data["params"])

        model.policy_old.load_state_dict(data["policy_old"])
        model.policy.load_state_dict(data["policy_old"])

        return model


    def learn(self, total_instances: int, total_timesteps: int, intermediate_test=None) -> None:
        """
        Learn over n environment instances or n timesteps. Break depending on which condition is met first
        One learning iteration consists of collecting rollouts and training the networks

        :param total_instances: Instance limit
        :param total_timesteps: Timestep limit
        :param intermediate_test: (IntermediateTest) intermediate test object. Must be created before.

        """
        instances = 0

        # iterate over n episodes = the agents has n episodes to interact with the environment
        for _ in range(total_instances):
            state = self.env.reset()
            done = False
            instances += 1
            print('instance nr', instances)

            # run agent on env until done
            while not done:
                action, prob = self.predict(state=state, deterministic=True)
                new_state, reward, done, info = self.env.step(action)
                self.num_timesteps += 1
                self.rollout_buffer.rewards.append(reward)
                self.rollout_buffer.dones.append(done)

                self.rollout_buffer.store_memory(state, action, prob, reward, done)

                # call intermediate_test on_step
                if intermediate_test:
                    intermediate_test.on_step(self.num_timesteps, instances, self)

                # break learn if total_timesteps are reached
                if self.num_timesteps >= total_timesteps:
                    print("total_timesteps reached")
                    self.logger.record(
                        {
                            'results_on_train_dataset/instances': instances,
                            'results_on_train_dataset/num_timesteps': self.num_timesteps
                        }
                    )
                    self.logger.dump()

                    return None

                # update every n rollout_steps
                if self.num_timesteps % self.rollout_steps == 0:
                    # predict the next reward, needed for the advantage computation of the last collected step
                    with torch.no_grad():
                        _, _ = self.predict(state=new_state, deterministic=True)

                    # train networks
                    self.train()

                    # reset buffer to continue collecting rollouts
                    self.rollout_buffer.reset()

                state = new_state

                # print('state', state, state['operation', 'prec', 'operation'].edge_index)

            print('Is the schedule valid? Answer: ', self.env.is_asp_schedule_valid(False))


            if instances % len(self.env.data) == len(self.env.data) - 1:
                mean_training_reward = np.mean(self.env.episodes_rewards)
                mean_training_makespan = np.mean(self.env.episodes_makespans)
                if len(self.env.episodes_tardinesses) == 0:
                    mean_training_tardiness = 0
                else:
                    mean_training_tardiness = np.mean(self.env.episodes_tardinesses)
                self.logger.record(
                    {
                        'results_on_train_dataset/mean_reward': mean_training_reward,
                        'results_on_train_dataset/mean_makespan': mean_training_makespan,
                        'results_on_train_dataset/mean_tardiness': mean_training_tardiness
                    }
                )
                self.logger.dump()

        print("TRAINING DONE")
        self.logger.record(
            {
                'results_on_train_dataset/instances': instances,
                'results_on_train_dataset/num_timesteps': self.num_timesteps
            }
        )
        self.logger.dump()
