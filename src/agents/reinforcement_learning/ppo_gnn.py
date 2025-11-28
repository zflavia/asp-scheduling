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
            print("buffer_size",buffer_size, "batch_size", batch_size, buffer_size % batch_size)
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
        self.logprobs.append(prob)#FM COMENTEZA MI SE PARE CA E DUPLICATA SI MAI SCRIE CINEVA
        #print("... all logprobs", self.logprobs)
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
        #Linear(-1,out_channels) - initialized lazily in case it is given as -1
        self.lin1 = Linear(-1, 8) #in_dim - nr of feature per node
        #self.lin2 = Linear(-1, out_channels)
        self.s = torch.nn.Softmax(dim=0)
        self.tanh = nn.Tanh()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            #edge_dim - numar features pe muchie
            #!!!!in_channels=hidden_layers,
            conv = GATv2Conv(-1, hidden_layers, add_self_loops=False,
                             edge_dim=3, heads = heads) # : instead of 5 should be the number of features per our edge features (3-4?)
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr_dict):
        # TODO: print x before and after call lin1

        print("forward() - edge_index",edge_index, "edge_attr_dict", edge_attr_dict )
        x = self.lin1(x)
        x = self.tanh(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr_dict)
            #x = self.tanh(x) - fiecare strat primește input normalizat în (-1, 1)
        x = self.tanh(x) #straturile intermediare ar avea libertate mai mare, iar doar rezultatul final ar fi limitat.
        #x = self.lin2(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, hidden_layers, out_channels, metadata, actor, num_layers, heads):
        super().__init__()
        self.actor = actor
        self.gnn = GAT(hidden_layers, out_channels, num_layers=num_layers, heads=heads)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')
        self.lin3 = Linear(-1, 1) #utilizat de actor out_features=1; in_feat = edge_no (machine, exec, op)
        self.lin4 = Linear(-1, hidden_layers) # - nu sunt utilizate
        self.lin5 = Linear(-1, hidden_layers) # - nu sunt utilizate
        self.lin6 = Linear(-1, 1) #utilizat de critic out_features=1; in_feature = op_no?
        self.tanh = nn.Tanh()

    def forward(self, data: HeteroData):
        # data: HeteroData
        res = self.gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

        if self.actor:
            x_src, x_dst = (res['machine'][data.edge_index_dict[('machine','exec','operation')][0]],
                            res['operation'][data.edge_index_dict['machine','exec','operation'][1]])
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
        #scoruri brute pt fiecare actiune posibila
        action_probs = self.actor(state).T[0]
        #Maschează acțiunile nefezabile
        action_probs[state[('machine','exec','operation')].mask] = float("-inf")
        #aplica soft-max
        action_probs = self.soft(action_probs)
        #Creează o distribuție Categorical normalizată. Dacă o acțiune a primit -inf înainte de softmax → acum are probabilitatea 0.
        dist = Categorical(action_probs)

        if not deterministic:
            #ia ia o acțiune proporțional cu probabilitatea
            action = dist.sample()
        else:
            #ia acțiunea cu cea mai mare probabilitate
            action = torch.argmax(action_probs)

        #Se obține log-probabilitatea acțiunii selectate
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach()


    def evaluate(self, state, action):
        action_probs = torch.Tensor([])
        #calculează logit-urile (scorurile) pentru toate acțiunile posibile (de fapt pentru toate muchiile machine–operation).
        res = self.actor(state)
        action_logprobs = []
        dist_entropy = []

        #row = index de machine, col = index de operation
        row, col = state[('machine', 'exec', 'operation')].edge_index
        batch_index = state["machine"].batch[row]

        #state["operation"].batch[-1] - id-ul ultimului graf din bach,numărul de grafuri din batch (se adauga+1 deoarece numerotarea incepe de la 0)
        for i in range(state["operation"].batch[-1]+1):
            #logit-urile actorului pentru muchiile grafului i
            action_probs = res[batch_index==i].T[0]

            #Masca acțiunile nefezabile pentru graful respectiv
            action_probs[state[('machine','exec','operation')].mask[batch_index==i]] = float("-inf")
            action_probs = self.soft(action_probs)
            dist = Categorical(action_probs)
            action_logprobs.append(dist.log_prob(action[i]))#FM se adauga de doaua ori se seteaza si in state_save())
            dist_entropy.append(dist.entropy())

        action_logprobs = torch.stack(action_logprobs)
        dist_entropy =  torch.stack(dist_entropy)

        state_values = self.critic(state)
        state_values = global_mean_pool(state_values, state["operation"].batch)

        #utile pt calcul: policy loss (folosind log_probs și avantaj); value loss (folosind state_values); entropy bonus (folosind dist_entropy)
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
       | rollout_episodes: Step interval within the update is performed. Has to be a multiple of batch_size
       | n_episodes: episodes number of train function (it should be much larger than the number of instances in the environment
       """

        self.env = env
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.n_epochs = config.get('n_epochs', 0.5)

        self.update_strategy = config.get('update_strategy', 'buffer_size')
        self.total_instances = config.get('total_instances', 1000)
        self.total_timesteps = config.get('total_timesteps',  3000000)
        self.rollout_episodes = config.get('rollout_episodes', 5)
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

        #early stoppping
        self.best_makespan = float('inf')
        self.no_improve_counter = 0
        self.patience =  config.get('early_stoppping_pacience', 10)


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

    def predict(self, state = None, observation = None, deterministic: bool = True):
        #este functia select_action() sau act() din alte implementari
        with torch.no_grad():
            state = self.env.normalize_state(state)
            state = state.to(device)
            action, log_prob = self.policy_old.forward(state, deterministic)

        # if deterministic:
        #     self.rollout_buffer.states.append(copy.deepcopy(state))#FM era state inainte in loc de observation
        #     self.rollout_buffer.actions.append(action) #FM era state inainte in loc de action
        #     self.rollout_buffer.logprobs.append(log_prob)
        #     print("predict() self.rollout_buffer.logprobs", log_prob, self.rollout_buffer.logprobs)

        return action, log_prob #FM-intoarce  log_prob in loc de state

    def train(self):
        #functia update() - din echeveria (implementarea buclei de update PPO)
        all_rewards = []
        discounted_reward = 0
        #Calculul reward-urilor reduse (G_t= r_t + gama*G_{t+1}
        for reward, done in zip(reversed(self.rollout_buffer.rewards), reversed(self.rollout_buffer.dones)):
            if done: #Când întâlnește un episod terminat (s-a generat o planificare completa)
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            all_rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32).to(device)
        if len(all_rewards)>1:
            all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-7)
            #all_rewards = (2*(all_rewards + 2200)/(2200 + 1e-7 )-1).float()

        # Pregătirea batch-urilor
        batch_size = self.batch_size
        batches = DataLoader(self.rollout_buffer.states, batch_size=batch_size)

        #old-policy
        all_actions = torch.squeeze(torch.stack(self.rollout_buffer.actions, dim=0)).detach().to(device)
        all_logprobs = torch.squeeze(torch.stack(self.rollout_buffer.logprobs, dim=0)).detach().to(device)

        all_losses = 0
        all_losses_cri = 0
        # Optimize policy for N epochs - Face mai multe treceri (epochs) peste aceleași date
        for epoch in range(self.n_epochs):

            # Evaluating old actions and values
            i = 0 #MUTAT i=0 aici ca e sinc cu buffer, de ce ar tine de epoch
            #print("batches", len(batches), "i=", i)
            for batch in batches:
                # print("!!!batch", batch, )
                batch = batch.to(device)
                old_actions = all_actions[i*batch_size: (i+1)*batch_size]
                old_logprobs = all_logprobs[i*batch_size: (i+1)*batch_size]
                rewards = all_rewards[i*batch_size: (i+1)*batch_size]

                #Reevaluarea politicii și a criticului
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch, old_actions) #FM-roxana
                state_values = torch.squeeze(state_values)

                #Calculul raportului PPO și al avantajelor
                ## Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())
                ## Finding Surrogate Loss
                advantages = rewards - state_values.detach()

                #Critic loss + Surrogate loss pentru actor (PPO clip)
                all_losses_cri += 0.5*self.MseLoss(state_values, rewards).mean()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip_range, 1+self.clip_range) * advantages
                # final loss of clipped objective PPO
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


        print('agent_training/loss', float(all_losses))
        print( 'agent_training/actor_loss', float(all_losses)/(i*self.n_epochs))
        print('agent_training/critic_loss', float(all_losses_cri)/(i*self.n_epochs))
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
        print("save() function called {file}.pkl")

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

        print("PPOGNN-load()")
        with open(f"{file}.pkl", "rb") as handle:
            data = pickle.load(handle)

        print("PPOGNN-load() data")
        #
        env = data["params"]["env"]
        print("PPOGNN-load() 1before model update")
        print("PPOGNN-load()",data.keys(), env.tasks)


        for task in env.tasks:
            print("load() task", task.task_id, task.done)


        print("PPOGNN-load()", config)


        metadata = env.get_metadata()
        print("PPOGNN-load() 2before model update")

        # create PPO object, commit necessary parameters. Update remaining parameters
        model = cls(env=env, config=config, logger=logger, metadata=metadata)
        print("PPOGNN-load() before model update")
        model.__dict__.update(data["params"])

        print("PPOGNN-load() model", model)

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
        #Funcția learn este  antrenare RL: face loop peste instanțe de FJSP, colectează experiență
        # (state–action–reward), o pune în rollout_buffer și, din când în când, cheamă self.train() ca să facă update agent.
        enter_train = 0

        print("total_instances", total_instances, "self.total_instances", self.total_instances)
        # iterate over n episodes = the agents has n episodes to interact with the environment
        for episode in range(self.total_instances): #Fiecare instanță corespunde unui episod: o planificare completă FJSP.
            state = self.env.reset()
            done = False

            # run agent on env until done
            while not done:
                initial_state = copy.deepcopy(state)
                action, prob = self.predict(state=state, deterministic=True) #agentPPO.act(state)
                new_state, reward, done, info = self.env.step(action)
                self.num_timesteps += 1
                #Stocarea tranziției în rollout buffer
                self.rollout_buffer.store_memory(initial_state, action, prob, reward, done)#FM - altfel face copie dupa prima iteratie si nu e ok, se modifica ceva in nr de muchii

                # #call intermediate_test on_step - da eroare de verificat ce nu merge
                # if intermediate_test:
                #     intermediate_test.on_step(self.num_timesteps, episode, self)

                # break learn if total_timesteps are reached
                if self.num_timesteps >= total_timesteps:
                    print("total_timesteps reached")
                    self.logger.record(
                        {
                            'results_on_train_dataset/instances': episode,
                            'results_on_train_dataset/num_timesteps': self.num_timesteps
                        }
                    )
                    self.logger.dump()

                    return None

                if self.update_strategy == 'buffer_size' and self.num_timesteps % self.rollout_steps == 0:
                    print("train(): -- rollout buffer size", len(self.rollout_buffer.states))
                    self.train()  # este partea de update de la echeveria - agentPPO.update
                    enter_train += 1

                state = new_state

            #dupa 10 episoade (instante planificate complet se chema train care: calculează discounted rewards,
            # optimizează actorul + criticul (PPO), golește buffer-ul, copiază policy → policy_old.
            # !!update-ul se face după ce ai colectat X tranziții (state, action, reward), nu după episoade.
            # !!se poate realiza in 2 moduri dupa numarul de episoade (conform chatgpt intre 5-10 instante)
            # !!sau dupa numarul de trazitii (1000–3000);
            #self.rollout_steps = 10
            if self.update_strategy == 'episodes_number' and episode % self.rollout_episodes == 0:
                # train networks
                print("train(): -- rollout buffer", len(self.rollout_buffer.states))
                self.train() #este partea de update de la echeveria - agentPPO.update
                enter_train+=1

            #print('Is the schedule valid? Answer: ', self.env.is_asp_schedule_valid(False))

            #Când s-a trecut o dată prin toate instanțele se calculeaza: reward mediu, makespan mediu, tardiness mediu,
            if episode % len(self.env.data) == len(self.env.data) - 1:
                mean_training_reward = np.mean(self.env.episodes_rewards)
                print("self.env.episodes_makespans", self.env.episodes_makespans)
                mean_training_makespan = np.mean(self.env.episodes_makespans)
                if len(self.env.episodes_tardinesses) == 0:
                    mean_training_tardiness = 0
                else:
                    mean_training_tardiness = np.mean(self.env.episodes_tardinesses)

                print("mean_training_makespan", mean_training_makespan)
                self.logger.record(
                    {
                        'results_on_train_dataset/mean_reward': mean_training_reward,
                        'results_on_train_dataset/mean_makespan': mean_training_makespan,
                        'results_on_train_dataset/mean_tardiness': mean_training_tardiness
                    }
                )
                self.logger.dump()

                # ===== EARLY STOPPING pe baza makespan-ului =====
                if mean_training_makespan < self.best_makespan - 1e-6:
                    # îmbunătățire
                    self.best_makespan = mean_training_makespan
                    self.no_improve_counter = 0
                    print(f"[ES] Îmbunătățire: best_makespan = {self.best_makespan:.2f}")
                else:
                    # fără îmbunătățire
                    self.no_improve_counter += 1
                    print(f"[ES] Fără îmbunătățire ({self.no_improve_counter}/{self.patience})")

                # condiție de oprire timpurie
                if self.no_improve_counter >= self.patience:
                    print("[ES] Early stopping declanșat – nu mai există îmbunătățiri.")
                    print("TRAINING DONE (early stop)")
                    self.logger.record(
                        {
                            'results_on_train_dataset/instances': episode,
                            'results_on_train_dataset/num_timesteps': self.num_timesteps
                        }
                    )
                    self.logger.dump()
                    break

        print("TRAINING DONE")
        self.logger.record(
            {
                'results_on_train_dataset/instances': episode,
                'results_on_train_dataset/num_timesteps': self.num_timesteps
            }
        )
        print("...train() function was called", enter_train, 'times and select action ', self.num_timesteps, 'times')
        self.logger.dump()
