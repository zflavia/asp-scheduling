Train

python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp/config_ASP_TUBES_ORIGINAL_GNN.yaml

python3 -m src.agents.train -fp training/ppo_gnn/config_ASP_TUBES_ORIGINAL_GNN.yaml
