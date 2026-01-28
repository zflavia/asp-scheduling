Train

python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp/config_ASP_TUBES_ORIGINAL_GNN.yaml

python3 -m src.agents.train -fp training/ppo_gnn/config_ASP_TUBES_ORIGINAL_GNN.yaml

--Testing

python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp/config_ASP_TUBES_ORIGINAL_GNN_TESTING.yaml

python3 -m src.agents.test -fp testing/ppo/config_ASP_TUBES_TESTING_ORIGINAL_GNN.yaml

- fac un script cu 2 comenzi +  drepturi cu chmode

----- GP

    python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp-sets/config_dataset_2asp.yaml
    
    python3 -m src.agents.train -fp training/gp/config_gp_ASP_variants_tubes.yaml 
     
    python3 -m src.agents.test -fp testing/gp/config_gp_ASP_variants_tubes.yaml


------------
Generate datesets
 python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp-sets/config_dataset_deep.yaml
 python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp-sets/config_dataset_wide.yaml
 python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp-sets/config_dataset_dyuthi.yaml
 python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp-sets/config_dataset_2asp.yaml
 python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp-sets/config_dataset_fjsspHurinkVdata.yaml


python3 -m src.agents.train -fp training/gp/trainn.gp-aos.yaml

-----------------------
 python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp-sets/config_dataset_fjsspHurinkVdata.yaml
 python3 -m src.data_generator.bom_instance_factory -fp data_generation/asp-sets/config_dataset_train_ga.yaml

 python3 -m src.agents.train -fp training/gp/config_gp_pair_train_ga.yaml 
 python3 -m src.agents.train -fp training/gp/config_gp_dr_train_ga.yaml 


 python3 -m src.agents.test -fp testing/gp/config_gp_dr_train_ga_best.yaml


